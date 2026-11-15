/* Offgrid: all image edits are local. Only explicit live analysis sends pixels. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const MAX_BYTES = 10 * 1024 * 1024;
  const MAX_PIXELS = 50_000_000;
  const MAX_EDGE = 1600;
  const MAX_EDITS = 12;
  const COLORS = { original: '#a9e5cf', edited: '#7f9cff', truth: '#f38d7e', baseline: '#b1a1db' };
  // Descriptions are of pixels, not locations; unknown manifest IDs get neutral labels.
  const DESCRIPTIONS = {
    '889133601646428': ['A road through the trees', 'Road markings & forest'],
    '144906930925908': ['Under open skies', 'Power lines & roadside'],
    '939483846864661': ['Around the shaded bend', 'Vegetation & road surface']
  };
  const state = {
    status: null, example: null, original: null, work: null, originalBlob: null,
    operations: [], view: 'original', tool: null, selection: null, pointer: null,
    originalPrediction: null, editedPrediction: null, historicalPrediction: null,
    liveOriginalPrediction: null, liveOriginalStatusKey: null, comparisonIssue: '', truthVisible: false,
    revision: 0, loadToken: 0, requestToken: 0, busy: false, loading: false,
    controller: null, ticker: null, map: null, markerLayer: null, tiles: null,
    tileGeneration: 0, geographicReady: false, sourceDescription: '', metadata: 'unchecked',
    presenterElapsed: 0, presenterStarted: null, presenterTimer: null, geographicLabels: null
  };
  const isNumber = n => typeof n === 'number' && Number.isFinite(n);
  const point = (lat, lon) => isNumber(lat) && isNumber(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180 ? { lat, lon } : null;
  const canPredict = () => state.status && (typeof state.status.can_predict === 'boolean'
    ? state.status.can_predict : state.status.mode === 'live' && state.status.cuda_available === true);
  const knownTruth = () => state.example && point(state.example.true_lat, state.example.true_lon);
  const formatNumber = (value, digits = 2) => isNumber(value) ? value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
  const formatPoint = p => `${Math.abs(p.lat).toFixed(4)}° ${p.lat < 0 ? 'S' : 'N'}, ${Math.abs(p.lon).toFixed(4)}° ${p.lon < 0 ? 'W' : 'E'}`;
  const formatDistance = km => `${formatNumber(km, km < 10 ? 2 : 1)} km`;
  function distance(a, b) {
    const r = Math.PI / 180;
    const h = Math.sin((b.lat - a.lat) * r / 2) ** 2 +
      Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin((b.lon - a.lon) * r / 2) ** 2;
    return 6371.0088 * 2 * Math.atan2(Math.sqrt(Math.max(0, Math.min(1, h))), Math.sqrt(Math.max(0, 1 - h)));
  }
  function configurationKey(config) {
    if (!config || !Number.isInteger(config.top_n) || !config.tower_weights ||
        typeof config.tower_weights !== 'object' || Array.isArray(config.tower_weights)) return null;
    const weights = Object.entries(config.tower_weights).sort(([a], [b]) => a.localeCompare(b));
    if (!weights.length || weights.some(([, value]) => !isNumber(value))) return null;
    return JSON.stringify([config.top_n, weights, config.variant ?? null, config.calibrated ?? null]);
  }
  function statusConfigurationKey() {
    const config = state.status?.prediction_config;
    return JSON.stringify([configurationKey(config), config?.model_revision ?? null, config?.retrieval_k ?? null]);
  }
  function cachedLiveOriginal() {
    return state.liveOriginalPrediction && configurationKey(state.liveOriginalPrediction.config) &&
      state.liveOriginalStatusKey === statusConfigurationKey() ? state.liveOriginalPrediction : null;
  }
  function matchedComparison(original, edited) {
    return original?.source === 'live' && edited?.source === 'live' &&
      configurationKey(original.config) !== null && configurationKey(original.config) === configurationKey(edited.config);
  }
  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function announce(message) { $('announcer').textContent = message; }
  function showError(message) {
    $('error-message').textContent = message || '';
    $('error-message').hidden = !message;
  }
  function describe(example, index = 0) {
    return DESCRIPTIONS[String(example.id)] || [`Street scene ${index + 1}`, 'Look for visible clues'];
  }
  function recordedPoint(example) {
    if (!example) return null;
    return point(example.prediction_lat, example.prediction_lon) ||
      (example.deployed === 'candidate' ? point(example.selected_lat, example.selected_lon)
        : example.deployed === 'baseline' ? point(example.baseline_lat, example.baseline_lon) : null);
  }
  function makeCanvas(width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
  }
  function exportPNG(canvas) {
    return new Promise((resolve, reject) => canvas.toBlob(blob => {
      if (!blob) reject(new Error('The browser could not export this image. Try a smaller photo.'));
      else resolve(blob);
    }, 'image/png'));
  }
  function sameOriginImage(url) {
    if (typeof url !== 'string') throw new Error('The recorded image is missing. Choose another example or your own photo.');
    const parsed = new URL(url, window.location.href);
    if (parsed.origin !== window.location.origin || !['http:', 'https:'].includes(parsed.protocol)) {
      throw new Error('The example image must be served by this demo.');
    }
    return parsed.href;
  }

  async function loadStatus() {
    $('retry-status').disabled = true;
    try {
      const response = await fetch('/api/status', { signal: AbortSignal.timeout(15000), cache: 'no-store' });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      if (!data || typeof data !== 'object' || !['recorded', 'live'].includes(data.mode)) throw new Error('Invalid status response');
      state.status = data;
      $('mode-badge').querySelector('b').textContent = canPredict() ? 'Live inference' : 'Recorded examples';
      $('mode-badge').classList.toggle('live', Boolean(canPredict()));
      $('service-notice').hidden = true;
      $('retry-status').hidden = true;
      if (data.mode === 'live' && !canPredict()) {
        $('service-message').textContent = `Live inference is unavailable${data.reason ? `: ${String(data.reason)}` : ' on this server'}. Recorded results and local editing still work.`;
        $('service-notice').hidden = false;
      }
      renderExamples();
      renderReport();
      if (!state.original && !state.loading) {
        const first = (Array.isArray(data.examples) ? data.examples : []).find(e => e && e.image_url);
        if (first) await loadExample(first);
      }
    } catch (error) {
      state.status = null;
      $('mode-badge').querySelector('b').textContent = 'Connection unavailable';
      $('mode-badge').classList.remove('live');
      $('service-message').textContent = 'The demo service could not be reached. Choose a photo to inspect, edit and download locally, or try connecting again.';
      $('service-notice').hidden = false;
      $('retry-status').hidden = false;
      renderExamples();
      renderReport();
    } finally {
      $('retry-status').disabled = false;
      updateControls();
    }
  }
  function renderExamples() {
    const holder = $('example-list');
    holder.replaceChildren();
    const examples = Array.isArray(state.status?.examples) ? state.status.examples.filter(e => e && typeof e === 'object').slice(0, 3) : [];
    if (!examples.length) {
      holder.append(element('p', 'empty-examples', state.status
        ? 'No recorded examples are available. Choose a photo to inspect metadata, cover a detail, or make a crop.'
        : 'Recorded examples are unavailable while the service is disconnected. Your own photos can still be edited locally.'));
      return;
    }
    examples.forEach((example, index) => {
      const [title, subtitle] = describe(example, index);
      const button = element('button', 'example-card');
      button.type = 'button';
      button.setAttribute('aria-pressed', String(state.example === example));
      button.dataset.index = String(index);
      const image = element('img');
      image.alt = title;
      image.loading = 'lazy';
      try { image.src = sameOriginImage(example.image_url); }
      catch (_) { image.alt = 'Image unavailable'; }
      image.addEventListener('error', () => { image.alt = 'Image unavailable'; });
      const copy = element('span', 'example-copy');
      copy.append(element('strong', '', title), element('span', '', subtitle));
      const check = element('span', 'example-check', '✓');
      check.setAttribute('aria-hidden', 'true');
      button.append(image, copy, check);
      button.addEventListener('click', () => loadExample(example));
      holder.append(button);
    });
  }

  // Read dimensions before decoding: prevent giant allocations from small compressed files.
  // This inspects only container headers, not EXIF; exifr handles original metadata below.
  function inspectImageHeader(buffer) {
    const b = new Uint8Array(buffer), v = new DataView(buffer);
    const text = (start, length) => String.fromCharCode(...b.subarray(start, start + length));
    if (b.length >= 24 && b[0] === 137 && text(1, 3) === 'PNG' && text(12, 4) === 'IHDR') {
      return { type: 'image/png', width: v.getUint32(16), height: v.getUint32(20) };
    }
    if (b.length >= 12 && text(0, 4) === 'RIFF' && text(8, 4) === 'WEBP') {
      for (let at = 12; at + 8 <= b.length;) {
        const name = text(at, 4), size = v.getUint32(at + 4, true), data = at + 8;
        if (data + size > b.length) break;
        const uint24 = pos => b[pos] | (b[pos + 1] << 8) | (b[pos + 2] << 16);
        if (name === 'VP8X' && size >= 10) return { type: 'image/webp', width: uint24(data + 4) + 1, height: uint24(data + 7) + 1 };
        if (name === 'VP8L' && size >= 5 && b[data] === 0x2f) {
          const bits = v.getUint32(data + 1, true);
          return { type: 'image/webp', width: (bits & 0x3fff) + 1, height: ((bits >>> 14) & 0x3fff) + 1 };
        }
        if (name === 'VP8 ' && size >= 10 && b[data + 3] === 0x9d && b[data + 4] === 1 && b[data + 5] === 0x2a) {
          return { type: 'image/webp', width: v.getUint16(data + 6, true) & 0x3fff, height: v.getUint16(data + 8, true) & 0x3fff };
        }
        at = data + size + (size % 2);
      }
    }
    if (b.length > 4 && b[0] === 0xff && b[1] === 0xd8) {
      const sof = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
      let at = 2;
      while (at + 3 < b.length) {
        if (b[at++] !== 0xff) break;
        while (b[at] === 0xff) at++;
        const marker = b[at++];
        if (marker === 0xda || marker === 0xd9) break;
        if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
        if (at + 2 > b.length) break;
        const size = v.getUint16(at);
        if (size < 2 || at + size > b.length) break;
        if (sof.has(marker) && size >= 8) return { type: 'image/jpeg', width: v.getUint16(at + 5), height: v.getUint16(at + 3) };
        at += size;
      }
    }
    throw new Error('This file is not a readable JPEG, PNG or WebP. Choose a supported still image.');
  }
  async function inspectMetadata(buffer, token) {
    let label = 'Metadata not checked', note = 'Metadata inspection is unavailable. This does not mean GPS tags are absent.';
    state.metadata = 'unchecked';
    try {
      if (!window.exifr) throw new Error('Metadata parser unavailable');
      const tags = await window.exifr.parse(buffer, { gps: true, xmp: true, tiff: true, iptc: false, icc: false });
      const gps = tags && Object.entries(tags).some(([key, value]) =>
        /^(GPS(Latitude|Longitude|Position)|latitude$|longitude$)/i.test(key) && value !== undefined && value !== null);
      label = gps ? 'GPS metadata found' : 'No GPS metadata found';
      note = gps ? 'The original contains location tags. Canvas exports remove them.'
        : 'The pixels may still carry location clues.';
      if (token === state.loadToken) state.metadata = gps ? 'found' : 'absent';
    } catch (_) { /* A parse failure is not evidence that metadata is absent. */ }
    if (token !== state.loadToken) return;
    $('metadata-label').textContent = label;
    $('metadata-note').textContent = note;
  }
  async function decodeImage(blob) {
    if ('createImageBitmap' in window) {
      // The browser applies EXIF orientation exactly once. Do not manually rotate again.
      try { return await createImageBitmap(blob, { imageOrientation: 'from-image' }); }
      catch (_) { /* Older browser codecs may support HTMLImageElement instead. */ }
    }
    const url = URL.createObjectURL(blob);
    try {
      const image = new Image();
      image.src = url;
      await image.decode();
      return image;
    } finally { URL.revokeObjectURL(url); }
  }
  function beginLoad() {
    state.loadToken++;
    state.revision++;
    state.loading = true;
    showError('');
    updateControls();
    return state.loadToken;
  }
  async function setImage(blob, example, title, token) {
    if (blob.size > MAX_BYTES) throw new Error('This image exceeds 10 MiB. Choose a smaller JPEG, PNG or WebP.');
    if (!blob.size) throw new Error('This image is empty. Choose another photo.');
    const buffer = await blob.arrayBuffer();
    const header = inspectImageHeader(buffer);
    if (!header.width || !header.height || header.width * header.height > MAX_PIXELS) {
      throw new Error('This image exceeds 50 million pixels or has invalid dimensions. Resize it before choosing it here.');
    }
    const decoded = await decodeImage(new Blob([buffer], { type: header.type }));
    if (token !== state.loadToken) { decoded.close?.(); return; }
    const width = decoded.width || decoded.naturalWidth, height = decoded.height || decoded.naturalHeight;
    if (!width || !height || width * height > MAX_PIXELS) {
      decoded.close?.();
      throw new Error('This image has invalid dimensions or exceeds 50 million pixels.');
    }
    const scale = Math.min(1, MAX_EDGE / Math.max(width, height));
    const original = makeCanvas(Math.max(1, Math.round(width * scale)), Math.max(1, Math.round(height * scale)));
    const context = original.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, original.width, original.height);
    context.drawImage(decoded, 0, 0, original.width, original.height);
    decoded.close?.();
    // A single cached normalized original is used for every custom original comparison.
    const originalBlob = await exportPNG(original);
    if (token !== state.loadToken) return;
    state.example = example;
    state.original = original;
    state.originalBlob = originalBlob;
    state.operations = [];
    state.view = 'original';
    state.tool = null;
    state.selection = null;
    state.pointer = null;
    state.originalPrediction = null;
    state.editedPrediction = null;
    state.historicalPrediction = null;
    state.liveOriginalPrediction = null;
    state.liveOriginalStatusKey = null;
    state.comparisonIssue = '';
    state.truthVisible = false;
    state.sourceDescription = scale < 1
      ? `Resized from ${width} × ${height} to ${original.width} × ${original.height} for all live analysis${example ? ' · recorded evaluation uses different source pixels' : ''}.`
      : `${original.width} × ${original.height} px${example ? ' · recorded evaluation / display copy' : ' · re-encoded pixels for analysis'}`;
    $('photo-title').textContent = title;
    $('photo-canvas').setAttribute('aria-label', `${title}. Photo editor; choose Cover or Crop below. Keyboard presets are available.`);
    $('baseline-toggle').checked = false;
    $('download-note').hidden = true;
    $('image-empty').hidden = true;
    $('photo-canvas').hidden = false;
    $('stage-tag').hidden = false;
    rebuildImage(false);
    renderResults();
    $('metadata-label').textContent = 'Metadata not checked';
    $('metadata-note').textContent = 'Inspecting the original locally…';
    inspectMetadata(buffer, token);
    announce(`${title} loaded. The estimate is hidden.`);
  }
  async function loadExample(example) {
    if (state.busy || state.loading) return;
    const token = beginLoad();
    try {
      const response = await fetch(sameOriginImage(example.image_url), { signal: AbortSignal.timeout(20000) });
      if (!response.ok) throw new Error('This recorded photo could not be loaded. Choose another example or your own photo.');
      const length = Number(response.headers.get('content-length'));
      if (length > MAX_BYTES) throw new Error('This example exceeds the 10 MiB image limit.');
      const examples = Array.isArray(state.status?.examples) ? state.status.examples : [];
      await setImage(await response.blob(), example, describe(example, examples.indexOf(example))[0], token);
    } catch (error) {
      if (token === state.loadToken) {
        showError(error.name === 'TimeoutError' ? 'The photo took too long to load. Try this example again.' : error.message);
      }
    } finally {
      if (token === state.loadToken) { state.loading = false; updateControls(); }
    }
  }
  async function loadFile(file) {
    if (!file || state.busy || state.loading) return;
    const token = beginLoad();
    try { await setImage(file, null, 'Your photo', token); }
    catch (error) {
      if (token === state.loadToken) {
        showError(error.message || 'The browser could not open this photo. Try another image.');
      }
    } finally {
      if (token === state.loadToken) { state.loading = false; updateControls(); }
    }
  }

  function rebuildImage(invalidate = true) {
    if (!state.original) return;
    let work = makeCanvas(state.original.width, state.original.height);
    work.getContext('2d').drawImage(state.original, 0, 0);
    for (const operation of state.operations) {
      const { x, y, width, height } = operation.rect;
      if (operation.type === 'cover') {
        const context = work.getContext('2d');
        context.fillStyle = '#152139';
        context.fillRect(x, y, width, height);
      } else {
        const cropped = makeCanvas(width, height);
        cropped.getContext('2d').drawImage(work, x, y, width, height, 0, 0, width, height);
        work = cropped;
      }
    }
    state.work = work;
    state.selection = null;
    state.pointer = null;
    if (invalidate) {
      state.revision++;
      state.editedPrediction = null;
      state.comparisonIssue = '';
      $('download-note').hidden = true;
      renderResults();
    }
    drawPhoto();
    updateControls();
  }
  function drawPhoto() {
    if (!state.original) return;
    const source = state.view === 'original' ? state.original : state.work;
    const canvas = $('photo-canvas');
    canvas.width = source.width;
    canvas.height = source.height;
    const context = canvas.getContext('2d');
    context.drawImage(source, 0, 0);
    if (state.selection && state.view === 'edited') {
      const { x, y, width, height } = state.selection;
      const lineScale = source.width / Math.max(1, canvas.getBoundingClientRect().width);
      if (state.tool === 'crop') {
        context.fillStyle = '#101e3499';
        context.beginPath();
        context.rect(0, 0, source.width, source.height);
        context.rect(x, y, width, height);
        context.fill('evenodd');
      } else {
        context.fillStyle = '#152139b3';
        context.fillRect(x, y, width, height);
      }
      context.strokeStyle = '#fff';
      context.lineWidth = 2 * lineScale;
      context.setLineDash([6 * lineScale, 4 * lineScale]);
      context.strokeRect(x, y, width, height);
      context.setLineDash([]);
    }
    $('stage-tag').textContent = state.view === 'original' ? 'Original photo'
      : state.operations.length ? 'Edited pixels' : 'Editing view';
  }
  function chooseTool(tool) {
    if (!state.work || state.busy || state.loading) return;
    state.tool = state.tool === tool ? null : tool;
    state.selection = null;
    state.pointer = null;
    if (state.tool) state.view = 'edited';
    drawPhoto();
    updateControls();
  }
  function presetSelection() {
    if (!state.tool || !state.work || state.busy || state.loading) return;
    const w = state.work.width, h = state.work.height;
    state.selection = state.tool === 'crop'
      ? { x: Math.round(w * .1), y: Math.round(h * .1), width: Math.round(w * .8), height: Math.round(h * .8) }
      : { x: Math.round(w * .3), y: Math.round(h * .35), width: Math.round(w * .4), height: Math.round(h * .3) };
    drawPhoto();
    updateControls();
    announce('Rectangle selected. Use Apply to commit this edit.');
  }
  function lowerCenterSelection() {
    if (!state.work || state.busy || state.loading || state.operations.length >= MAX_EDITS) return;
    state.tool = 'cover';
    state.view = 'edited';
    const w = state.work.width, h = state.work.height;
    // Inclusive pixel range: 318..591, 204..511 for the 910 × 512 display export.
    const x = Math.floor(w * .35), y = Math.floor(h * .4);
    state.selection = { x, y, width: Math.floor(w * .65) - x + 1, height: h - y };
    drawPhoto();
    updateControls();
    announce('Lower center selected. Apply cover to hide those pixels.');
  }
  function applySelection() {
    if (!state.tool || !state.selection || state.busy || state.loading) return;
    if (state.operations.length >= MAX_EDITS) { showError('This photo has 12 edits. Undo an edit or reset before adding another.'); return; }
    const { x, y, width, height } = state.selection;
    const rect = { x: Math.round(x), y: Math.round(y), width: Math.round(width), height: Math.round(height) };
    rect.width = Math.min(rect.width, state.work.width - rect.x);
    rect.height = Math.min(rect.height, state.work.height - rect.y);
    const minimum = state.tool === 'crop' ? 32 : 2;
    if (rect.width < minimum || rect.height < minimum) {
      showError(`Select an area at least ${minimum} × ${minimum} pixels${state.tool === 'crop' ? ' to keep' : ''}.`);
      return;
    }
    showError('');
    state.operations.push({ type: state.tool, rect });
    state.view = 'edited';
    rebuildImage();
    announce(`${state.tool === 'crop' ? 'Crop' : 'Opaque cover'} applied. Any previous edited estimate has been cleared.`);
  }
  function setView(view) {
    if (!state.original || state.busy || state.loading) return;
    state.view = view;
    state.tool = null;
    state.selection = null;
    drawPhoto();
    updateControls();
  }
  function editorPoint(event) {
    const rect = $('photo-canvas').getBoundingClientRect();
    return {
      x: Math.min(state.work.width, Math.max(0, (event.clientX - rect.left) * state.work.width / rect.width)),
      y: Math.min(state.work.height, Math.max(0, (event.clientY - rect.top) * state.work.height / rect.height))
    };
  }
  function startPointer(event) {
    if (!state.tool || state.view !== 'edited' || state.busy || state.loading || (event.button !== 0 && event.pointerType === 'mouse') || state.pointer) return;
    event.preventDefault();
    state.pointer = { id: event.pointerId, start: editorPoint(event) };
    state.selection = null;
    $('photo-canvas').setPointerCapture(event.pointerId);
    updateControls();
  }
  function movePointer(event) {
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    const p = editorPoint(event), start = state.pointer.start;
    state.selection = { x: Math.min(p.x, start.x), y: Math.min(p.y, start.y), width: Math.abs(p.x - start.x), height: Math.abs(p.y - start.y) };
    drawPhoto();
  }
  function endPointer(event) {
    if (!state.pointer || state.pointer.id !== event.pointerId) return;
    if (event.type === 'pointercancel') state.selection = null;
    else movePointer(event);
    state.pointer = null;
    if ($('photo-canvas').hasPointerCapture(event.pointerId)) $('photo-canvas').releasePointerCapture(event.pointerId);
    updateControls();
    drawPhoto();
  }
  async function downloadImage() {
    if (!state.work || state.busy || state.loading) return;
    $('download-button').disabled = true;
    const revision = state.revision;
    try {
      // Always export the working bitmap, never the display canvas with selection overlays.
      const blob = await exportPNG(state.work);
      if (revision !== state.revision) return;
      const url = URL.createObjectURL(blob);
      const link = element('a');
      link.href = url;
      link.download = 'offgrid-edited.png';
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      $('download-note').textContent = 'PNG exported from edited pixels without original metadata. Visual clues can remain.';
      $('download-note').hidden = false;
      announce('Edited PNG downloaded. Original metadata is not copied into the export.');
    } catch (error) { showError(error.message); }
    finally { updateControls(); }
  }

  function updateControls() {
    const locked = state.busy || state.loading;
    const hasImage = Boolean(state.original);
    const hasEdits = state.operations.length > 0;
    $('upload-button').disabled = locked;
    $('file-input').disabled = locked;
    $('reset-story').disabled = locked || !hasImage;
    document.querySelectorAll('.example-card').forEach(button => {
      button.disabled = locked;
      const examples = (state.status?.examples || []).filter(e => e && typeof e === 'object');
      button.setAttribute('aria-pressed', String(examples[Number(button.dataset.index)] === state.example));
    });
    for (const id of ['cover-tool', 'crop-tool', 'original-view', 'download-button']) $(id).disabled = locked || !hasImage;
    $('cover-tool').disabled ||= state.operations.length >= MAX_EDITS;
    $('crop-tool').disabled ||= state.operations.length >= MAX_EDITS;
    $('edited-view').disabled = locked || !hasImage;
    $('undo-button').disabled = locked || !hasEdits;
    $('reset-edits').disabled = locked || !hasEdits;
    $('preset-button').disabled = locked;
    $('road-preset').disabled = locked || !hasImage || state.operations.length >= MAX_EDITS;
    $('road-preset').hidden = state.tool !== 'cover';
    $('apply-tool').disabled = locked || !state.selection || Boolean(state.pointer) || state.operations.length >= MAX_EDITS;
    $('cancel-tool').disabled = locked;
    $('original-view').setAttribute('aria-pressed', String(state.view === 'original'));
    $('edited-view').setAttribute('aria-pressed', String(state.view === 'edited'));
    $('cover-tool').setAttribute('aria-pressed', String(state.tool === 'cover'));
    $('crop-tool').setAttribute('aria-pressed', String(state.tool === 'crop'));
    $('tool-options').hidden = !state.tool;
    $('apply-tool').textContent = state.tool === 'crop' ? 'Apply crop' : 'Apply cover';
    $('preset-button').textContent = state.tool === 'crop' ? 'Trim outer 10%' : 'Select center';
    $('tool-help').textContent = state.tool === 'crop'
      ? 'Drag to select the area to keep, or trim the edges with a preset. Then apply the crop.'
      : 'Drag to select a detail, or select the center. Apply an opaque cover to hide those pixels.';
    $('photo-canvas').classList.toggle('editing', Boolean(state.tool) && !locked);
    $('photo-canvas').setAttribute('aria-disabled', String(locked));
    $('pixel-note').textContent = state.sourceDescription || 'JPEG, PNG or WebP · up to 10 MiB / 50 MP';
    $('edit-count').textContent = hasEdits ? `${state.operations.length} / ${MAX_EDITS} edits · ${state.work.width} × ${state.work.height} px`
      : 'Edits stay in your browser';
    const hasOriginal = Boolean(state.originalPrediction);
    const needsEdited = hasOriginal && hasEdits;
    const savedAvailable = Boolean(state.example && recordedPoint(state.example));
    const label = !hasOriginal
      ? state.example ? (hasEdits ? 'Reveal original estimate' : 'Reveal estimate') : (hasEdits ? 'Analyze original photo' : 'Analyze photo')
      : needsEdited ? (state.editedPrediction ? 'Edited photo analyzed' : 'Analyze edited photo') : 'Original estimate revealed';
    $('analyze-label').textContent = state.loading ? 'Loading photo…' : state.busy ? 'Analyzing photo…' : label;
    $('analyze-button').disabled = locked || !hasImage || (hasOriginal && (!hasEdits || Boolean(state.editedPrediction)))
      || (!hasOriginal && state.example && !savedAvailable) || ((!state.example || needsEdited) && !canPredict());
    $('truth-button').hidden = !hasOriginal || !knownTruth();
    $('truth-button').disabled = locked;
    $('truth-button').textContent = state.truthVisible ? 'Hide known location' : 'Reveal known location';
    $('truth-button').setAttribute('aria-pressed', String(state.truthVisible));
    $('wait-state').hidden = !state.busy;
    $('baseline-toggle').disabled = !state.example || !hasOriginal || locked || !point(state.example.baseline_lat, state.example.baseline_lon);
    let note;
    if (!hasImage) note = 'Choose a photo to start. Local inspection and editing do not require live inference.';
    else if (hasEdits && !canPredict()) note = 'Live inference required to analyze your edits. Editing and PNG downloads still work locally.';
    else if (!state.example && !canPredict()) note = 'Live inference required to analyze your photo. You can still inspect metadata, edit and download locally.';
    else if (state.example && !hasOriginal) note = savedAvailable
      ? 'Saved evaluation result. This photo is a display export, not the original inference input.'
      : 'No saved estimate is available for this original. Local editing and downloads still work.';
    else if (!hasOriginal) note = 'Analysis sends re-encoded pixels to the server, without original metadata. Only analyze photos you have permission to use.';
    else if (state.comparisonIssue) note = state.comparisonIssue;
    else if (needsEdited && !state.editedPrediction) note = cachedLiveOriginal()
      ? 'Reuses the live original; analyzes the edited PNG. Results are compared only when model settings match.'
      : 'Analyzes the original pixels first, then the edited PNG. Only matched live results are compared.';
    else if (state.editedPrediction) note = 'Estimate moved measures the distance between predictions. It is not an accuracy gain or a privacy score.';
    else note = 'Try a cover or crop to test a visual clue. The original estimate stays available for comparison.';
    $('action-note').textContent = note;
    $('action-note').dataset.presenterNote = !hasOriginal && state.example ? 'Saved evaluation · display copy, not a fresh inference.'
      : state.editedPrediction ? 'Matched live pixels. Movement is not accuracy or proof of privacy.'
      : needsEdited ? (canPredict() ? 'Matched pixel comparison. Original first, then edited.' : 'Live inference required. Local editing still works.')
      : note;
    if (document.body.classList.contains('presentation')) $('action-note').textContent = state.comparisonIssue || $('action-note').dataset.presenterNote;
    document.querySelectorAll('[data-cue="photo"],[data-cue="reveal"],[data-cue="edit"]').forEach(button => button.disabled = locked);
  }

  async function analyze() {
    if ($('analyze-button').disabled || state.busy || state.loading) return;
    showError('');
    if (!state.originalPrediction && state.example) {
      const saved = recordedPoint(state.example);
      if (!saved) { showError('This example has no saved location estimate.'); return; }
      state.historicalPrediction = { ...saved, source: 'recorded' };
      state.originalPrediction = state.historicalPrediction;
      renderResults();
      updateControls();
      announce('Recorded estimate revealed. The known location is still hidden.');
      return;
    }
    if (!canPredict()) return;
    const wantsEdited = Boolean(state.originalPrediction && state.operations.length);
    let baseline = wantsEdited ? cachedLiveOriginal() : null;
    const needsOriginal = !baseline;
    const revision = state.revision, token = ++state.requestToken;
    const originalBody = state.originalBlob;
    // Freeze both inputs before the first request; busy guards also prevent all editing.
    const editedSnapshot = wantsEdited ? makeCanvas(state.work.width, state.work.height) : null;
    editedSnapshot?.getContext('2d').drawImage(state.work, 0, 0);
    state.busy = true;
    state.editedPrediction = null;
    state.comparisonIssue = '';
    state.controller = new AbortController();
    const controller = state.controller;
    const started = performance.now();
    let timedOut = false;
    let stage = 'Preparing image pixels';
    const current = () => token === state.requestToken && revision === state.revision && !controller.signal.aborted;
    const tick = () => {
      const elapsed = Math.floor((performance.now() - started) / 1000);
      $('wait-message').textContent = `${stage} · ${elapsed}s elapsed`;
      $('continue-evidence').hidden = elapsed < 20;
    };
    tick();
    state.ticker = setInterval(tick, 1000);
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 180000);
    updateControls();
    renderResults();
    try {
      const editedBody = editedSnapshot ? await exportPNG(editedSnapshot) : null;
      if (originalBody.size > MAX_BYTES || editedBody?.size > MAX_BYTES) throw new Error('The exported pixels exceed 10 MiB. Crop or choose a smaller photo before analysis.');
      const predict = async (body, label) => {
        if (!current()) return null;
        stage = label;
        tick();
        const response = await fetch('/api/predict', {
          method: 'POST', headers: { 'Content-Type': 'image/png' }, body, signal: controller.signal
        });
        let payload;
        try { payload = await response.json(); }
        catch (_) { throw new Error(`The server returned an unreadable response (HTTP ${response.status}). No new estimate was accepted.`); }
        if (!response.ok) {
          const detail = typeof payload?.detail === 'string' ? payload.detail : `HTTP ${response.status}`;
          throw new Error(response.status === 409 ? `The model is busy. Wait a moment, then try again. ${detail}` : `Analysis failed: ${detail}`);
        }
        const predicted = point(payload.lat, payload.lon);
        if (!predicted) throw new Error('The server did not return valid coordinates. No new estimate was accepted.');
        return current() ? { ...predicted, source: 'live', config: payload } : null;
      };
      if (needsOriginal) {
        baseline = await predict(originalBody, wantsEdited ? 'Analyzing original pixels (1 of 2)' : 'Analyzing original pixels');
        if (!baseline || !current()) return;
        state.liveOriginalPrediction = baseline;
        state.liveOriginalStatusKey = statusConfigurationKey();
        state.originalPrediction = baseline;
        renderResults();
      }
      if (wantsEdited) {
        if (!configurationKey(baseline.config)) {
          state.comparisonIssue = 'The original response did not include verifiable model settings. No edit comparison is shown.';
          throw new Error(state.comparisonIssue);
        }
        const edited = await predict(editedBody, needsOriginal ? 'Analyzing edited pixels (2 of 2)' : 'Analyzing edited pixels');
        if (!edited || !current()) return;
        state.originalPrediction = baseline;
        if (!matchedComparison(baseline, edited)) {
          state.liveOriginalPrediction = null; // Retry must refresh the original under the new settings.
          state.comparisonIssue = 'Model settings differ or are incomplete. These results cannot be compared. Retry to analyze both images again.';
          throw new Error(state.comparisonIssue);
        }
        state.editedPrediction = edited;
      }
      renderResults();
      announce(wantsEdited ? 'Matched live original and edited estimates are ready.' : 'Live original estimate ready.');
    } catch (error) {
      if (token !== state.requestToken || revision !== state.revision) return;
      showError(error.name === 'AbortError'
        ? timedOut ? 'Analysis timed out after 180 seconds. No edit comparison is shown. The server may still be finishing the request.'
          : 'Stopped waiting. The server may still be finishing. Any completed live original is retained; no edit comparison is shown.'
        : error.message || 'Analysis failed. No edit comparison is shown.');
    } finally {
      clearTimeout(timeout);
      if (token === state.requestToken) {
        clearInterval(state.ticker);
        state.busy = false;
        state.controller = null;
        $('continue-evidence').hidden = true;
        renderResults();
        updateControls();
      }
    }
  }
  function cancelRequest() {
    if (!state.busy) return;
    state.controller?.abort();
    state.requestToken++; // Even a response already in the event queue cannot replace current state.
    clearInterval(state.ticker);
    state.busy = false;
    state.controller = null;
    state.editedPrediction = null;
    $('continue-evidence').hidden = true;
    showError('Stopped waiting. The server may still be finishing. Any completed live original is retained; no edit comparison is shown.');
    renderResults();
    updateControls();
    announce('Analysis canceled. No new estimate was accepted.');
  }
  function renderResults() {
    const original = state.originalPrediction;
    const edited = matchedComparison(original, state.editedPrediction) ? state.editedPrediction : null;
    $('workbench').classList.toggle('has-comparison', Boolean(edited));
    $('results').hidden = !original;
    $('result-empty').hidden = Boolean(original);
    $('map-curtain').hidden = Boolean(original);
    const list = $('coordinate-list');
    list.replaceChildren();
    const points = [];
    const row = (key, label, p) => {
      const wrapper = element('div', 'coordinate-row'), dt = element('dt');
      dt.append(element('i', `pin-dot ${key}`), document.createTextNode(label));
      wrapper.append(dt, element('dd', '', formatPoint(p)));
      list.append(wrapper);
      points.push({ key, label, ...p });
    };
    if (original) {
      $('result-provenance').textContent = original.source === 'recorded'
        ? `Recorded evaluation · ${state.example?.deployed === 'candidate' ? 'deployed candidate' : state.example?.deployed === 'baseline' ? 'retained baseline' : 'saved prediction'}`
        : edited ? 'Live pixel comparison · matched model settings' : 'Live original · normalized display pixels';
      row('original', original.source === 'live' ? 'Live original' : 'Recorded estimate', original);
      if (edited) row('edited', 'Live edited', edited);
      if (state.truthVisible && knownTruth()) row('truth', 'Known location', knownTruth());
      if ($('baseline-toggle').checked && state.example) {
        const baseline = point(state.example.baseline_lat, state.example.baseline_lon);
        if (baseline) points.push({ key: 'baseline', label: 'Recorded baseline', ...baseline });
      }
    }
    const metrics = $('distance-result');
    metrics.replaceChildren();
    const metric = (value, label, className = '') => {
      const node = element('div', className);
      node.append(element('strong', '', formatDistance(value)), element('span', '', label));
      metrics.append(node);
    };
    if (original && edited) metric(distance(original, edited), 'Estimate moved · between predictions');
    if (original && state.truthVisible && knownTruth()) {
      metric(distance(original, knownTruth()), 'Original error · distance from known location', 'truth-error');
      if (edited) metric(distance(edited, knownTruth()), 'Edited error · distance from known location', 'truth-error');
    }
    metrics.hidden = !metrics.childElementCount;
    $('historical-result').textContent = state.historicalPrediction
      ? `Historical recorded evaluation: ${formatPoint(state.historicalPrediction)}${state.truthVisible && knownTruth() ? `; error ${formatDistance(distance(state.historicalPrediction, knownTruth()))}` : ''}. This result is not used in the live edit comparison.`
      : 'Reveal a recorded example to inspect its historical evaluation result.';
    $('live-settings').textContent = original?.source === 'live'
      ? `Current live original: top_n ${original.config.top_n ?? 'not reported'}; tower weights ${Object.entries(original.config.tower_weights || {}).map(([key, value]) => `${key}: ${value}`).join(', ') || 'not reported'}.${edited ? ' The live edited response has matching settings.' : ' No completed matched edit comparison is shown.'} The historical evaluation report above is unchanged.`
      : 'No live pixel analysis in this story yet.';
    $('baseline-note').textContent = original && $('baseline-toggle').checked && state.example
      ? `Recorded baseline: ${formatPoint(point(state.example.baseline_lat, state.example.baseline_lon))}. A model comparison, not a known location.`
      : 'Reveal an example’s estimate first. The baseline is a model comparison, not a known location.';
    drawMap(points);
  }

  function initMap() {
    if (!window.L) { useFallbackMap('Map library unavailable · schematic world overview'); return; }
    try {
      state.map = L.map('map', { zoomControl: true, attributionControl: true, worldCopyJump: true, scrollWheelZoom: false, minZoom: 1, maxZoom: 17 });
      state.map.attributionControl.setPrefix(false);
      state.map.setView([22, 15], 1);
      state.map.attributionControl.addAttribution('Natural Earth');
      // Keep the local geographic fallback beneath optional raster tiles, not over roads.
      state.map.createPane('local-world');
      state.map.getPane('local-world').style.zIndex = '150';
      state.map.getPane('local-world').style.pointerEvents = 'none';
      state.markerLayer = L.layerGroup().addTo(state.map);
      state.geographicLabels = L.layerGroup().addTo(state.map);
      state.map.on('zoomend', renderGeographicLabels);
      renderGeographicLabels();
      for (let lat = -60; lat <= 60; lat += 30) {
        L.polyline([[lat, -180], [lat, 180]], { pane: 'local-world', color: '#2b3c55', weight: .6, interactive: false }).addTo(state.map);
      }
      for (let lon = -180; lon <= 180; lon += 30) {
        L.polyline([[-80, lon], [80, lon]], { pane: 'local-world', color: '#2b3c55', weight: .6, interactive: false }).addTo(state.map);
      }
      fetch('/assets/land-110m.geojson').then(response => {
        if (!response.ok) throw new Error('Local world map missing');
        return response.json();
      }).then(land => {
        if (!state.map) return;
        L.geoJSON(land, { pane: 'local-world', style: { color: '#405572', weight: .7, fillColor: '#273b54', fillOpacity: 1 }, interactive: false }).addTo(state.map).bringToBack();
        state.geographicReady = true;
      }).catch(() => {
        if (!state.tiles) {
          useFallbackMap('Local outlines unavailable · schematic world overview');
          renderResults();
        }
      });
    } catch (_) { useFallbackMap('Map unavailable · schematic world overview'); }
  }
  function useFallbackMap(note) {
    state.map?.remove();
    state.map = null;
    $('map').hidden = true;
    $('map-fallback').removeAttribute('hidden');
    $('tiles-button').disabled = true;
    $('map-note').textContent = note;
  }
  function renderGeographicLabels() {
    if (!state.map || !state.geographicLabels) return;
    state.geographicLabels.clearLayers();
    if (state.tiles) return;
    // Fixed atlas context, never reverse-geocoded claims about a prediction.
    const labels = state.map.getZoom() < 3
      ? [['North America', 48, -105], ['South America', -15, -60], ['Europe', 52, 16], ['Africa', 8, 20], ['Asia', 42, 100], ['Australia', -25, 134]]
      : [['Russia', 62, 85], ['Kazakhstan', 48, 67], ['China', 36, 103], ['India', 22, 79],
        ['Mongolia', 47, 103], ['Finland', 64, 26], ['Ukraine', 49, 32], ['Türkiye', 39, 35],
        ['France', 47, 2], ['Spain', 40, -4], ['Germany', 51, 10], ['Italy', 43, 12],
        ['Canada', 57, -110], ['United States', 39, -101], ['Mexico', 23, -103],
        ['Brazil', -12, -53], ['Argentina', -36, -65], ['Australia', -25, 134],
        ['South Africa', -29, 25], ['Egypt', 27, 30], ['Algeria', 28, 3]];
    labels.forEach(([name, lat, lon]) => {
      L.marker([lat, lon], {
        icon: L.divIcon({ className: 'geographic-label', html: element('span', '', name), iconSize: [100, 18], iconAnchor: [50, 9] }),
        interactive: false, keyboard: false
      }).addTo(state.geographicLabels);
    });
  }
  function drawMap(points) {
    $('map').setAttribute('aria-label', points.length
      ? `Location map. ${points.map(p => `${p.label}: ${formatPoint(p)}`).join('. ')}.`
      : 'World map. No location estimate revealed.');
    if (!state.map) {
      const group = $('fallback-pins');
      group.replaceChildren();
      for (const p of points) {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', String((p.lon + 180) * 2));
        circle.setAttribute('cy', String((90 - p.lat) * 2));
        circle.setAttribute('r', '5');
        circle.setAttribute('fill', COLORS[p.key]);
        circle.setAttribute('stroke', '#fff');
        circle.setAttribute('stroke-width', '1.5');
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = `${p.label}: ${formatPoint(p)}`;
        circle.append(title);
        group.append(circle);
      }
      return;
    }
    state.markerLayer.clearLayers();
    state.map.invalidateSize({ animate: false });
    if (!points.length) { state.map.setView([22, 15], 1, { animate: false }); return; }
    // Cut at the largest longitude gap, so markers near ±180° fit across the dateline.
    const lons = points.map(p => (p.lon + 360) % 360).sort((a, b) => a - b);
    let gap = -1, start = lons[0];
    for (let i = 0; i < lons.length; i++) {
      const next = lons[(i + 1) % lons.length] + (i === lons.length - 1 ? 360 : 0);
      if (next - lons[i] > gap) { gap = next - lons[i]; start = next % 360; }
    }
    const positions = points.map(p => {
      let lon = (p.lon + 360) % 360;
      if (lon < start) lon += 360;
      if (start > 180) lon -= 360;
      return [Math.max(-85, Math.min(85, p.lat)), lon];
    });
    points.forEach((p, index) => {
      const marker = L.circleMarker(positions[index], {
        radius: p.key === 'original' ? 7 : p.key === 'truth' ? 5 : 6, color: '#f4f6fa', weight: 1.5,
        fillColor: COLORS[p.key], fillOpacity: 1
      }).addTo(state.markerLayer);
      marker.bindTooltip(element('span', '', p.label), { direction: 'top' });
    });
    // The 1:110m local outline is useful at regional scale, not as an empty street map.
    const options = { animate: false, padding: [36, 35], maxZoom: state.tiles ? 12 : 3 };
    if (points.length === 1) state.map.setView(positions[0], state.tiles ? 8 : 3, { animate: false });
    else state.map.fitBounds(positions, options);
    if (points.some(p => Math.abs(p.lat) > 85)) $('map-note').textContent = 'Polar marker clamped at 85° · exact coordinates shown below';
  }
  function toggleTiles() {
    if (!state.map) return;
    const generation = ++state.tileGeneration;
    if (state.tiles) {
      state.map.removeLayer(state.tiles);
      state.tiles = null;
      $('tiles-button').setAttribute('aria-pressed', 'false');
      $('tiles-label').textContent = 'Enable street map';
      renderGeographicLabels();
      $('map-note').textContent = state.geographicReady ? 'Local world map · no external requests' : 'Local coordinate grid · no external requests';
      renderResults();
      return;
    }
    $('tiles-button').setAttribute('aria-pressed', 'true');
    $('tiles-label').textContent = 'Use local overview';
    $('map-note').textContent = 'Loading external OpenStreetMap tiles…';
    let loaded = false, errors = 0;
    state.tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 17, attribution: '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>',
      crossOrigin: false
    });
    state.tiles.on('tileload', () => {
      if (generation !== state.tileGeneration) return;
      loaded = true;
      $('map-note').textContent = errors ? 'Some street tiles unavailable · coordinates remain valid' : 'OpenStreetMap · external tiles enabled';
    });
    state.tiles.on('tileerror', () => {
      if (generation !== state.tileGeneration) return;
      errors++;
      $('map-note').textContent = loaded ? 'Some street tiles unavailable · coordinates remain valid' : 'Map tiles unavailable · local overview remains';
    });
    state.tiles.addTo(state.map);
    renderGeographicLabels();
    setTimeout(() => {
      if (generation === state.tileGeneration && state.tiles && !loaded) $('map-note').textContent = 'Map tiles unavailable · local overview remains';
    }, 10000);
    renderResults();
  }

  function renderReport() {
    const holder = $('report-content');
    holder.replaceChildren();
    const report = state.status?.report, config = state.status?.prediction_config;
    if (!report || typeof report !== 'object') {
      holder.append(element('p', 'report-scope', 'The evaluation report is unavailable. No performance figures are shown. Local editing still works.'));
      return;
    }
    const accepted = report.accept_candidate === true, rejected = report.accept_candidate === false;
    const baseline = report.baseline || {}, selected = report.selected || {};
    const deployed = accepted ? selected : rejected ? baseline : null;
    const grid = element('div', 'report-grid'), tableWrap = element('div', 'table-scroll'), table = element('table', 'report-table');
    const caption = element('caption', 'sr-only', 'Evaluation metrics derived from the server report');
    const head = element('thead'), header = element('tr');
    ['Metric', 'Baseline', accepted ? 'Deployed candidate' : rejected ? 'Deployed baseline' : 'Deployment unknown', ...(!accepted ? ['Selected candidate'] : [])]
      .forEach(text => header.append(element('th', '', text)));
    Array.from(header.children).forEach(th => th.scope = 'col');
    head.append(header);
    const body = element('tbody');
    [['geoscore', 'GeoScore', ''], ['acc@25km', 'Within 25 km', '%'], ['acc@1km', 'Within 1 km', '%']].forEach(([key, label, unit]) => {
      const row = element('tr'), name = element('th', '', label);
      name.scope = 'row';
      row.append(name);
      const values = [baseline[key], deployed?.[key], ...(!accepted ? [selected[key]] : [])];
      values.forEach((value, index) => row.append(element('td', index > 0 && isNumber(value) && isNumber(baseline[key]) && value < baseline[key] ? 'decline' : '',
        `${formatNumber(value, 2)}${isNumber(value) ? unit : ''}`)));
      body.append(row);
    });
    table.append(caption, head, body);
    tableWrap.append(table);
    grid.append(tableWrap);
    const facts = element('dl', 'report-facts');
    const fact = (label, value) => facts.append(element('dt', '', label), element('dd', '', value));
    fact('Evaluation queries', formatNumber(report.evaluation_queries, 0));
    fact('Evaluation sequences', formatNumber(report.evaluation_sequences, 0));
    fact('Candidate GeoScore delta', formatNumber(report.mean_geoscore_delta, 4));
    const interval = report.paired_sequence_bootstrap_95pct;
    fact('Paired sequence 95% interval', Array.isArray(interval) && interval.length === 2 ? `[${formatNumber(interval[0], 4)}, ${formatNumber(interval[1], 4)}]` : 'Not reported');
    if (Array.isArray(report.paired_sequence_bootstrap_adjusted)) {
      fact(`Adjusted ${formatNumber(report.familywise_confidence_percent, 2)}% interval`, `[${report.paired_sequence_bootstrap_adjusted.map(n => formatNumber(n, 4)).join(', ')}]`);
    }
    fact('Deployed top_n', config?.top_n != null ? String(config.top_n) : 'Not reported');
    grid.append(facts);
    holder.append(grid);
    holder.append(element('p', 'report-decision', `${accepted ? 'Candidate accepted and deployed.' : rejected ? 'Candidate rejected; baseline retained.' : 'Deployment decision not reported.'}${report.decision ? ` Report decision: ${String(report.decision)}.` : ''}`));
    if (isNumber(selected['acc@1km']) && isNumber(baseline['acc@1km']) && selected['acc@1km'] < baseline['acc@1km']) {
      holder.append(element('p', 'report-scope', `The selected candidate’s within-1-km rate declines from ${formatNumber(baseline['acc@1km'], 2)}% to ${formatNumber(selected['acc@1km'], 2)}%. A higher aggregate GeoScore does not mean every location threshold improves.`));
    }
    if (report.scope) holder.append(element('p', 'report-scope', `Report scope: ${String(report.scope)}`));
    const weights = values => values && typeof values === 'object'
      ? Object.entries(values).map(([key, value]) => `${key}: ${value}`).join(', ') || 'Not reported' : 'Not reported';
    holder.append(element('p', 'report-weights', `Deployed tower weights — ${weights(config?.tower_weights)}. Selected candidate weights — ${weights(report.selected_tower_weights)}. Configuration: ${config?.kind || 'not reported'}.`));
    holder.append(element('p', 'report-scope', 'Percentages are evaluation hit rates, not confidence in this photo. The paired interval describes the candidate-minus-baseline GeoScore difference; it is not a location radius.'));
  }
  function resetStory() {
    if (state.busy || state.loading || !state.original) return;
    state.operations = [];
    state.originalPrediction = null;
    state.editedPrediction = null;
    state.historicalPrediction = null;
    state.liveOriginalPrediction = null;
    state.liveOriginalStatusKey = null;
    state.comparisonIssue = '';
    state.truthVisible = false;
    state.view = 'original';
    state.tool = null;
    $('baseline-toggle').checked = false;
    showError('');
    rebuildImage();
    const first = (state.status?.examples || []).find(example => example?.image_url);
    if (first && state.example !== first) loadExample(first);
    announce('Story reset to the first available scene. Estimates and truth are hidden; live comparison cache cleared.');
  }
  function presenterSeconds() {
    return state.presenterElapsed + (state.presenterStarted === null ? 0 : (performance.now() - state.presenterStarted) / 1000);
  }
  function renderPresenterTimer() {
    const seconds = Math.floor(presenterSeconds());
    $('presenter-time').textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
    $('presenter-bar').classList.toggle('time-warning', seconds >= 285);
    $('timer-note').textContent = seconds >= 300 ? 'Five minutes reached' : seconds >= 295 ? 'Close with takeaway' : seconds >= 285 ? 'Wrap up · target 4:55' : 'Close by 4:55';
    $('timer-toggle').textContent = state.presenterStarted !== null ? 'Pause' : seconds ? 'Resume' : 'Start timer';
  }
  function togglePresenterTimer() {
    if (state.presenterStarted !== null) {
      state.presenterElapsed = presenterSeconds();
      state.presenterStarted = null;
      clearInterval(state.presenterTimer);
    } else {
      state.presenterStarted = performance.now();
      state.presenterTimer = setInterval(renderPresenterTimer, 250);
    }
    renderPresenterTimer();
  }
  function resetPresenterTimer() {
    clearInterval(state.presenterTimer);
    state.presenterElapsed = 0;
    state.presenterStarted = null;
    renderPresenterTimer();
  }
  function markCue(cue) {
    document.querySelectorAll('[data-cue]').forEach(button => {
      if (button.dataset.cue === cue) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
    });
  }
  function openEvidence(stoppedWaiting = false) {
    markCue('evidence');
    if (document.body.classList.contains('presentation')) {
      const body = document.querySelector('.evidence-body');
      $('evidence-slot').append(body);
      $('evidence-wait-note').hidden = !stoppedWaiting;
      if (!$('evidence-dialog').open) $('evidence-dialog').showModal();
    } else {
      $('evidence').open = true;
      $('evidence').scrollIntoView({ block: 'start' });
      $('evidence').querySelector('summary').focus({ preventScroll: true });
    }
  }
  function goToCue(cue) {
    markCue(cue);
    if (cue === 'evidence') { openEvidence(); return; }
    if (cue === 'takeaway') { $('takeaway-dialog').showModal(); return; }
    const target = cue === 'edit' ? $('cover-tool') : cue === 'reveal'
      ? (!$('analyze-button').disabled ? $('analyze-button') : !$('truth-button').hidden ? $('truth-button') : $('workbench')) : $('photo-canvas');
    target.focus({ preventScroll: true });
    if (!document.body.classList.contains('presentation') || window.innerWidth < 1000) target.scrollIntoView({ block: 'center' });
  }

  $('upload-button').addEventListener('click', () => $('file-input').click());
  $('file-input').addEventListener('change', event => {
    const file = event.target.files?.[0];
    event.target.value = '';
    loadFile(file);
  });
  $('retry-status').addEventListener('click', loadStatus);
  $('cover-tool').addEventListener('click', () => chooseTool('cover'));
  $('crop-tool').addEventListener('click', () => chooseTool('crop'));
  $('preset-button').addEventListener('click', presetSelection);
  $('road-preset').addEventListener('click', lowerCenterSelection);
  $('apply-tool').addEventListener('click', applySelection);
  $('cancel-tool').addEventListener('click', () => { state.tool = null; state.selection = null; drawPhoto(); updateControls(); });
  $('undo-button').addEventListener('click', () => { if (state.busy || state.loading) return; state.operations.pop(); state.view = state.operations.length ? 'edited' : 'original'; rebuildImage(); announce('Last edit undone.'); });
  $('reset-edits').addEventListener('click', () => { if (state.busy || state.loading) return; state.operations = []; state.view = 'original'; state.tool = null; rebuildImage(); announce('All edits removed. The original estimate is unchanged.'); });
  $('original-view').addEventListener('click', () => setView('original'));
  $('edited-view').addEventListener('click', () => setView('edited'));
  $('photo-canvas').addEventListener('pointerdown', startPointer);
  $('photo-canvas').addEventListener('pointermove', movePointer);
  $('photo-canvas').addEventListener('pointerup', endPointer);
  $('photo-canvas').addEventListener('pointercancel', endPointer);
  $('photo-canvas').addEventListener('keydown', event => {
    if (event.key === 'Escape' && !state.busy) { state.selection = null; state.tool = null; drawPhoto(); updateControls(); }
  });
  $('download-button').addEventListener('click', downloadImage);
  $('analyze-button').addEventListener('click', analyze);
  $('cancel-request').addEventListener('click', cancelRequest);
  $('continue-evidence').addEventListener('click', () => { cancelRequest(); openEvidence(true); });
  $('truth-button').addEventListener('click', () => { if (state.busy) return; state.truthVisible = !state.truthVisible; renderResults(); updateControls(); announce(state.truthVisible ? 'Known location revealed. Error is distance, not confidence.' : 'Known location hidden.'); });
  $('baseline-toggle').addEventListener('change', renderResults);
  $('tiles-button').addEventListener('click', toggleTiles);
  $('reset-story').addEventListener('click', resetStory);
  $('presentation-button').addEventListener('click', () => {
    const enabled = document.body.classList.toggle('presentation');
    $('presentation-button').setAttribute('aria-pressed', String(enabled));
    updateControls();
    state.map?.invalidateSize({ animate: false });
    window.scrollTo(0, 0);
    announce(enabled ? 'Presentation mode on. Five-minute cues and timer available; evidence opens in a dialog.' : 'Presentation mode off.');
  });
  $('timer-toggle').addEventListener('click', togglePresenterTimer);
  $('timer-reset').addEventListener('click', resetPresenterTimer);
  document.querySelectorAll('[data-cue]').forEach(button => button.addEventListener('click', () => goToCue(button.dataset.cue)));
  $('evidence').querySelector('summary').addEventListener('click', event => {
    if (document.body.classList.contains('presentation')) { event.preventDefault(); openEvidence(); }
  });
  $('close-evidence').addEventListener('click', () => $('evidence-dialog').close());
  $('evidence-dialog').addEventListener('close', () => {
    const body = $('evidence-slot').querySelector('.evidence-body');
    if (body) $('evidence').append(body);
    $('evidence').open = false;
  });
  $('close-takeaway').addEventListener('click', () => $('takeaway-dialog').close());
  $('finish-takeaway').addEventListener('click', () => { $('takeaway-dialog').close(); $('workbench').focus({ preventScroll: true }); });
  $('how-button').addEventListener('click', () => $('how-dialog').showModal());
  $('close-how').addEventListener('click', () => $('how-dialog').close());
  $('start-button').addEventListener('click', () => { $('how-dialog').close(); $('workbench').focus(); });
  $('how-dialog').addEventListener('click', event => { if (event.target === $('how-dialog')) { const r = event.target.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close(); } });
  window.addEventListener('resize', () => { state.map?.invalidateSize({ animate: false }); });
  initMap();
  updateControls();
  loadStatus();
})();

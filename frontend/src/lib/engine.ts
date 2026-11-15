/* OffgridEngine — port of the verified demo/exposure.js state machine.
 * Holds all mutable state; React renders derived views via derive*.
 * notify() mirrors the original updateControls()/renderResults() calls. */
import exifr from 'exifr'
import type {
  ExampleRow,
  MapPoint,
  MetadataState,
  Operation,
  PredictPayload,
  Prediction,
  Rect,
  StatusResponse,
  ToolType,
  ViewMode,
} from './types'
import {
  configurationKey,
  distance,
  formatDistance,
  formatNumber,
  formatPoint,
  isNumber,
  point,
  statusConfigurationKey,
} from './geo'
import {
  COVER_FILL,
  MAX_BYTES,
  MAX_EDGE,
  MAX_EDITS,
  MAX_PIXELS,
  decodeImage,
  exportPNG,
  inspectImageHeader,
  makeCanvas,
  sameOriginImage,
} from './image'

// Descriptions are of pixels, not locations; unknown manifest IDs get neutral labels.
const DESCRIPTIONS: Record<string, [string, string]> = {
  '889133601646428': ['A road through the trees', 'Road markings & forest'],
  '144906930925908': ['Under open skies', 'Power lines & roadside'],
  '939483846864661': ['Around the shaded bend', 'Vegetation & road surface'],
}

export interface EngineState {
  status: StatusResponse | null
  statusFailed: boolean
  example: ExampleRow | null
  original: HTMLCanvasElement | null
  work: HTMLCanvasElement | null
  originalBlob: Blob | null
  operations: Operation[]
  view: ViewMode
  tool: ToolType | null
  selection: Rect | null
  pointer: { id: number; start: { x: number; y: number } } | null
  originalPrediction: Prediction | null
  editedPrediction: Prediction | null
  historicalPrediction: Prediction | null
  liveOriginalPrediction: Prediction | null
  liveOriginalStatusKey: string | null
  comparisonIssue: string
  truthVisible: boolean
  baselineOn: boolean
  revision: number
  loadToken: number
  requestToken: number
  busy: boolean
  loading: boolean
  controller: AbortController | null
  ticker: ReturnType<typeof setInterval> | null
  waitStage: string
  waitElapsed: number
  showContinueEvidence: boolean
  sourceDescription: string
  photoTitle: string
  metadata: MetadataState
  metadataLabel: string
  metadataNote: string
  error: string
  announcement: { seq: number; text: string }
  downloadNote: string
  downloadBusy: boolean
  presenter: boolean
  presenterElapsed: number
  presenterStarted: number | null
  presenterTimer: ReturnType<typeof setInterval> | null
  activeCue: string
  evidenceOpen: boolean
  evidenceStoppedWaiting: boolean
  takeawayOpen: boolean
  howOpen: boolean
}

export function initialState(): EngineState {
  return {
    status: null,
    statusFailed: false,
    example: null,
    original: null,
    work: null,
    originalBlob: null,
    operations: [],
    view: 'original',
    tool: null,
    selection: null,
    pointer: null,
    originalPrediction: null,
    editedPrediction: null,
    historicalPrediction: null,
    liveOriginalPrediction: null,
    liveOriginalStatusKey: null,
    comparisonIssue: '',
    truthVisible: false,
    baselineOn: false,
    revision: 0,
    loadToken: 0,
    requestToken: 0,
    busy: false,
    loading: false,
    controller: null,
    ticker: null,
    waitStage: 'Preparing image pixels',
    waitElapsed: 0,
    showContinueEvidence: false,
    sourceDescription: '',
    photoTitle: 'Your starting point',
    metadata: 'unchecked',
    metadataLabel: 'Metadata not checked',
    metadataNote: 'Location tags are only one part of the picture.',
    error: '',
    announcement: { seq: 0, text: '' },
    downloadNote: '',
    downloadBusy: false,
    presenter: false,
    presenterElapsed: 0,
    presenterStarted: null,
    presenterTimer: null,
    activeCue: 'photo',
    evidenceOpen: false,
    evidenceStoppedWaiting: false,
    takeawayOpen: false,
    howOpen: false,
  }
}

export const describe = (example: ExampleRow, index = 0): [string, string] =>
  DESCRIPTIONS[String(example.id)] || [`Street scene ${index + 1}`, 'Look for visible clues']

export const canPredict = (state: EngineState): boolean =>
  Boolean(
    state.status &&
      (typeof state.status.can_predict === 'boolean'
        ? state.status.can_predict
        : state.status.mode === 'live' && state.status.cuda_available === true),
  )

export const knownTruth = (state: EngineState) =>
  state.example ? point(state.example.true_lat, state.example.true_lon) : null

export function recordedPoint(example: ExampleRow | null) {
  if (!example) return null
  return (
    point(example.prediction_lat, example.prediction_lon) ||
    (example.deployed === 'candidate'
      ? point(example.selected_lat, example.selected_lon)
      : example.deployed === 'baseline'
        ? point(example.baseline_lat, example.baseline_lon)
        : null)
  )
}

export function matchedComparison(
  original: Prediction | null,
  edited: Prediction | null,
): edited is Prediction {
  return (
    original?.source === 'live' &&
    edited?.source === 'live' &&
    configurationKey(original.config) !== null &&
    configurationKey(original.config) === configurationKey(edited.config)
  )
}

export class OffgridEngine {
  state: EngineState = initialState()
  notify: () => void = () => {}

  private announce(text: string) {
    this.state.announcement = { seq: this.state.announcement.seq + 1, text }
  }

  private showError(message: string) {
    this.state.error = message || ''
  }

  private cachedLiveOriginal(): Prediction | null {
    return this.state.liveOriginalPrediction &&
      configurationKey(this.state.liveOriginalPrediction.config) &&
      this.state.liveOriginalStatusKey === statusConfigurationKey(this.state.status)
      ? this.state.liveOriginalPrediction
      : null
  }

  /* ---- status ---------------------------------------------------------- */

  async loadStatus() {
    const state = this.state
    state.statusFailed = false
    this.notify()
    try {
      const response = await fetch('/api/status', {
        signal: AbortSignal.timeout(15000),
        cache: 'no-store',
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const data = (await response.json()) as StatusResponse
      if (!data || typeof data !== 'object' || !['recorded', 'live'].includes(data.mode))
        throw new Error('Invalid status response')
      state.status = data
      if (!state.original && !state.loading) {
        const first = (Array.isArray(data.examples) ? data.examples : []).find(
          e => e && e.image_url,
        )
        this.notify()
        if (first) await this.loadExample(first)
      }
    } catch {
      state.status = null
      state.statusFailed = true
    }
    this.notify()
  }

  /* ---- image loading --------------------------------------------------- */

  private beginLoad(): number {
    const state = this.state
    state.loadToken++
    state.revision++
    state.loading = true
    this.showError('')
    this.notify()
    return state.loadToken
  }

  async loadExample(example: ExampleRow) {
    const state = this.state
    if (state.busy || state.loading) return
    const token = this.beginLoad()
    try {
      const response = await fetch(sameOriginImage(example.image_url), {
        signal: AbortSignal.timeout(20000),
      })
      if (!response.ok)
        throw new Error(
          'This recorded photo could not be loaded. Choose another example or your own photo.',
        )
      const length = Number(response.headers.get('content-length'))
      if (length > MAX_BYTES) throw new Error('This example exceeds the 10 MiB image limit.')
      const examples = Array.isArray(state.status?.examples) ? state.status.examples : []
      const blob = await response.blob()
      await this.setImage(blob, example, describe(example, examples.indexOf(example))[0], token)
    } catch (error) {
      if (token === state.loadToken) {
        this.showError(
          (error as Error).name === 'TimeoutError'
            ? 'The photo took too long to load. Try this example again.'
            : (error as Error).message,
        )
      }
    } finally {
      if (token === state.loadToken) {
        state.loading = false
        this.notify()
      }
    }
  }

  async loadFile(file: File | null | undefined) {
    const state = this.state
    if (!file || state.busy || state.loading) return
    const token = this.beginLoad()
    try {
      await this.setImage(file, null, 'Your photo', token)
    } catch (error) {
      if (token === state.loadToken) {
        this.showError(
          (error as Error).message || 'The browser could not open this photo. Try another image.',
        )
      }
    } finally {
      if (token === state.loadToken) {
        state.loading = false
        this.notify()
      }
    }
  }

  private async setImage(
    blob: Blob,
    example: ExampleRow | null,
    title: string,
    token: number,
  ) {
    const state = this.state
    if (blob.size > MAX_BYTES)
      throw new Error('This image exceeds 10 MiB. Choose a smaller JPEG, PNG or WebP.')
    if (!blob.size) throw new Error('This image is empty. Choose another photo.')
    const buffer = await blob.arrayBuffer()
    const header = inspectImageHeader(buffer)
    if (!header.width || !header.height || header.width * header.height > MAX_PIXELS) {
      throw new Error(
        'This image exceeds 50 million pixels or has invalid dimensions. Resize it before choosing it here.',
      )
    }
    const decoded = await decodeImage(new Blob([buffer], { type: header.type }))
    if (token !== state.loadToken) {
      ;(decoded as ImageBitmap).close?.()
      return
    }
    const width =
      (decoded as ImageBitmap).width || (decoded as HTMLImageElement).naturalWidth
    const height =
      (decoded as ImageBitmap).height || (decoded as HTMLImageElement).naturalHeight
    if (!width || !height || width * height > MAX_PIXELS) {
      ;(decoded as ImageBitmap).close?.()
      throw new Error('This image has invalid dimensions or exceeds 50 million pixels.')
    }
    const scale = Math.min(1, MAX_EDGE / Math.max(width, height))
    const original = makeCanvas(
      Math.max(1, Math.round(width * scale)),
      Math.max(1, Math.round(height * scale)),
    )
    const context = original.getContext('2d')!
    context.fillStyle = '#ffffff'
    context.fillRect(0, 0, original.width, original.height)
    context.drawImage(decoded as CanvasImageSource, 0, 0, original.width, original.height)
    ;(decoded as ImageBitmap).close?.()
    // A single cached normalized original is used for every custom original comparison.
    const originalBlob = await exportPNG(original)
    if (token !== state.loadToken) return
    state.example = example
    state.original = original
    state.originalBlob = originalBlob
    state.operations = []
    state.view = 'original'
    state.tool = null
    state.selection = null
    state.pointer = null
    state.originalPrediction = null
    state.editedPrediction = null
    state.historicalPrediction = null
    state.liveOriginalPrediction = null
    state.liveOriginalStatusKey = null
    state.comparisonIssue = ''
    state.truthVisible = false
    state.baselineOn = false
    state.sourceDescription =
      scale < 1
        ? `Resized from ${width} × ${height} to ${original.width} × ${original.height} for all live analysis${example ? ' · recorded evaluation uses different source pixels' : ''}.`
        : `${original.width} × ${original.height} px${example ? ' · recorded evaluation / display copy' : ' · re-encoded pixels for analysis'}`
    state.photoTitle = title
    state.downloadNote = ''
    this.rebuildImage(false)
    state.metadataLabel = 'Metadata not checked'
    state.metadataNote = 'Inspecting the original locally…'
    state.metadata = 'unchecked'
    this.inspectMetadata(buffer, token)
    this.announce(`${title} loaded. The estimate is hidden.`)
    this.notify()
  }

  private async inspectMetadata(buffer: ArrayBuffer, token: number) {
    const state = this.state
    let label = 'Metadata not checked'
    let note = 'Metadata inspection is unavailable. This does not mean GPS tags are absent.'
    state.metadata = 'unchecked'
    try {
      const tags = await exifr.parse(buffer, {
        gps: true,
        xmp: true,
        tiff: true,
        iptc: false,
        icc: false,
      })
      const gps =
        tags &&
        Object.entries(tags).some(
          ([key, value]) =>
            /^(GPS(Latitude|Longitude|Position)|latitude$|longitude$)/i.test(key) &&
            value !== undefined &&
            value !== null,
        )
      label = gps ? 'GPS metadata found' : 'No GPS metadata found'
      note = gps
        ? 'The original contains location tags. Canvas exports remove them.'
        : 'The pixels may still carry location clues.'
      if (token === state.loadToken) state.metadata = gps ? 'found' : 'absent'
    } catch {
      /* A parse failure is not evidence that metadata is absent. */
    }
    if (token !== state.loadToken) return
    state.metadataLabel = label
    state.metadataNote = note
    this.notify()
  }

  /* ---- canvas editing -------------------------------------------------- */

  rebuildImage(invalidate = true) {
    const state = this.state
    if (!state.original) return
    let work = makeCanvas(state.original.width, state.original.height)
    work.getContext('2d')!.drawImage(state.original, 0, 0)
    for (const operation of state.operations) {
      const { x, y, width, height } = operation.rect
      if (operation.type === 'cover') {
        const context = work.getContext('2d')!
        context.fillStyle = COVER_FILL
        context.fillRect(x, y, width, height)
      } else {
        const cropped = makeCanvas(width, height)
        cropped.getContext('2d')!.drawImage(work, x, y, width, height, 0, 0, width, height)
        work = cropped
      }
    }
    state.work = work
    state.selection = null
    state.pointer = null
    if (invalidate) {
      state.revision++
      state.editedPrediction = null
      state.comparisonIssue = ''
      state.downloadNote = ''
    }
    this.notify()
  }

  chooseTool(tool: ToolType) {
    const state = this.state
    if (!state.work || state.busy || state.loading) return
    state.tool = state.tool === tool ? null : tool
    state.selection = null
    state.pointer = null
    if (state.tool) state.view = 'edited'
    this.notify()
  }

  presetSelection() {
    const state = this.state
    if (!state.tool || !state.work || state.busy || state.loading) return
    const w = state.work.width
    const h = state.work.height
    state.selection =
      state.tool === 'crop'
        ? {
            x: Math.round(w * 0.1),
            y: Math.round(h * 0.1),
            width: Math.round(w * 0.8),
            height: Math.round(h * 0.8),
          }
        : {
            x: Math.round(w * 0.3),
            y: Math.round(h * 0.35),
            width: Math.round(w * 0.4),
            height: Math.round(h * 0.3),
          }
    this.announce('Rectangle selected. Use Apply to commit this edit.')
    this.notify()
  }

  lowerCenterSelection() {
    const state = this.state
    if (!state.work || state.busy || state.loading || state.operations.length >= MAX_EDITS)
      return
    state.tool = 'cover'
    state.view = 'edited'
    const w = state.work.width
    const h = state.work.height
    // Inclusive pixel range: 318..591, 204..511 for the 910 × 512 display export.
    const x = Math.floor(w * 0.35)
    const y = Math.floor(h * 0.4)
    state.selection = { x, y, width: Math.floor(w * 0.65) - x + 1, height: h - y }
    this.announce('Lower center selected. Apply cover to hide those pixels.')
    this.notify()
  }

  applySelection() {
    const state = this.state
    if (!state.tool || !state.selection || state.busy || state.loading) return
    if (state.operations.length >= MAX_EDITS) {
      this.showError('This photo has 12 edits. Undo an edit or reset before adding another.')
      this.notify()
      return
    }
    const { x, y, width, height } = state.selection
    const rect: Rect = {
      x: Math.round(x),
      y: Math.round(y),
      width: Math.round(width),
      height: Math.round(height),
    }
    rect.width = Math.min(rect.width, state.work!.width - rect.x)
    rect.height = Math.min(rect.height, state.work!.height - rect.y)
    const minimum = state.tool === 'crop' ? 32 : 2
    if (rect.width < minimum || rect.height < minimum) {
      this.showError(
        `Select an area at least ${minimum} × ${minimum} pixels${state.tool === 'crop' ? ' to keep' : ''}.`,
      )
      this.notify()
      return
    }
    this.showError('')
    state.operations.push({ type: state.tool, rect })
    state.view = 'edited'
    this.rebuildImage()
    this.announce(
      `${state.tool === 'crop' ? 'Crop' : 'Opaque cover'} applied. Any previous edited estimate has been cleared.`,
    )
  }

  setView(view: ViewMode) {
    const state = this.state
    if (!state.original || state.busy || state.loading) return
    state.view = view
    state.tool = null
    state.selection = null
    this.notify()
  }

  cancelTool() {
    const state = this.state
    state.tool = null
    state.selection = null
    this.notify()
  }

  undo() {
    const state = this.state
    if (state.busy || state.loading) return
    state.operations.pop()
    state.view = state.operations.length ? 'edited' : 'original'
    this.rebuildImage()
    this.announce('Last edit undone.')
  }

  resetEdits() {
    const state = this.state
    if (state.busy || state.loading) return
    state.operations = []
    state.view = 'original'
    state.tool = null
    this.rebuildImage()
    this.announce('All edits removed. The original estimate is unchanged.')
  }

  private editorPoint(canvas: HTMLCanvasElement, event: { clientX: number; clientY: number }) {
    const state = this.state
    const rect = canvas.getBoundingClientRect()
    return {
      x: Math.min(
        state.work!.width,
        Math.max(0, ((event.clientX - rect.left) * state.work!.width) / rect.width),
      ),
      y: Math.min(
        state.work!.height,
        Math.max(0, ((event.clientY - rect.top) * state.work!.height) / rect.height),
      ),
    }
  }

  startPointer(canvas: HTMLCanvasElement, event: React.PointerEvent<HTMLCanvasElement>) {
    const state = this.state
    if (
      !state.tool ||
      state.view !== 'edited' ||
      state.busy ||
      state.loading ||
      (event.button !== 0 && event.pointerType === 'mouse') ||
      state.pointer
    )
      return
    event.preventDefault()
    state.pointer = { id: event.pointerId, start: this.editorPoint(canvas, event) }
    state.selection = null
    canvas.setPointerCapture(event.pointerId)
    this.notify()
  }

  movePointer(canvas: HTMLCanvasElement, event: { clientX: number; clientY: number; pointerId: number }) {
    const state = this.state
    if (!state.pointer || state.pointer.id !== event.pointerId) return
    const p = this.editorPoint(canvas, event)
    const start = state.pointer.start
    state.selection = {
      x: Math.min(p.x, start.x),
      y: Math.min(p.y, start.y),
      width: Math.abs(p.x - start.x),
      height: Math.abs(p.y - start.y),
    }
    this.notify()
  }

  endPointer(canvas: HTMLCanvasElement, event: React.PointerEvent<HTMLCanvasElement>) {
    const state = this.state
    if (!state.pointer || state.pointer.id !== event.pointerId) return
    if (event.type === 'pointercancel') state.selection = null
    else this.movePointer(canvas, event)
    state.pointer = null
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
    this.notify()
  }

  escapeCanvas() {
    const state = this.state
    if (state.busy) return
    state.selection = null
    state.tool = null
    this.notify()
  }

  async downloadImage() {
    const state = this.state
    if (!state.work || state.busy || state.loading) return
    state.downloadBusy = true
    const revision = state.revision
    this.notify()
    try {
      // Always export the working bitmap, never the display canvas with selection overlays.
      const blob = await exportPNG(state.work)
      if (revision !== state.revision) return
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = 'offgrid-edited.png'
      document.body.append(link)
      link.click()
      link.remove()
      setTimeout(() => URL.revokeObjectURL(url), 10000)
      state.downloadNote =
        'PNG exported from edited pixels without original metadata. Visual clues can remain.'
      this.announce('Edited PNG downloaded. Original metadata is not copied into the export.')
    } catch (error) {
      this.showError((error as Error).message)
    } finally {
      state.downloadBusy = false
      this.notify()
    }
  }

  /* ---- analysis -------------------------------------------------------- */

  async analyze() {
    const state = this.state
    if (state.busy || state.loading) return
    this.showError('')
    if (!state.originalPrediction && state.example) {
      const saved = recordedPoint(state.example)
      if (!saved) {
        this.showError('This example has no saved location estimate.')
        this.notify()
        return
      }
      state.historicalPrediction = { ...saved, source: 'recorded' }
      state.originalPrediction = state.historicalPrediction
      this.announce('Recorded estimate revealed. The known location is still hidden.')
      this.notify()
      return
    }
    if (!canPredict(state)) return
    const wantsEdited = Boolean(state.originalPrediction && state.operations.length)
    let baseline = wantsEdited ? this.cachedLiveOriginal() : null
    const needsOriginal = !baseline
    const revision = state.revision
    const token = ++state.requestToken
    const originalBody = state.originalBlob!
    // Freeze both inputs before the first request; busy guards also prevent all editing.
    const editedSnapshot = wantsEdited
      ? makeCanvas(state.work!.width, state.work!.height)
      : null
    editedSnapshot?.getContext('2d')!.drawImage(state.work!, 0, 0)
    state.busy = true
    state.editedPrediction = null
    state.comparisonIssue = ''
    state.controller = new AbortController()
    const controller = state.controller
    const started = performance.now()
    let timedOut = false
    let stage = 'Preparing image pixels'
    const current = () =>
      token === state.requestToken && revision === state.revision && !controller.signal.aborted
    const tick = () => {
      const elapsed = Math.floor((performance.now() - started) / 1000)
      state.waitStage = stage
      state.waitElapsed = elapsed
      state.showContinueEvidence = elapsed >= 20
      this.notify()
    }
    tick()
    state.ticker = setInterval(tick, 1000)
    const timeout = setTimeout(() => {
      timedOut = true
      controller.abort()
    }, 180000)
    this.notify()
    try {
      const editedBody = editedSnapshot ? await exportPNG(editedSnapshot) : null
      if (originalBody.size > MAX_BYTES || (editedBody && editedBody.size > MAX_BYTES))
        throw new Error(
          'The exported pixels exceed 10 MiB. Crop or choose a smaller photo before analysis.',
        )
      const predict = async (body: Blob, label: string): Promise<Prediction | null> => {
        if (!current()) return null
        stage = label
        tick()
        const response = await fetch('/api/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'image/png' },
          body,
          signal: controller.signal,
        })
        let payload: PredictPayload | { detail?: string } | null
        try {
          payload = await response.json()
        } catch {
          throw new Error(
            `The server returned an unreadable response (HTTP ${response.status}). No new estimate was accepted.`,
          )
        }
        if (!response.ok) {
          const detail =
            typeof payload?.detail === 'string' ? payload.detail : `HTTP ${response.status}`
          throw new Error(
            response.status === 409
              ? `The model is busy. Wait a moment, then try again. ${detail}`
              : `Analysis failed: ${detail}`,
          )
        }
        const predicted = point(
          (payload as PredictPayload | null)?.lat,
          (payload as PredictPayload | null)?.lon,
        )
        if (!predicted)
          throw new Error(
            'The server did not return valid coordinates. No new estimate was accepted.',
          )
        return current()
          ? { ...predicted, source: 'live', config: payload as PredictPayload }
          : null
      }
      if (needsOriginal) {
        baseline = await predict(
          originalBody,
          wantsEdited ? 'Analyzing original pixels (1 of 2)' : 'Analyzing original pixels',
        )
        if (!baseline || !current()) return
        state.liveOriginalPrediction = baseline
        state.liveOriginalStatusKey = statusConfigurationKey(state.status)
        state.originalPrediction = baseline
        this.notify()
      }
      if (wantsEdited) {
        if (!configurationKey(baseline!.config)) {
          state.comparisonIssue =
            'The original response did not include verifiable model settings. No edit comparison is shown.'
          throw new Error(state.comparisonIssue)
        }
        const edited = await predict(
          editedBody!,
          needsOriginal ? 'Analyzing edited pixels (2 of 2)' : 'Analyzing edited pixels',
        )
        if (!edited || !current()) return
        state.originalPrediction = baseline
        if (!matchedComparison(baseline, edited)) {
          state.liveOriginalPrediction = null // Retry must refresh the original under the new settings.
          state.comparisonIssue =
            'Model settings differ or are incomplete. These results cannot be compared. Retry to analyze both images again.'
          throw new Error(state.comparisonIssue)
        }
        state.editedPrediction = edited
      }
      this.announce(
        wantsEdited
          ? 'Matched live original and edited estimates are ready.'
          : 'Live original estimate ready.',
      )
    } catch (error) {
      if (token === state.requestToken && revision === state.revision) {
        const err = error as Error
        this.showError(
          err.name === 'AbortError'
            ? timedOut
              ? 'Analysis timed out after 180 seconds. No edit comparison is shown. The server may still be finishing the request.'
              : 'Stopped waiting. The server may still be finishing. Any completed live original is retained; no edit comparison is shown.'
            : err.message || 'Analysis failed. No edit comparison is shown.',
        )
      }
    } finally {
      clearTimeout(timeout)
      if (token === state.requestToken) {
        if (state.ticker) clearInterval(state.ticker)
        state.ticker = null
        state.busy = false
        state.controller = null
        state.showContinueEvidence = false
        this.notify()
      }
    }
  }

  cancelRequest() {
    const state = this.state
    if (!state.busy) return
    state.controller?.abort()
    state.requestToken++ // Even a response already in the event queue cannot replace current state.
    if (state.ticker) clearInterval(state.ticker)
    state.ticker = null
    state.busy = false
    state.controller = null
    state.editedPrediction = null
    state.showContinueEvidence = false
    this.showError(
      'Stopped waiting. The server may still be finishing. Any completed live original is retained; no edit comparison is shown.',
    )
    this.announce('Analysis canceled. No new estimate was accepted.')
    this.notify()
  }

  toggleTruth() {
    const state = this.state
    if (state.busy) return
    state.truthVisible = !state.truthVisible
    this.announce(
      state.truthVisible
        ? 'Known location revealed. Error is distance, not confidence.'
        : 'Known location hidden.',
    )
    this.notify()
  }

  toggleBaseline() {
    this.state.baselineOn = !this.state.baselineOn
    this.notify()
  }

  resetStory() {
    const state = this.state
    if (state.busy || state.loading || !state.original) return
    state.operations = []
    state.originalPrediction = null
    state.editedPrediction = null
    state.historicalPrediction = null
    state.liveOriginalPrediction = null
    state.liveOriginalStatusKey = null
    state.comparisonIssue = ''
    state.truthVisible = false
    state.baselineOn = false
    state.view = 'original'
    state.tool = null
    this.showError('')
    this.rebuildImage()
    const first = (state.status?.examples || []).find(example => example?.image_url)
    if (first && state.example !== first) void this.loadExample(first)
    this.announce(
      'Story reset to the first available scene. Estimates and truth are hidden; live comparison cache cleared.',
    )
    this.notify()
  }

  /* ---- presenter -------------------------------------------------------- */

  presenterSeconds(): number {
    const state = this.state
    return (
      state.presenterElapsed +
      (state.presenterStarted === null ? 0 : (performance.now() - state.presenterStarted) / 1000)
    )
  }

  togglePresenterTimer() {
    const state = this.state
    if (state.presenterStarted !== null) {
      state.presenterElapsed = this.presenterSeconds()
      state.presenterStarted = null
      if (state.presenterTimer) clearInterval(state.presenterTimer)
      state.presenterTimer = null
    } else {
      state.presenterStarted = performance.now()
      state.presenterTimer = setInterval(() => this.notify(), 250)
    }
    this.notify()
  }

  resetPresenterTimer() {
    const state = this.state
    if (state.presenterTimer) clearInterval(state.presenterTimer)
    state.presenterTimer = null
    state.presenterElapsed = 0
    state.presenterStarted = null
    this.notify()
  }

  togglePresenter() {
    const state = this.state
    state.presenter = !state.presenter
    this.announce(
      state.presenter
        ? 'Presentation mode on. Five-minute cues and timer available; evidence opens in a dialog.'
        : 'Presentation mode off.',
    )
    this.notify()
  }

  markCue(cue: string) {
    this.state.activeCue = cue
    this.notify()
  }

  openEvidence(stoppedWaiting = false) {
    this.state.activeCue = 'evidence'
    this.state.evidenceStoppedWaiting = stoppedWaiting
    this.state.evidenceOpen = true
    this.notify()
  }

  openTakeaway() {
    this.state.activeCue = 'takeaway'
    this.state.takeawayOpen = true
    this.notify()
  }

  continueToEvidence() {
    this.cancelRequest()
    this.openEvidence(true)
  }

  setHowOpen(open: boolean) {
    this.state.howOpen = open
    this.notify()
  }

  setEvidenceOpen(open: boolean) {
    this.state.evidenceOpen = open
    this.notify()
  }

  setTakeawayOpen(open: boolean) {
    this.state.takeawayOpen = open
    this.notify()
  }
}

/* ---- derived view state (port of updateControls + renderResults) -------- */

export interface DerivedControls {
  locked: boolean
  hasImage: boolean
  hasEdits: boolean
  analyzeLabel: string
  analyzeDisabled: boolean
  truthHidden: boolean
  truthDisabled: boolean
  truthLabel: string
  actionNote: string
  presenterNote: string
  coverDisabled: boolean
  cropDisabled: boolean
  undoDisabled: boolean
  resetEditsDisabled: boolean
  editedViewDisabled: boolean
  originalViewDisabled: boolean
  applyDisabled: boolean
  applyLabel: string
  presetLabel: string
  presetDisabled: boolean
  roadPresetHidden: boolean
  roadPresetDisabled: boolean
  toolHelp: string
  pixelNote: string
  editCount: string
  stageTag: string
  baselineDisabled: boolean
  uploadDisabled: boolean
  resetStoryDisabled: boolean
  downloadDisabled: boolean
  modeLabel: string
  modeLive: boolean
  serviceMessage: string | null
  showRetry: boolean
}

export function deriveControls(state: EngineState): DerivedControls {
  const locked = state.busy || state.loading
  const hasImage = Boolean(state.original)
  const hasEdits = state.operations.length > 0
  const hasOriginal = Boolean(state.originalPrediction)
  const needsEdited = hasOriginal && hasEdits
  const savedAvailable = Boolean(state.example && recordedPoint(state.example))
  const analyzeLabel = !hasOriginal
    ? state.example
      ? hasEdits
        ? 'Reveal original estimate'
        : 'Reveal estimate'
      : hasEdits
        ? 'Analyze original photo'
        : 'Analyze photo'
    : needsEdited
      ? state.editedPrediction
        ? 'Edited photo analyzed'
        : 'Analyze edited photo'
      : 'Original estimate revealed'
  const analyzeDisabled =
    locked ||
    !hasImage ||
    (hasOriginal && (!hasEdits || Boolean(state.editedPrediction))) ||
    (!hasOriginal && state.example !== null && !savedAvailable) ||
    ((!state.example || needsEdited) && !canPredict(state))
  const truth = knownTruth(state)
  const truthHidden = !hasOriginal || !truth

  let note: string
  if (!hasImage)
    note = 'Choose a photo to start. Local inspection and editing do not require live inference.'
  else if (hasEdits && !canPredict(state))
    note =
      'Live inference required to analyze your edits. Editing and PNG downloads still work locally.'
  else if (!state.example && !canPredict(state))
    note =
      'Live inference required to analyze your photo. You can still inspect metadata, edit and download locally.'
  else if (state.example && !hasOriginal)
    note = savedAvailable
      ? 'Saved evaluation result. This photo is a display export, not the original inference input.'
      : 'No saved estimate is available for this original. Local editing and downloads still work.'
  else if (!hasOriginal)
    note =
      'Analysis sends re-encoded pixels to the server, without original metadata. Only analyze photos you have permission to use.'
  else if (state.comparisonIssue) note = state.comparisonIssue
  else if (needsEdited && !state.editedPrediction)
    note = state.liveOriginalPrediction &&
      configurationKey(state.liveOriginalPrediction.config) &&
      state.liveOriginalStatusKey === statusConfigurationKey(state.status)
      ? 'Reuses the live original; analyzes the edited PNG. Results are compared only when model settings match.'
      : 'Analyzes the original pixels first, then the edited PNG. Only matched live results are compared.'
  else if (state.editedPrediction)
    note =
      'Estimate moved measures the distance between predictions. It is not an accuracy gain or a privacy score.'
  else
    note =
      'Try a cover or crop to test a visual clue. The original estimate stays available for comparison.'

  const presenterNote = !hasOriginal && state.example
    ? 'Saved evaluation · display copy, not a fresh inference.'
    : state.editedPrediction
      ? 'Matched live pixels. Movement is not accuracy or proof of privacy.'
      : needsEdited
        ? canPredict(state)
          ? 'Matched pixel comparison. Original first, then edited.'
          : 'Live inference required. Local editing still works.'
        : note

  const hasEditsMax = state.operations.length >= MAX_EDITS
  const coverDisabled = locked || !hasImage || hasEditsMax
  const cropDisabled = locked || !hasImage || hasEditsMax

  let serviceMessage: string | null = null
  let showRetry = false
  if (state.status) {
    if (state.status.mode === 'live' && !canPredict(state)) {
      serviceMessage = `Live inference is unavailable${state.status.prediction_unavailable_reason || state.status.reason ? `: ${String(state.status.prediction_unavailable_reason || state.status.reason)}` : ' on this server'}. Recorded results and local editing still work.`
    }
  } else if (state.statusFailed) {
    serviceMessage =
      'The demo service could not be reached. Choose a photo to inspect, edit and download locally, or try connecting again.'
    showRetry = true
  }

  return {
    locked,
    hasImage,
    hasEdits,
    analyzeLabel: state.loading
      ? 'Loading photo…'
      : state.busy
        ? 'Analyzing photo…'
        : analyzeLabel,
    analyzeDisabled,
    truthHidden,
    truthDisabled: locked,
    truthLabel: state.truthVisible ? 'Hide known location' : 'Reveal known location',
    actionNote: state.presenter ? state.comparisonIssue || presenterNote : note,
    presenterNote,
    coverDisabled,
    cropDisabled,
    undoDisabled: locked || !hasEdits,
    resetEditsDisabled: locked || !hasEdits,
    editedViewDisabled: locked || !hasImage,
    originalViewDisabled: locked || !hasImage,
    applyDisabled: locked || !state.selection || Boolean(state.pointer) || hasEditsMax,
    applyLabel: state.tool === 'crop' ? 'Apply crop' : 'Apply cover',
    presetLabel: state.tool === 'crop' ? 'Trim outer 10%' : 'Select center',
    presetDisabled: locked,
    roadPresetHidden: state.tool !== 'cover',
    roadPresetDisabled: locked || !hasImage || hasEditsMax,
    toolHelp:
      state.tool === 'crop'
        ? 'Drag to select the area to keep, or trim the edges with a preset. Then apply the crop.'
        : 'Drag to select a detail, or select the center. Apply an opaque cover to hide those pixels.',
    pixelNote: state.sourceDescription || 'JPEG, PNG or WebP · up to 10 MiB / 50 MP',
    editCount: hasEdits
      ? `${state.operations.length} / ${MAX_EDITS} edits · ${state.work!.width} × ${state.work!.height} px`
      : 'Edits stay in your browser',
    stageTag:
      state.view === 'original'
        ? 'Original photo'
        : state.operations.length
          ? 'Edited pixels'
          : 'Editing view',
    baselineDisabled:
      !state.example ||
      !hasOriginal ||
      locked ||
      !point(state.example?.baseline_lat, state.example?.baseline_lon),
    uploadDisabled: locked,
    resetStoryDisabled: locked || !hasImage,
    downloadDisabled: locked || !hasImage || state.downloadBusy,
    modeLabel: state.status
      ? canPredict(state)
        ? 'Live inference'
        : 'Recorded examples'
      : state.statusFailed
        ? 'Connection unavailable'
        : 'Connecting',
    modeLive: Boolean(state.status && canPredict(state)),
    serviceMessage,
    showRetry,
  }
}

export interface ResultRow {
  key: 'original' | 'edited' | 'truth'
  label: string
  pointText: string
  point: { lat: number; lon: number }
}

export interface ResultMetric {
  value: string
  label: string
  truthError: boolean
}

export interface DerivedResults {
  hasResult: boolean
  provenance: string
  rows: ResultRow[]
  metrics: ResultMetric[]
  points: MapPoint[]
  historicalText: string
  liveSettingsText: string
  baselineNote: string
  hasComparison: boolean
}

export function deriveResults(state: EngineState): DerivedResults {
  const original = state.originalPrediction
  const edited = matchedComparison(original, state.editedPrediction)
    ? state.editedPrediction
    : null
  const rows: ResultRow[] = []
  const points: MapPoint[] = []
  const metrics: ResultMetric[] = []
  const truth = knownTruth(state)

  let provenance = ''
  if (original) {
    provenance =
      original.source === 'recorded'
        ? `Recorded evaluation · ${state.example?.deployed === 'candidate' ? 'deployed candidate' : state.example?.deployed === 'baseline' ? 'retained baseline' : 'saved prediction'}`
        : edited
          ? 'Live pixel comparison · matched model settings'
          : 'Live original · normalized display pixels'
    const originalLabel = original.source === 'live' ? 'Live original' : 'Recorded estimate'
    rows.push({
      key: 'original',
      label: originalLabel,
      pointText: formatPoint(original),
      point: { lat: original.lat, lon: original.lon },
    })
    points.push({ key: 'original', label: originalLabel, ...original })
    if (edited) {
      rows.push({
        key: 'edited',
        label: 'Live edited',
        pointText: formatPoint(edited),
        point: { lat: edited.lat, lon: edited.lon },
      })
      points.push({ key: 'edited', label: 'Live edited', ...edited })
    }
    if (state.truthVisible && truth) {
      rows.push({
        key: 'truth',
        label: 'Known location',
        pointText: formatPoint(truth),
        point: { lat: truth.lat, lon: truth.lon },
      })
      points.push({ key: 'truth', label: 'Known location', ...truth })
    }
    if (state.baselineOn && state.example) {
      const baseline = point(state.example.baseline_lat, state.example.baseline_lon)
      if (baseline) points.push({ key: 'baseline', label: 'Recorded baseline', ...baseline })
    }

    if (original && edited)
      metrics.push({
        value: formatDistance(distance(original, edited)),
        label: 'Estimate moved · between predictions',
        truthError: false,
      })
    if (original && state.truthVisible && truth) {
      metrics.push({
        value: formatDistance(distance(original, truth)),
        label: 'Original error · distance from known location',
        truthError: true,
      })
      if (edited)
        metrics.push({
          value: formatDistance(distance(edited, truth)),
          label: 'Edited error · distance from known location',
          truthError: true,
        })
    }
  }

  const baselinePoint =
    state.example && point(state.example.baseline_lat, state.example.baseline_lon)
  const historicalText = state.historicalPrediction
    ? `Historical recorded evaluation: ${formatPoint(state.historicalPrediction)}${state.truthVisible && truth ? `; error ${formatDistance(distance(state.historicalPrediction, truth))}` : ''}. This result is not used in the live edit comparison.`
    : 'Reveal a recorded example to inspect its historical evaluation result.'
  const liveSettingsText =
    original?.source === 'live'
      ? `Current live original: top_n ${original.config?.top_n ?? 'not reported'}; tower weights ${
          Object.entries(original.config?.tower_weights || {})
            .map(([key, value]) => `${key}: ${value}`)
            .join(', ') || 'not reported'
        }.${edited ? ' The live edited response has matching settings.' : ' No completed matched edit comparison is shown.'} The historical evaluation report above is unchanged.`
      : 'No live pixel analysis in this story yet.'
  const baselineNote =
    original && state.baselineOn && state.example
      ? `Recorded baseline: ${formatPoint(baselinePoint)}. A model comparison, not a known location.`
      : 'Reveal an example’s estimate first. The baseline is a model comparison, not a known location.'

  return {
    hasResult: Boolean(original),
    provenance,
    rows,
    metrics,
    points,
    historicalText,
    liveSettingsText,
    baselineNote,
    hasComparison: Boolean(edited),
  }
}

export { isNumber, formatNumber, formatPoint, formatDistance, distance, point }

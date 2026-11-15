import { useEffect, useRef, useState } from 'react'
import { ClipboardPaste, Crop, Download, ImagePlus, RotateCcw, ScanLine, ShieldCheck, SquareDashed, Undo2, Upload } from 'lucide-react'
import type { DerivedControls, EngineState, OffgridEngine } from '@/lib/engine'
import { Button } from '@/components/ui/button'
import PanelMenu from '@/components/PanelMenu'
import { describe } from '@/lib/engine'
import { sameOriginImage } from '@/lib/image'

function drawPhoto(canvas: HTMLCanvasElement, state: EngineState) {
  if (!state.original) return
  const source = state.view === 'original' ? state.original : state.work!
  canvas.width = source.width
  canvas.height = source.height
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(source, 0, 0)
  if (state.selection && state.view === 'edited') {
    const { x, y, width, height } = state.selection
    const scale = source.width / Math.max(1, canvas.getBoundingClientRect().width)
    ctx.fillStyle = state.tool === 'crop' ? '#15213977' : '#152139b3'
    if (state.tool === 'crop') {
      ctx.beginPath()
      ctx.rect(0, 0, source.width, source.height)
      ctx.rect(x, y, width, height)
      ctx.fill('evenodd')
    } else ctx.fillRect(x, y, width, height)
    ctx.strokeStyle = '#fff'
    ctx.lineWidth = 2 * scale
    ctx.setLineDash([6 * scale, 4 * scale])
    ctx.strokeRect(x, y, width, height)
  }
}

export default function WorkspacePhoto({ engine, controls }: { engine: OffgridEngine; controls: DerivedControls }) {
  const state = engine.state
  const canvas = useRef<HTMLCanvasElement>(null)
  const input = useRef<HTMLInputElement>(null)
  const dragDepth = useRef(0)
  const [dragging, setDragging] = useState(false)
  const [importMessage, setImportMessage] = useState('')
  const [pasting, setPasting] = useState(false)
  const examples = (state.status?.examples || []).slice(0, 3)
  const shortcut = /Mac|iPhone|iPad/.test(navigator.platform) ? 'Cmd+V' : 'Ctrl+V'

  const loadImage = (file: File) => {
    if (engine.state.busy || engine.state.loading) return
    setImportMessage('')
    void engine.loadFile(file)
  }

  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const target = event.target
      if (target instanceof Element && target.closest('input, textarea, [contenteditable="true"]')) return
      const image = Array.from(event.clipboardData?.items || [])
        .find(item => item.kind === 'file' && item.type.startsWith('image/'))?.getAsFile()
      if (!image) return
      event.preventDefault()
      if (engine.state.busy || engine.state.loading) return
      setImportMessage('')
      void engine.loadFile(image)
    }
    // File drops outside the photo must not navigate away from the demo.
    const onDragOver = (event: DragEvent) => {
      if (event.dataTransfer?.types.includes('Files')) event.preventDefault()
    }
    const onDrop = (event: DragEvent) => {
      if (event.defaultPrevented || !event.dataTransfer?.files.length) return
      event.preventDefault()
      if (engine.state.busy || engine.state.loading) return
      setImportMessage('')
      void engine.loadFile(event.dataTransfer.files[0])
    }
    window.addEventListener('paste', onPaste)
    window.addEventListener('dragover', onDragOver)
    window.addEventListener('drop', onDrop)
    return () => {
      window.removeEventListener('paste', onPaste)
      window.removeEventListener('dragover', onDragOver)
      window.removeEventListener('drop', onDrop)
    }
  }, [engine])

  const pasteImage = async () => {
    if (controls.locked || pasting) return
    if (!navigator.clipboard?.read) {
      setImportMessage(`Copy an image, then press ${shortcut} anywhere on this page.`)
      return
    }
    setPasting(true)
    setImportMessage('')
    try {
      const items = await navigator.clipboard.read()
      for (const item of items) {
        const type = item.types.find(value => /^image\/(png|jpeg|webp)$/.test(value))
        if (!type) continue
        const blob = await item.getType(type)
        loadImage(new File([blob], 'pasted-photo', { type }))
        return
      }
      setImportMessage(`No image found. Copy the image itself (not its link), then press ${shortcut}.`)
    } catch {
      setImportMessage(`Clipboard access is blocked. Press ${shortcut} to paste, or drop an image file here.`)
    } finally {
      setPasting(false)
    }
  }
  useEffect(() => { if (canvas.current) drawPhoto(canvas.current, state) })
  return (
    <article className={`photo-panel ${dragging ? 'drag-over' : ''}`} aria-label="Photo inspector"
      onDragEnter={event => {
        if (!event.dataTransfer.types.includes('Files')) return
        event.preventDefault()
        dragDepth.current++
        if (!controls.locked) setDragging(true)
      }}
      onDragOver={event => {
        if (!event.dataTransfer.types.includes('Files')) return
        event.preventDefault()
        event.dataTransfer.dropEffect = controls.locked ? 'none' : 'copy'
      }}
      onDragLeave={event => {
        if (!event.dataTransfer.types.includes('Files')) return
        dragDepth.current = Math.max(0, dragDepth.current - 1)
        if (!dragDepth.current) setDragging(false)
      }}
      onDrop={event => {
        if (!event.dataTransfer.files.length) return
        event.preventDefault()
        dragDepth.current = 0
        setDragging(false)
        loadImage(event.dataTransfer.files[0])
      }}>
      <PanelMenu label="Photo menu">
        <h2>{state.photoTitle}</h2>
        <div className="photo-import-actions">
          <Button variant="ghost" className="paste-button" disabled={controls.uploadDisabled || pasting} onClick={() => void pasteImage()}
            title={`Paste an image (${shortcut})`} aria-label="Paste image from clipboard"><ClipboardPaste size={16} /><span>{pasting ? 'Reading...' : 'Paste'}</span></Button>
          <Button variant="outline" className="upload-button" disabled={controls.uploadDisabled} onClick={() => input.current?.click()}><ImagePlus size={16} /><span>Choose photo</span></Button>
        </div>
        <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" hidden disabled={controls.uploadDisabled} aria-label="Choose a photo to inspect"
          onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; if (file) loadImage(file) }} />
        <p className="menu-hint">Drop an image anywhere or paste with {shortcut}.</p>
        <div className="editor-toolbar" role="group" aria-label="Photo editing tools">
        <Button id="cover-tool" variant="ghost" className={state.tool === 'cover' ? 'active-tool' : ''} aria-pressed={state.tool === 'cover'} disabled={controls.coverDisabled}
          data-menu-close onClick={() => engine.chooseTool('cover')}><SquareDashed size={17} />Cover</Button>
        <Button id="crop-tool" variant="ghost" className={state.tool === 'crop' ? 'active-tool' : ''} aria-pressed={state.tool === 'crop'} disabled={controls.cropDisabled}
          data-menu-close onClick={() => engine.chooseTool('crop')}><Crop size={17} />Crop</Button>
        <span className="toolbar-divider" />
        <Button variant="ghost" className="icon-tool" aria-label="Undo last edit" title="Undo last edit" disabled={controls.undoDisabled} onClick={() => engine.undo()}><Undo2 size={17} /></Button>
        <Button variant="ghost" className="icon-tool" aria-label="Reset edits" title="Reset edits" disabled={controls.resetEditsDisabled} onClick={() => engine.resetEdits()}><RotateCcw size={16} /></Button>
        <div className="view-switch" role="group" aria-label="Photo view">
          {(['original', 'edited'] as const).map(view => <button key={view} aria-pressed={state.view === view}
            disabled={view === 'original' ? controls.originalViewDisabled : controls.editedViewDisabled} onClick={() => engine.setView(view)}>{view === 'original' ? 'Original' : 'Edited'}</button>)}
        </div>
        <Button variant="ghost" className="icon-tool download-tool" aria-label="Download current edit as metadata-free PNG" title="Download edited PNG" disabled={controls.downloadDisabled}
          onClick={() => void engine.downloadImage()}><Download size={17} /></Button>
        </div>
        <div className="menu-divider" />
        <h3>Try a recorded example</h3>
        <div className="example-list">
          {examples.map((example, index) => {
            const [title] = describe(example, index)
            let image = ''
            try { image = sameOriginImage(example.image_url) } catch { /* Skip an invalid preview URL. */ }
            return <button key={String(example.id ?? index)} className="example-card" data-menu-close
              disabled={state.busy || state.loading} aria-pressed={state.example === example} onClick={() => void engine.loadExample(example)}>
              {image && <img src={image} alt="" />}<span>{title}</span>
            </button>
          })}
        </div>
        <p className="menu-hint">First evaluation entries, not selected successes.</p>
        <a className="menu-link" href="/assets/offgrid-demo-photos.zip" download>Download all 18 sample photos</a>
      </PanelMenu>
      <div id="image-stage" className={`photo-stage ${state.tool ? 'is-editing' : ''}`}>
        {state.original ? <canvas ref={canvas} id="photo-canvas" tabIndex={0} role="img"
          aria-label={`${state.photoTitle}. Photo editor. Choose Cover or Crop from the photo menu.`} aria-disabled={controls.locked}
          onPointerDown={event => engine.startPointer(event.currentTarget, event)} onPointerMove={event => engine.movePointer(event.currentTarget, event)}
          onPointerUp={event => engine.endPointer(event.currentTarget, event)} onPointerCancel={event => engine.endPointer(event.currentTarget, event)}
          onKeyDown={event => { if (event.key === 'Escape') engine.escapeCanvas() }} />
          : <div className="photo-empty"><ImagePlus size={34} /><h2>Drop a photo here.</h2><p>Or paste with {shortcut}</p></div>}
        {state.original && <span className="photo-view-label"><ScanLine size={15} />{controls.stageTag}</span>}
        {dragging && <div className="photo-drop-overlay"><Upload size={32} /><strong>Drop it here.</strong><span>JPEG, PNG or WebP, up to 10 MiB</span></div>}
      </div>
      {importMessage && <p className="photo-import-message" role="status">{importMessage}</p>}
      {state.tool && <div className="tool-options">
        <p id="tool-help">{controls.toolHelp}</p>
        <div>
          {!controls.roadPresetHidden && <Button variant="secondary" disabled={controls.roadPresetDisabled} onClick={() => engine.lowerCenterSelection()}>Cover lower center</Button>}
          <Button variant="secondary" disabled={controls.presetDisabled} onClick={() => engine.presetSelection()}>{controls.presetLabel}</Button>
          <Button disabled={controls.applyDisabled} onClick={() => engine.applySelection()}>{controls.applyLabel}</Button>
          <Button variant="ghost" disabled={controls.locked} onClick={() => engine.cancelTool()}>Done</Button>
        </div>
      </div>}
      <div className={`metadata-line ${state.metadata === 'found' ? 'gps-found' : ''}`}><ShieldCheck size={17} /><span>{state.metadataLabel}</span>
        <span className="metadata-extra">{state.operations.length ? `${state.operations.length} edit${state.operations.length === 1 ? '' : 's'} applied` : 'Pixels can still carry clues.'}</span></div>
      {state.downloadNote && <p className="download-note" role="status">{state.downloadNote}</p>}
    </article>
  )
}

/* PhotoPanel — local photo inspector. Canvas editor ported 1:1 from
 * exposure.js drawPhoto/pointer handling; edits never leave the browser. */
import { useEffect, useRef } from 'react'
import {
  Crop,
  Download,
  ImagePlus,
  SquareDashedMousePointer,
  RotateCcw,
  ShieldCheck,
  SquareDashed,
  Undo2,
} from 'lucide-react'
import type { DerivedControls, EngineState, OffgridEngine } from '@/lib/engine'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { cn } from '@/lib/utils'

/* Port of drawPhoto: source canvas + selection overlay. */
function drawPhoto(canvas: HTMLCanvasElement, state: EngineState) {
  if (!state.original) return
  const source = state.view === 'original' ? state.original : state.work!
  canvas.width = source.width
  canvas.height = source.height
  const context = canvas.getContext('2d')!
  context.drawImage(source, 0, 0)
  if (state.selection && state.view === 'edited') {
    const { x, y, width, height } = state.selection
    const lineScale = source.width / Math.max(1, canvas.getBoundingClientRect().width)
    if (state.tool === 'crop') {
      context.fillStyle = '#101e3499'
      context.beginPath()
      context.rect(0, 0, source.width, source.height)
      context.rect(x, y, width, height)
      context.fill('evenodd')
    } else {
      context.fillStyle = '#152139b3'
      context.fillRect(x, y, width, height)
    }
    context.strokeStyle = '#fff'
    context.lineWidth = 2 * lineScale
    context.setLineDash([6 * lineScale, 4 * lineScale])
    context.strokeRect(x, y, width, height)
    context.setLineDash([])
  }
}

interface PhotoPanelProps {
  engine: OffgridEngine
  controls: DerivedControls
  compact: boolean
}

export default function PhotoPanel({ engine, controls, compact }: PhotoPanelProps) {
  const state = engine.state
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const opCountRef = useRef(state.operations.length)
  const flashRef = useRef<HTMLSpanElement>(null)

  /* Apply confirmation: brief frame pulse whenever an edit commits. */
  useEffect(() => {
    const element = flashRef.current
    if (!element) return
    const grew = state.operations.length > opCountRef.current
    opCountRef.current = state.operations.length
    if (!grew) return
    element.classList.remove('og-apply-flash')
    void element.offsetWidth
    element.classList.add('og-apply-flash')
  }, [state.operations.length])

  /* Redraw on every engine notification — draws are idempotent. */
  useEffect(() => {
    const canvas = canvasRef.current
    if (canvas) drawPhoto(canvas, engine.state)
  })

  const onFile = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    void engine.loadFile(file)
  }

  const onDrop = (event: React.DragEvent) => {
    event.preventDefault()
    const file = event.dataTransfer.files?.[0]
    if (file) void engine.loadFile(file)
  }

  return (
    <article
      className="flex min-h-0 flex-col border border-border bg-card"
      aria-label="Photo inspector"
    >
      <div className="flex h-10 shrink-0 items-center justify-between gap-3 border-b border-border px-3">
        <div className="flex min-w-0 items-center gap-2">
          <span className="size-1.5 shrink-0 rounded-full bg-[#f0a828]" aria-hidden="true" />
          <h2 className="truncate text-xs font-medium tracking-wide text-foreground">
            {state.photoTitle}
          </h2>
          <span className="hidden font-mono text-[9px] tracking-[0.22em] text-muted-foreground uppercase sm:inline">
            / photo
          </span>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="h-7 gap-1.5 border-border bg-surface-2 px-2.5 text-[11px]"
          disabled={controls.uploadDisabled}
          onClick={() => fileRef.current?.click()}
        >
          <ImagePlus className="size-3.5" />
          Choose a photo
        </Button>
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          className="hidden"
          disabled={controls.uploadDisabled}
          onChange={onFile}
          aria-label="Choose a photo to inspect"
        />
      </div>

      <div
        id="image-stage"
        className={cn(
          'scanline relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[#0b1017] p-2',
          !compact && 'min-h-[300px]',
        )}
        onDragOver={event => event.preventDefault()}
        onDrop={onDrop}
      >
        {!state.original ? (
          <div className="flex flex-col items-center gap-2 text-center text-muted-foreground">
            <SquareDashedMousePointer className="size-8 opacity-50" aria-hidden="true" />
            <strong className="text-sm font-medium text-foreground/90">
              A photo is the starting point.
            </strong>
            <span className="max-w-56 text-xs">
              Choose a JPEG, PNG or WebP to explore locally.
            </span>
          </div>
        ) : (
          <canvas
            id="photo-canvas"
            ref={canvasRef}
            tabIndex={0}
            role="img"
            aria-label={`${state.photoTitle}. Photo editor; choose Cover or Crop below. Keyboard presets are available.`}
            aria-disabled={controls.locked}
            className={cn(
              'max-h-full max-w-full object-contain outline-none focus-visible:ring-2 focus-visible:ring-ring',
              state.tool && !controls.locked && 'cursor-crosshair',
            )}
            onPointerDown={event => engine.startPointer(event.currentTarget, event)}
            onPointerMove={event => engine.movePointer(event.currentTarget, event)}
            onPointerUp={event => engine.endPointer(event.currentTarget, event)}
            onPointerCancel={event => engine.endPointer(event.currentTarget, event)}
            onKeyDown={event => {
              if (event.key === 'Escape') engine.escapeCanvas()
            }}
          />
        )}
        {state.original && (
          <span className="absolute top-2 left-2 border border-[#f0a828]/40 bg-[#0a0d12]/80 px-1.5 py-0.5 font-mono text-[9px] tracking-[0.2em] text-[#f0a828] uppercase backdrop-blur-sm">
            {controls.stageTag}
          </span>
        )}
        <span
          ref={flashRef}
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 opacity-0 shadow-[inset_0_0_0_1px_rgba(95,212,208,0.9),inset_0_0_26px_rgba(95,212,208,0.22)]"
        />
      </div>

      <div className="shrink-0 border-t border-border px-3 py-2">
        <div
          className="flex flex-wrap items-center gap-x-3 gap-y-2"
          role="group"
          aria-label="Photo editing tools"
        >
          <div className="flex items-center gap-1">
            <Button
              id="cover-tool"
              variant={state.tool === 'cover' ? 'secondary' : 'outline'}
              size="sm"
              className={cn(
                'h-7 gap-1.5 border-border px-2.5 text-[11px]',
                state.tool === 'cover' &&
                  'border-[#5fd4d0]/50 bg-[#5fd4d0]/10 text-[#5fd4d0]',
              )}
              aria-pressed={state.tool === 'cover'}
              disabled={controls.coverDisabled}
              onClick={() => engine.chooseTool('cover')}
            >
              <SquareDashed className="size-3.5" />
              Cover
            </Button>
            <Button
              id="crop-tool"
              variant={state.tool === 'crop' ? 'secondary' : 'outline'}
              size="sm"
              className={cn(
                'h-7 gap-1.5 border-border px-2.5 text-[11px]',
                state.tool === 'crop' && 'border-[#5fd4d0]/50 bg-[#5fd4d0]/10 text-[#5fd4d0]',
              )}
              aria-pressed={state.tool === 'crop'}
              disabled={controls.cropDisabled}
              onClick={() => engine.chooseTool('crop')}
            >
              <Crop className="size-3.5" />
              Crop
            </Button>
          </div>

          <Separator orientation="vertical" className="hidden h-5 sm:block" />

          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon-sm"
              className="size-7 border-border"
              title="Undo last edit"
              aria-label="Undo last edit"
              disabled={controls.undoDisabled}
              onClick={() => engine.undo()}
            >
              <Undo2 className="size-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-[11px] text-muted-foreground"
              disabled={controls.resetEditsDisabled}
              onClick={() => engine.resetEdits()}
            >
              <RotateCcw className="size-3" />
              Reset edits
            </Button>
          </div>

          <Separator orientation="vertical" className="hidden h-5 sm:block" />

          <div
            className="flex items-center border border-border p-0.5"
            role="group"
            aria-label="Photo view"
          >
            {(['original', 'edited'] as const).map(view => (
              <button
                key={view}
                type="button"
                aria-pressed={state.view === view}
                disabled={view === 'edited' ? controls.editedViewDisabled : controls.originalViewDisabled}
                onClick={() => engine.setView(view)}
                className={cn(
                  'px-2.5 py-1 font-mono text-[10px] tracking-[0.14em] uppercase transition-colors',
                  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                  'disabled:cursor-not-allowed disabled:opacity-40',
                  state.view === view
                    ? 'bg-[#f0a828]/15 text-[#f0a828]'
                    : 'text-muted-foreground hover:text-foreground',
                )}
              >
                {view}
              </button>
            ))}
          </div>
        </div>

        {state.tool && (
          <div className="mt-2 border-t border-border/60 pt-2">
            <p id="tool-help" className="text-[11px] text-muted-foreground">
              {controls.toolHelp}
            </p>
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              {!controls.roadPresetHidden && (
                <Button
                  variant="secondary"
                  size="sm"
                  className="h-6 px-2 text-[11px]"
                  disabled={controls.roadPresetDisabled}
                  onClick={() => engine.lowerCenterSelection()}
                >
                  Cover lower center
                </Button>
              )}
              <Button
                variant="secondary"
                size="sm"
                className="h-6 px-2 text-[11px]"
                disabled={controls.presetDisabled}
                onClick={() => engine.presetSelection()}
              >
                {controls.presetLabel}
              </Button>
              <Button
                size="sm"
                className="h-6 bg-[#f0a828] px-2.5 text-[11px] font-semibold text-[#14100a] hover:bg-[#f0a828]/85"
                disabled={controls.applyDisabled}
                onClick={() => engine.applySelection()}
              >
                {controls.applyLabel}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-[11px] text-muted-foreground"
                disabled={controls.locked}
                onClick={() => engine.cancelTool()}
              >
                Done
              </Button>
            </div>
          </div>
        )}

        {!compact && (
          <div className="mt-2 flex flex-wrap items-center justify-between gap-x-4 gap-y-1 font-mono text-[10px] text-muted-foreground/80">
            <span>{controls.pixelNote}</span>
            <span>{controls.editCount}</span>
          </div>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2.5 border-t border-border bg-surface-2/60 px-3 py-2">
        <ShieldCheck
          className={cn(
            'size-4 shrink-0',
            state.metadata === 'found'
              ? 'text-[#f06548]'
              : state.metadata === 'absent'
                ? 'text-[#5fd4d0]'
                : 'text-muted-foreground',
          )}
          aria-hidden="true"
        />
        <div className="min-w-0 flex-1">
          <strong className="block text-[11px] font-medium text-foreground">
            {state.metadataLabel}
          </strong>
          <span className="block truncate text-[10px] text-muted-foreground">
            {state.metadataNote}
          </span>
        </div>
        <Button
          variant="outline"
          size="icon-sm"
          className="size-7 shrink-0 border-border"
          aria-label="Download current edit as metadata-free PNG"
          title="Download edited PNG"
          disabled={controls.downloadDisabled}
          onClick={() => void engine.downloadImage()}
        >
          <Download className="size-3.5" />
        </Button>
      </div>
      {state.downloadNote && (
        <p className="shrink-0 border-t border-border px-3 py-1.5 text-[10px] text-muted-foreground" role="status">
          {state.downloadNote}
        </p>
      )}
    </article>
  )
}

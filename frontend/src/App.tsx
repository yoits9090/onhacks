/* OFFGRID — dark reconnaissance console. A photo says where. */
import { useEffect, useReducer, useRef, useState } from 'react'
import { TriangleAlert, RotateCcw } from 'lucide-react'
import { MotionConfig } from 'framer-motion'
import {
  OffgridEngine,
  deriveControls,
  deriveResults,
  initialState,
} from '@/lib/engine'
import { formatPoint } from '@/lib/geo'
import TopBar from '@/components/TopBar'
import PresenterBar from '@/components/PresenterBar'
import PhotoPanel from '@/components/PhotoPanel'
import MapView from '@/components/MapView'
import ResultsPanel from '@/components/ResultsPanel'
import ExamplesStrip from '@/components/ExamplesStrip'
import Dialogs from '@/components/Dialogs'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

function focusById(id: string) {
  const target = document.getElementById(id)
  target?.focus({ preventScroll: true })
  return target
}

export default function App() {
  const [, setTick] = useReducer((x: number) => x + 1, 0)
  const [engine] = useState(() => {
    const e = new OffgridEngine()
    e.state = initialState()
    e.notify = () => setTick() // stable dispatch — set once, forever valid
    return e
  })
  const startedRef = useRef(false)

  useEffect(() => {
    if (!startedRef.current) {
      startedRef.current = true
      void engine.loadStatus()
    }
  }, [engine])

  const state = engine.state
  const controls = deriveControls(state)
  const results = deriveResults(state)
  const presenter = state.presenter

  const goToCue = (cue: string) => {
    engine.markCue(cue)
    if (cue === 'evidence') {
      engine.openEvidence()
      return
    }
    if (cue === 'takeaway') {
      engine.openTakeaway()
      return
    }
    let target: HTMLElement | null
    if (cue === 'edit') target = focusById('cover-tool')
    else if (cue === 'reveal') {
      const analyze = document.getElementById('analyze-button') as HTMLButtonElement | null
      const truth = document.getElementById('truth-button')
      target = analyze && !analyze.disabled ? analyze : truth ?? focusById('workbench')
      target?.focus({ preventScroll: true })
    } else target = focusById('photo-canvas')
    if (!presenter || window.innerWidth < 1000)
      target?.scrollIntoView({ block: 'center' })
  }

  const mapAriaLabel = results.points.length
    ? `Location map. ${results.points.map(p => `${p.label}: ${formatPoint(p)}`).join('. ')}.`
    : 'World map. No location estimate revealed.'

  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider>
      <div
        className={cn(
          'flex min-h-dvh flex-col bg-background text-foreground',
          presenter && 'presentation',
        )}
      >
        <a
          href="#workbench"
          className="sr-only focus:not-sr-only focus:absolute focus:z-[100] focus:bg-surface-2 focus:px-3 focus:py-2 focus:text-sm focus:text-foreground"
        >
          Skip to photo workbench
        </a>

        <TopBar engine={engine} controls={controls} />
        {presenter && <PresenterBar engine={engine} onCue={goToCue} />}

        <main
          className={cn(
            'bg-grid mx-auto flex w-full max-w-[1440px] flex-1 flex-col px-3 sm:px-4',
            presenter ? 'min-h-0 py-2' : 'py-4',
          )}
        >
          <section
            className={cn(
              'flex shrink-0 items-end justify-between gap-4',
              presenter ? 'mb-2' : 'mb-4',
            )}
            aria-labelledby="page-title"
          >
            <div className="min-w-0">
              <h1
                id="page-title"
                className={cn(
                  'font-semibold tracking-tight text-foreground',
                  presenter ? 'text-lg leading-tight' : 'text-2xl sm:text-3xl',
                )}
              >
                A photo says where.
              </h1>
              {!presenter && (
                <p className="mt-1 text-xs text-muted-foreground sm:text-sm">
                  See what a street photo reveals. Hide a detail. Test it again.
                </p>
              )}
            </div>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 shrink-0 gap-1.5 px-2 text-[11px] text-muted-foreground hover:text-foreground"
              title="Return to the first example and hide all results"
              disabled={controls.resetStoryDisabled}
              onClick={() => engine.resetStory()}
            >
              <RotateCcw className="size-3.5" />
              Reset story
            </Button>
          </section>

          {controls.serviceMessage && (
            <Alert className="mb-3 shrink-0 border-[#f0a828]/40 bg-[#f0a828]/6 py-2 text-foreground [&>svg]:text-[#f0a828]">
              <TriangleAlert className="size-4" />
              <AlertDescription className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
                <span>{controls.serviceMessage}</span>
                {controls.showRetry && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-6 border-[#f0a828]/50 px-2 text-[11px] text-[#f0a828] hover:bg-[#f0a828]/10"
                    onClick={() => void engine.loadStatus()}
                  >
                    Try connection again
                  </Button>
                )}
              </AlertDescription>
            </Alert>
          )}
          {state.error && (
            <Alert
              variant="destructive"
              className="mb-3 shrink-0 border-[#f06548]/50 bg-[#f06548]/8 py-2 text-foreground [&>svg]:text-[#f06548]"
            >
              <TriangleAlert className="size-4" />
              <AlertDescription className="text-xs">{state.error}</AlertDescription>
            </Alert>
          )}

          <section
            id="workbench"
            tabIndex={-1}
            aria-label="Photo and location workbench"
            className={cn(
              'grid min-h-0 flex-1 content-start gap-3 outline-none',
              'lg:grid-cols-[minmax(0,11fr)_minmax(0,9fr)] lg:content-stretch',
              presenter ? 'lg:h-[calc(100dvh-9.75rem)]' : 'lg:h-[min(74dvh,700px)] lg:min-h-[470px]',
            )}
          >
            <PhotoPanel engine={engine} controls={controls} compact={presenter} />

            <article
              className="reticle-corners flex min-h-0 flex-col border border-border bg-card"
              aria-label="Location estimate"
            >
              <div className="flex h-10 shrink-0 items-center justify-between border-b border-border px-3">
                <div className="flex items-center gap-2">
                  <span className="size-1.5 rounded-full bg-[#f0a828]" aria-hidden="true" />
                  <h2 className="text-xs font-medium tracking-wide">A point of view</h2>
                  <span className="hidden font-mono text-[9px] tracking-[0.22em] text-muted-foreground uppercase sm:inline">
                    / estimate
                  </span>
                </div>
              </div>
              <div className="min-h-64 flex-1 basis-0 lg:min-h-0">
                <MapView points={results.points} ariaLabel={mapAriaLabel} />
              </div>
              <div className="max-h-[46%] shrink-0 overflow-y-auto border-t border-border">
                <ResultsPanel
                  engine={engine}
                  controls={controls}
                  results={results}
                  state={state}
                />
              </div>
            </article>
          </section>

          {!presenter && (
            <div className="mt-4 shrink-0">
              <ExamplesStrip engine={engine} state={state} />
            </div>
          )}

          {!presenter && (
            <footer className="mt-6 flex shrink-0 items-center justify-between border-t border-border pt-3 pb-2 font-mono text-[10px] text-muted-foreground/70">
              <span>Less in the frame. Still a place in the world.</span>
              <span>OFFGRID / a location-exposure study</span>
            </footer>
          )}
        </main>

        <Dialogs engine={engine} results={results} />

        <div
          className="sr-only"
          role="status"
          aria-live="polite"
          key={state.announcement.seq}
        >
          {state.announcement.text}
        </div>
        <Toaster theme="dark" position="bottom-right" />
      </div>
      </TooltipProvider>
    </MotionConfig>
  )
}

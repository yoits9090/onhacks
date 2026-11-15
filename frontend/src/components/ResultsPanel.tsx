/* ResultsPanel — mono coordinate readouts, headline distance metric,
 * analyze/truth actions, wait state. Port of renderResults markup. */
import { useEffect, useRef, useState } from 'react'
import { Crosshair, LoaderCircle, MapPin } from 'lucide-react'
import { cubicBezier, useReducedMotion } from 'framer-motion'
import type {
  DerivedControls,
  DerivedResults,
  EngineState,
  OffgridEngine,
} from '@/lib/engine'
import { formatPoint } from '@/lib/geo'
import type { GeoPoint } from '@/lib/types'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const PIN_CLASSES: Record<string, string> = {
  original: 'bg-[#f0a828]',
  edited: 'bg-[#5fd4d0]',
  truth: 'bg-[#f06548]',
  baseline: 'bg-[#b1a1db]',
}

/* Coordinates count up on mount — the readout's moment of arrival. */
function AnimatedCoord({ target, animate }: { target: GeoPoint; animate: boolean }) {
  const reduce = useReducedMotion()
  const [display, setDisplay] = useState<GeoPoint>(animate && !reduce ? { lat: 0, lon: 0 } : target)
  const frameRef = useRef<number>(0)
  useEffect(() => {
    if (!animate || reduce) {
      setDisplay(target)
      return
    }
    const started = performance.now()
    const duration = 620
    const ease = cubicBezier(0.16, 1, 0.3, 1)
    const step = (now: number) => {
      const t = Math.min(1, (now - started) / duration)
      setDisplay({ lat: target.lat * ease(t), lon: target.lon * ease(t) })
      if (t < 1) frameRef.current = requestAnimationFrame(step)
    }
    frameRef.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frameRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target.lat, target.lon])
  return <>{formatPoint(display)}</>
}

interface ResultsPanelProps {
  engine: OffgridEngine
  controls: DerivedControls
  results: DerivedResults
  state: EngineState
}

export default function ResultsPanel({ engine, controls, results, state }: ResultsPanelProps) {
  return (
    <div className="flex min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        {!results.hasResult ? (
          <div className="flex h-full min-h-16 flex-col justify-center">
            <h3 className="text-xs font-medium text-foreground/90">
              What can the pixels reveal?
            </h3>
            <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
              Start with the original. Then cover or crop a detail and compare what
              changes.
            </p>
          </div>
        ) : (
          <div>
            <p className="font-mono text-[9px] tracking-[0.22em] text-[#5fd4d0] uppercase">
              {results.provenance}
            </p>
            <dl className="mt-1.5 space-y-1">
              {results.rows.map(row => (
                <div key={row.key} className="flex items-baseline justify-between gap-3">
                  <dt className="flex items-center gap-2 text-[11px] text-muted-foreground">
                    <i
                      className={cn('size-2 rounded-full', PIN_CLASSES[row.key])}
                      aria-hidden="true"
                    />
                    {row.label}
                  </dt>
                  <dd className="coord-mono text-[11.5px] text-foreground">
                    <AnimatedCoord target={row.point} animate />
                  </dd>
                </div>
              ))}
            </dl>
            {results.metrics.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1.5 border-t border-border/60 pt-2">
                {results.metrics.map((metric, index) => (
                  <div key={index} className={metric.truthError ? 'truth-error' : ''}>
                    <strong
                      className={cn(
                        'coord-mono block text-xl font-semibold leading-none',
                        metric.truthError ? 'text-[#f06548]' : 'text-[#f0a828]',
                      )}
                    >
                      {metric.value}
                    </strong>
                    <span className="mt-0.5 block text-[10px] text-muted-foreground">
                      {metric.label}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <p className="mt-2 text-[10px] text-muted-foreground/80">
              A location estimate, not a verified location.
            </p>
          </div>
        )}
      </div>

      <div className="shrink-0 border-t border-border px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <Button
            id="analyze-button"
            className="h-8 gap-2 bg-[#f0a828] px-3.5 text-xs font-semibold text-[#14100a] hover:bg-[#f0a828]/85"
            disabled={controls.analyzeDisabled}
            onClick={() => void engine.analyze()}
          >
            {state.busy ? (
              <LoaderCircle className="size-3.5 animate-spin" />
            ) : (
              <Crosshair className="size-3.5" />
            )}
            {controls.analyzeLabel}
          </Button>
          {!controls.truthHidden && (
            <Button
              id="truth-button"
              variant="outline"
              className="h-8 gap-1.5 border-[#f06548]/50 px-3 text-xs text-[#f06548] hover:bg-[#f06548]/10 hover:text-[#f06548]"
              aria-pressed={state.truthVisible}
              disabled={controls.truthDisabled}
              onClick={() => engine.toggleTruth()}
            >
              <MapPin className="size-3.5" />
              {controls.truthLabel}
            </Button>
          )}
        </div>

        {state.busy && (
          <div
            className="mt-2 flex flex-wrap items-center gap-2 border border-border bg-surface-2 px-2.5 py-1.5"
            role="status"
          >
            <LoaderCircle className="size-3.5 animate-spin text-[#5fd4d0]" aria-hidden="true" />
            <span className="coord-mono text-[11px] text-foreground">
              {state.waitStage} · {state.waitElapsed}s elapsed
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-6 px-2 text-[11px] text-muted-foreground"
              onClick={() => engine.cancelRequest()}
            >
              Cancel
            </Button>
            {state.showContinueEvidence && (
              <Button
                variant="ghost"
                size="sm"
                className="h-6 px-2 text-[11px] text-[#5fd4d0]"
                onClick={() => engine.continueToEvidence()}
              >
                Continue to evidence
              </Button>
            )}
          </div>
        )}
        <p className="mt-1.5 text-[10px] leading-snug text-muted-foreground" aria-live="polite">
          {controls.actionNote}
        </p>
      </div>

      <div className="shrink-0 border-t border-border px-3 py-1.5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1" aria-label="Map legend">
          <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <i className={cn('size-2 rounded-full', PIN_CLASSES.original)} />
            Original estimate
          </span>
          <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <i className={cn('size-2 rounded-full', PIN_CLASSES.edited)} />
            Edited
          </span>
          <span className="flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <i className={cn('size-2 rounded-full', PIN_CLASSES.truth)} />
            Known location
          </span>
          <label className="ml-auto flex items-center gap-1.5 text-[10px] text-muted-foreground">
            <input
              type="checkbox"
              className="size-3 accent-[#b1a1db]"
              checked={state.baselineOn}
              disabled={controls.baselineDisabled}
              onChange={() => engine.toggleBaseline()}
            />
            Baseline
          </label>
        </div>
      </div>
    </div>
  )
}

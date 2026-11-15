/* ExamplesStrip — filmstrip of the recorded manifest examples. */
import { Check } from 'lucide-react'
import { describe } from '@/lib/engine'
import type { EngineState, OffgridEngine } from '@/lib/engine'
import { sameOriginImage } from '@/lib/image'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import { cn } from '@/lib/utils'

interface ExamplesStripProps {
  engine: OffgridEngine
  state: EngineState
}

export default function ExamplesStrip({ engine, state }: ExamplesStripProps) {
  const examples = Array.isArray(state.status?.examples)
    ? state.status.examples.filter(e => e && typeof e === 'object').slice(0, 3)
    : []

  return (
    <section aria-labelledby="examples-heading" className="shrink-0">
      <div className="mb-2 flex items-baseline justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="size-1.5 rounded-full bg-[#5fd4d0]" aria-hidden="true" />
          <h2 id="examples-heading" className="text-xs font-medium tracking-wide">
            Recorded examples
          </h2>
          <span className="font-mono text-[9px] tracking-[0.22em] text-muted-foreground uppercase">
            / first three manifest entries
          </span>
        </div>
        <p className="hidden text-[10px] text-muted-foreground/80 sm:block">
          Deterministic order, not selected successes.
        </p>
      </div>
      <ScrollArea className="w-full whitespace-nowrap">
        <div className="flex gap-2 pb-2">
          {examples.length === 0 && (
            <p className="text-[11px] text-muted-foreground">
              {state.status
                ? 'No recorded examples are available. Choose a photo to inspect metadata, cover a detail, or make a crop.'
                : 'Recorded examples are unavailable while the service is disconnected. Your own photos can still be edited locally.'}
            </p>
          )}
          {examples.map((example, index) => {
            const [title, subtitle] = describe(example, index)
            let src = ''
            try {
              src = sameOriginImage(example.image_url)
            } catch {
              src = ''
            }
            const selected = state.example === example
            return (
              <button
                key={index}
                type="button"
                aria-pressed={selected}
                disabled={state.busy || state.loading}
                onClick={() => void engine.loadExample(example)}
                className={cn(
                  'group relative flex w-44 shrink-0 flex-col overflow-hidden border text-left transition-colors',
                  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                  'disabled:cursor-not-allowed disabled:opacity-50',
                  selected
                    ? 'border-[#f0a828]/70 bg-surface-2'
                    : 'border-border bg-card hover:border-[#33465f]',
                )}
              >
                <div className="relative h-20 w-full overflow-hidden bg-[#0b1017]">
                  {src ? (
                    <img
                      src={src}
                      alt={title}
                      loading="lazy"
                      className="h-full w-full object-cover opacity-90 transition-opacity group-hover:opacity-100"
                      onError={event => {
                        ;(event.target as HTMLImageElement).alt = 'Image unavailable'
                      }}
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-[10px] text-muted-foreground">
                      Image unavailable
                    </div>
                  )}
                  {selected && (
                    <span className="absolute top-1.5 right-1.5 flex size-4 items-center justify-center bg-[#f0a828] text-[#14100a]">
                      <Check className="size-3" aria-hidden="true" />
                    </span>
                  )}
                </div>
                <div className="px-2 py-1.5">
                  <strong className="block truncate text-[11px] font-medium text-foreground">
                    {title}
                  </strong>
                  <span className="block truncate font-mono text-[9px] text-muted-foreground">
                    {subtitle}
                  </span>
                </div>
              </button>
            )
          })}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </section>
  )
}

/* PresenterBar — five-minute cue rail: Photo → Reveal → Edit → Evidence → Takeaway. */
import { RotateCcw } from 'lucide-react'
import type { OffgridEngine } from '@/lib/engine'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const CUES: { cue: string; label: string; time: string }[] = [
  { cue: 'photo', label: 'Photo', time: '0:00' },
  { cue: 'reveal', label: 'Reveal', time: '0:30' },
  { cue: 'edit', label: 'Edit', time: '1:30' },
  { cue: 'evidence', label: 'Evidence', time: '2:45' },
  { cue: 'takeaway', label: 'Takeaway', time: '3:45' },
]

interface PresenterBarProps {
  engine: OffgridEngine
  onCue: (cue: string) => void
}

export default function PresenterBar({ engine, onCue }: PresenterBarProps) {
  const state = engine.state
  const seconds = Math.floor(engine.presenterSeconds())
  const clock = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`
  const warning = seconds >= 285
  const timerNote =
    seconds >= 300
      ? 'Five minutes reached'
      : seconds >= 295
        ? 'Close with takeaway'
        : seconds >= 285
          ? 'Wrap up · target 4:55'
          : 'Close by 4:55'

  return (
    <section
      className={cn(
        'flex h-11 shrink-0 items-center gap-3 overflow-x-auto border-b px-3 sm:px-4',
        warning ? 'border-[#f06548]/50 bg-[#f06548]/5' : 'border-border bg-[#0e141d]',
      )}
      aria-label="Five-minute presentation controls"
    >
      <div className="flex shrink-0 items-center gap-2">
        <output
          role="timer"
          aria-live="off"
          aria-label="Presentation elapsed time"
          className={cn(
            'coord-mono w-11 text-sm font-semibold',
            warning ? 'text-[#f06548]' : 'text-[#f0a828]',
          )}
        >
          {clock}
        </output>
        <Button
          variant="outline"
          size="sm"
          className="h-6 border-border px-2 text-[10px]"
          onClick={() => engine.togglePresenterTimer()}
        >
          {state.presenterStarted !== null ? 'Pause' : seconds ? 'Resume' : 'Start timer'}
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          className="size-6 text-muted-foreground"
          aria-label="Reset presentation timer"
          title="Reset timer"
          onClick={() => engine.resetPresenterTimer()}
        >
          <RotateCcw className="size-3" />
        </Button>
      </div>

      <nav
        className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto"
        aria-label="Presentation cues"
      >
        {CUES.map(({ cue, label, time }) => (
          <button
            key={cue}
            type="button"
            data-cue={cue}
            aria-current={state.activeCue === cue ? 'step' : undefined}
            disabled={state.busy || state.loading}
            onClick={() => onCue(cue)}
            className={cn(
              'presenter-cue flex shrink-0 items-center gap-1.5 px-2 py-1 text-[11px] transition-colors',
              'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-40',
              state.activeCue === cue ? 'text-[#f0a828]' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            <span
              className={cn(
                'cue-dot size-1.5 rounded-full',
                state.activeCue === cue ? 'bg-[#f0a828]' : 'bg-muted-foreground/40',
              )}
              aria-hidden="true"
            />
            <span>{label}</span>
            <span className="coord-mono text-[9px] text-muted-foreground/70">{time}</span>
          </button>
        ))}
      </nav>

      <span
        className={cn(
          'shrink-0 font-mono text-[10px]',
          warning ? 'text-[#f06548]' : 'text-muted-foreground',
        )}
      >
        {timerNote}
      </span>
    </section>
  )
}

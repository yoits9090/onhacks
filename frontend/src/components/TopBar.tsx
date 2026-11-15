/* TopBar — OFFGRID wordmark, mode badge, nav, presenter toggle. */
import { CircleQuestionMark, MonitorPlay } from 'lucide-react'
import type { DerivedControls, OffgridEngine } from '@/lib/engine'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export function ReticleMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 40 40"
      className={className}
      aria-hidden="true"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
    >
      <circle cx="20" cy="20" r="13" />
      <path d="M20 2v8M20 30v8M2 20h8M30 20h8" />
      <circle cx="20" cy="20" r="2.4" fill="currentColor" stroke="none" />
    </svg>
  )
}

interface TopBarProps {
  engine: OffgridEngine
  controls: DerivedControls
}

export default function TopBar({ engine, controls }: TopBarProps) {
  const presenter = engine.state.presenter
  return (
    <header className="flex h-12 shrink-0 items-center justify-between gap-3 border-b border-border bg-[#0c1017] px-3 sm:px-4">
      <a
        href="/"
        className="flex min-w-0 items-center gap-2 text-foreground"
        aria-label="Offgrid home"
      >
        <ReticleMark className="size-5 shrink-0 text-[#f0a828]" />
        <span className="truncate text-sm font-semibold tracking-[0.16em] uppercase">
          offgrid<span className="text-[#f0a828]">.</span>
        </span>
      </a>

      <div className="flex items-center gap-2 sm:gap-3">
        <Badge
          variant="outline"
          className={cn(
            'gap-1.5 rounded-sm border-border px-2 py-0.5 font-mono text-[10px] font-normal tracking-[0.14em] uppercase',
            controls.modeLive
              ? 'og-badge-shimmer border-[#5fd4d0]/50 text-[#5fd4d0]'
              : 'text-muted-foreground',
          )}
        >
          <span
            className={cn(
              'size-1.5 rounded-full',
              controls.modeLive ? 'bg-[#5fd4d0]' : 'bg-[#f0a828]',
              !engine.state.status && !controls.modeLive && 'animate-pulse',
            )}
            aria-hidden="true"
          />
          {controls.modeLabel}
        </Badge>

        <a
          href="/lab"
          className="hidden font-mono text-[11px] text-muted-foreground transition-colors hover:text-foreground sm:inline"
        >
          Calibration lab →
        </a>

        <Button
          variant="ghost"
          size="sm"
          className="hidden h-7 gap-1.5 px-2 text-[11px] text-muted-foreground hover:text-foreground sm:flex"
          onClick={() => engine.setHowOpen(true)}
        >
          <CircleQuestionMark className="size-3.5" />
          How it works
        </Button>

        <Button
          id="presentation-button"
          variant={presenter ? 'secondary' : 'ghost'}
          size="icon-sm"
          className={cn(
            'size-7',
            presenter && 'bg-[#f0a828]/15 text-[#f0a828] hover:bg-[#f0a828]/25',
          )}
          aria-pressed={presenter}
          title="Presentation mode"
          aria-label="Presentation mode"
          onClick={() => engine.togglePresenter()}
        >
          <MonitorPlay className="size-4" />
        </Button>
      </div>
    </header>
  )
}

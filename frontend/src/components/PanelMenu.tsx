import { useEffect, useRef } from 'react'
import type { ReactNode } from 'react'
import { Menu, X } from 'lucide-react'

export default function PanelMenu({ label, children }: { label: string; children: ReactNode }) {
  const ref = useRef<HTMLDetailsElement>(null)
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (event.target instanceof Node && !ref.current?.contains(event.target) && ref.current) ref.current.open = false
    }
    document.addEventListener('pointerdown', close)
    return () => document.removeEventListener('pointerdown', close)
  }, [])
  return (
    <details ref={ref} className="panel-menu" onKeyDown={event => {
      if (event.key === 'Escape' && ref.current) {
        ref.current.open = false
        ref.current.querySelector('summary')?.focus()
        event.stopPropagation()
      }
    }} onClick={event => {
      if (event.target instanceof Element && event.target.closest('[data-menu-close]') && ref.current) ref.current.open = false
    }}>
      <summary aria-label={label} title={label}><Menu className="menu-open-icon" size={21} /><X className="menu-close-icon" size={21} /></summary>
      <div className="panel-menu-content">{children}</div>
    </details>
  )
}

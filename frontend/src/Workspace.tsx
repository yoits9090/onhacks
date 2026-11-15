import { useEffect, useReducer, useRef, useState } from 'react'
import { MonitorPlay, RotateCcw, TriangleAlert } from 'lucide-react'
import { MotionConfig } from 'framer-motion'
import { OffgridEngine, deriveControls, deriveResults, initialState } from '@/lib/engine'
import { formatPoint } from '@/lib/geo'
import PresenterBar from '@/components/PresenterBar'
import WorkspacePhoto from '@/components/WorkspacePhoto'
import StreetMap from '@/components/StreetMap'
import WorkspaceResults from '@/components/WorkspaceResults'
import Dialogs from '@/components/Dialogs'
import { Button } from '@/components/ui/button'
import { Toaster } from '@/components/ui/sonner'
import { TooltipProvider } from '@/components/ui/tooltip'
import PanelMenu from '@/components/PanelMenu'
import OffgridLogo from '@/components/OffgridLogo'

export default function Workspace() {
  const [, redraw] = useReducer((x: number) => x + 1, 0)
  const [engine] = useState(() => {
    const value = new OffgridEngine()
    value.state = initialState()
    value.notify = () => redraw()
    return value
  })
  const started = useRef(false)
  useEffect(() => {
    if (!started.current) {
      started.current = true
      void engine.loadStatus()
    }
  }, [engine])
  const state = engine.state
  const controls = deriveControls(state)
  const results = deriveResults(state)
  const goToCue = (cue: string) => {
    engine.markCue(cue)
    if (cue === 'evidence') return engine.openEvidence()
    if (cue === 'takeaway') return engine.openTakeaway()
    const target = document.getElementById(cue === 'edit' ? 'cover-tool' : cue === 'reveal' ? 'analyze-button' : 'photo-canvas')
    target?.focus({ preventScroll: true })
    if (window.innerWidth < 900) target?.scrollIntoView({ block: 'center' })
  }
  return (
    <MotionConfig reducedMotion="user">
      <TooltipProvider>
        <div className={`offgrid-app ${state.presenter ? 'presentation' : ''}`}>
          <a href="#workbench" className="skip-link">Skip to workbench</a>
          {state.presenter && <PresenterBar engine={engine} onCue={goToCue} />}
          <main className="workspace">
            <h1 className="sr-only">Offgrid: explore a photo's location</h1>
            {(controls.serviceMessage || state.error) && <div className={`service-notice ${state.error ? 'error' : ''}`} role={state.error ? 'alert' : 'status'}>
              <TriangleAlert size={17} /><span>{state.error || controls.serviceMessage}</span>
              {controls.showRetry && <Button variant="outline" onClick={() => void engine.loadStatus()}>Reconnect</Button>}
            </div>}
            <section id="workbench" className={`workbench ${results.hasResult ? 'has-result' : ''}`} tabIndex={-1} aria-label="Photo and location workbench">
              <div className="photo-column">
                <WorkspacePhoto engine={engine} controls={controls} />
                <OffgridLogo />
              </div>
              <article className="location-column" aria-label="Location estimate">
                <StreetMap points={results.points} ariaLabel={results.points.length
                  ? `Location map. ${results.points.map(p => `${p.label}: ${formatPoint(p)}`).join('. ')}.`
                  : 'OpenStreetMap world map. No location estimate revealed.'} />
                <PanelMenu label="Map menu">
                  <h2>Location</h2>
                  <span className={`connection-status ${controls.modeLive ? 'connected' : ''}`}><i />{controls.modeLabel}</span>
                  {!controls.truthHidden && <Button id="truth-button" variant="secondary" disabled={controls.truthDisabled}
                    data-menu-close aria-pressed={state.truthVisible} onClick={() => engine.toggleTruth()}>{controls.truthLabel}</Button>}
                  <Button variant="ghost" data-menu-close onClick={() => engine.openEvidence()}>The evidence</Button>
                  <Button variant="ghost" data-menu-close onClick={() => engine.setHowOpen(true)}>How it works</Button>
                  <label className="baseline-option"><input type="checkbox" checked={state.baselineOn}
                    disabled={controls.baselineDisabled} onChange={() => engine.toggleBaseline()} />Show recorded baseline</label>
                  {results.metrics.length > 0 && <dl className="menu-metrics">
                    {results.metrics.map(metric => <div key={metric.label}><dt>{metric.label}</dt><dd>{metric.value}</dd></div>)}
                  </dl>}
                  <div className="menu-divider" />
                  <Button id="presentation-button" variant="ghost" data-menu-close aria-pressed={state.presenter}
                    onClick={() => engine.togglePresenter()}><MonitorPlay size={18} />{state.presenter ? 'Leave presentation' : 'Presentation mode'}</Button>
                  <Button variant="ghost" data-menu-close disabled={controls.resetStoryDisabled} onClick={() => engine.resetStory()}><RotateCcw size={17} />Start over</Button>
                  <a href="/lab" className="menu-link">Calibration lab</a>
                </PanelMenu>
                <WorkspaceResults engine={engine} controls={controls} results={results} state={state} />
              </article>
            </section>
          </main>
          <Dialogs engine={engine} results={results} />
          <div className="sr-only" role="status" aria-live="polite" key={state.announcement.seq}>{state.announcement.text}</div>
          <Toaster theme="dark" position="bottom-right" />
        </div>
      </TooltipProvider>
    </MotionConfig>
  )
}

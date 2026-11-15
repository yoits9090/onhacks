import { ArrowUpRight, LocateFixed, LoaderCircle } from 'lucide-react'
import type { DerivedControls, DerivedResults, EngineState, OffgridEngine } from '@/lib/engine'
import { formatPoint } from '@/lib/geo'
import { Button } from '@/components/ui/button'

export default function WorkspaceResults({ engine, controls, results, state }: {
  engine: OffgridEngine; controls: DerivedControls; results: DerivedResults; state: EngineState
}) {
  const headline = results.metrics.find(metric => !metric.truthError) || results.metrics[0]
  const label = state.busy ? controls.analyzeLabel
    : controls.analyzeDisabled && results.hasResult ? 'Result shown'
      : state.operations.length ? 'Compare edit' : 'Find result'
  return (
    <div className={`results-dock ${results.hasResult ? 'revealed' : ''}`}>
      {results.hasResult && <div className="result-readout">
        <p className="result-provenance"><span />{results.provenance}</p>
        {headline && <div className={`headline-metric ${headline.truthError ? 'truth-error' : ''}`}>
          <strong>{headline.value}</strong><span>{headline.label}</span>
        </div>}
        <dl className="coordinate-list">{results.rows.map(row => <div className="coordinate-row" key={row.key}>
          <dt><i className={`point-dot ${row.key}`} />{row.label}</dt>
          <dd><a className="google-maps-link"
            href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${row.point.lat},${row.point.lon}`)}`}
            target="_blank" rel="noopener noreferrer" referrerPolicy="no-referrer"
            aria-label={`Open ${row.label.toLowerCase()} in Google Maps`}
            title="Open Google Maps in a new tab. Shares these coordinates with Google.">
            {formatPoint(row.point)}<ArrowUpRight size={15} />
          </a></dd>
        </div>)}</dl>
        <p className="result-footnote">{results.hasComparison ? 'A changed estimate is not proof of privacy.' : 'An estimate, not a verified location.'}</p>
      </div>}
      <div className="result-actions">
        {!results.hasResult && <span className="result-source-label">{state.example && !state.operations.length ? 'Recorded example' : 'Live inference'}</span>}
        <Button id="analyze-button" className="analyze-button" title={controls.analyzeLabel} disabled={controls.analyzeDisabled} onClick={() => void engine.analyze()}>
          {state.busy ? <LoaderCircle size={21} className="animate-spin" /> : <LocateFixed size={21} />}{label}{!state.busy && <ArrowUpRight size={20} />}
        </Button>
      </div>
      {state.busy ? <div className="inference-wait" role="status"><span>{state.waitStage} <b>{state.waitElapsed}s</b></span>
        <button onClick={() => engine.cancelRequest()}>Cancel</button>
        {state.showContinueEvidence && <button onClick={() => engine.continueToEvidence()}>Continue to evidence</button>}
      </div> : null}
    </div>
  )
}

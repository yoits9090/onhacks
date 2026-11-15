/* EvidenceContent — port of renderReport: frozen evaluation report,
 * deployed configuration, honesty disclaimers. */
import { Suspense, lazy } from 'react'
import type { DerivedResults, EngineState } from '@/lib/engine'
import { formatNumber, isNumber } from '@/lib/geo'
import { cn } from '@/lib/utils'

const MetricChart = lazy(() => import('./MetricChart'))

const METRIC_ROWS: [key: string, label: string, unit: string][] = [
  ['geoscore', 'GeoScore', ''],
  ['acc@25km', 'Within 25 km', '%'],
  ['acc@1km', 'Within 1 km', '%'],
]

interface EvidenceContentProps {
  state: EngineState
  results: DerivedResults
}

export default function EvidenceContent({ state, results }: EvidenceContentProps) {
  const report = state.status?.report
  const config = state.status?.prediction_config

  if (!report || typeof report !== 'object') {
    return (
      <p className="text-xs text-muted-foreground">
        The evaluation report is unavailable. No performance figures are shown.
        Local editing still works.
      </p>
    )
  }

  const accepted = report.accept_candidate === true
  const rejected = report.accept_candidate === false
  const baseline = report.baseline || {}
  const selected = report.selected || {}
  const deployed = accepted ? selected : rejected ? baseline : null
  const interval = report.paired_sequence_bootstrap_95pct
  const adjusted = report.paired_sequence_bootstrap_adjusted
  const weights = (values: unknown) =>
    values && typeof values === 'object'
      ? Object.entries(values as Record<string, number>)
          .map(([key, value]) => `${key}: ${value}`)
          .join(', ') || 'Not reported'
      : 'Not reported'
  const acc1kmDeclines =
    isNumber(selected['acc@1km']) &&
    isNumber(baseline['acc@1km']) &&
    (selected['acc@1km'] as number) < (baseline['acc@1km'] as number)

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-sm font-medium text-foreground">A result you can inspect.</h3>
        <p className="font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
          OSV-5M development subset · not an official benchmark
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="overflow-x-auto border border-border">
          <table className="w-full text-left text-[11px]">
            <caption className="sr-only">
              Evaluation metrics derived from the server report
            </caption>
            <thead>
              <tr className="border-b border-border bg-surface-2 font-mono text-[9px] tracking-[0.14em] text-muted-foreground uppercase">
                <th scope="col" className="px-2.5 py-1.5 font-normal">Metric</th>
                <th scope="col" className="px-2.5 py-1.5 font-normal">Baseline</th>
                <th scope="col" className="px-2.5 py-1.5 font-normal">
                  {accepted ? 'Deployed candidate' : rejected ? 'Deployed baseline' : 'Deployment unknown'}
                </th>
                {!accepted && (
                  <th scope="col" className="px-2.5 py-1.5 font-normal">Selected candidate</th>
                )}
              </tr>
            </thead>
            <tbody>
              {METRIC_ROWS.map(([key, label, unit]) => {
                const baseValue = baseline[key]
                const values = [baseValue, deployed?.[key], ...(!accepted ? [selected[key]] : [])]
                return (
                  <tr key={key} className="border-b border-border/60 last:border-0">
                    <th scope="row" className="px-2.5 py-1.5 font-normal text-muted-foreground">
                      {label}
                    </th>
                    {values.map((value, index) => (
                      <td
                        key={index}
                        className={cn(
                          'coord-mono px-2.5 py-1.5',
                          index === 0 && 'text-muted-foreground',
                          index > 0 &&
                            isNumber(value) &&
                            isNumber(baseValue) &&
                            (value as number) < (baseValue as number)
                            ? 'font-semibold text-[#f06548]'
                            : index > 0
                              ? 'text-foreground'
                              : '',
                        )}
                      >
                        {`${formatNumber(value, 2)}${isNumber(value) ? unit : ''}`}
                        {index > 0 &&
                          isNumber(value) &&
                          isNumber(baseValue) &&
                          (value as number) < (baseValue as number) && (
                            <span className="ml-1 font-mono text-[8px] uppercase tracking-widest">
                              decline
                            </span>
                          )}
                      </td>
                    ))}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-2 self-start font-mono text-[11px]">
          <dt className="text-[10px] text-muted-foreground">Evaluation queries</dt>
          <dd className="text-right text-foreground">
            {formatNumber(report.evaluation_queries, 0)}
          </dd>
          <dt className="text-[10px] text-muted-foreground">Evaluation sequences</dt>
          <dd className="text-right text-foreground">
            {formatNumber(report.evaluation_sequences, 0)}
          </dd>
          <dt className="text-[10px] text-muted-foreground">Candidate GeoScore delta</dt>
          <dd className="text-right text-[#f0a828]">
            {formatNumber(report.mean_geoscore_delta, 4)}
          </dd>
          <dt className="text-[10px] text-muted-foreground">Paired sequence 95% interval</dt>
          <dd className="text-right text-foreground">
            {Array.isArray(interval) && interval.length === 2
              ? `[${formatNumber(interval[0], 4)}, ${formatNumber(interval[1], 4)}]`
              : 'Not reported'}
          </dd>
          {Array.isArray(adjusted) && (
            <>
              <dt className="text-[10px] text-muted-foreground">
                {`Adjusted ${formatNumber(report.familywise_confidence_percent, 2)}% interval`}
              </dt>
              <dd className="text-right text-foreground">
                {`[${adjusted.map(n => formatNumber(n, 4)).join(', ')}]`}
              </dd>
            </>
          )}
          <dt className="text-[10px] text-muted-foreground">Deployed top_n</dt>
          <dd className="text-right text-foreground">
            {config?.top_n != null ? String(config.top_n) : 'Not reported'}
          </dd>
        </dl>
      </div>

      {deployed && (
        <Suspense
          fallback={
            <div
              className="grid gap-3 sm:grid-cols-3"
              aria-hidden="true"
            >
              <div className="h-20 border border-border bg-surface-2/40" />
              <div className="h-20 border border-border bg-surface-2/40" />
              <div className="h-20 border border-border bg-surface-2/40" />
            </div>
          }
        >
          <MetricChart
            baseline={baseline}
            deployed={deployed}
            deployedLabel={accepted ? 'Deployed' : 'Deployed base'}
          />
        </Suspense>
      )}

      <p className="text-xs text-foreground">
        {accepted
          ? 'Candidate accepted and deployed.'
          : rejected
            ? 'Candidate rejected; baseline retained.'
            : 'Deployment decision not reported.'}
        {report.decision ? ` Report decision: ${String(report.decision)}.` : ''}
      </p>
      {acc1kmDeclines && (
        <p className="border-l-2 border-[#f06548] pl-2.5 text-xs text-[#f06548]">
          The selected candidate’s within-1-km rate declines from{' '}
          {formatNumber(baseline['acc@1km'], 2)}% to {formatNumber(selected['acc@1km'], 2)}%.
          A higher aggregate GeoScore does not mean every location threshold improves.
        </p>
      )}
      {report.scope && (
        <p className="text-[11px] text-muted-foreground">
          Report scope: {String(report.scope)}
        </p>
      )}
      <p className="font-mono text-[10px] text-muted-foreground">
        Deployed tower weights — {weights(config?.tower_weights)}. Selected candidate
        weights — {weights(report.selected_tower_weights)}. Configuration:{' '}
        {config?.kind || 'not reported'}.
      </p>
      <p className="text-[10px] text-muted-foreground/90">
        Percentages are evaluation hit rates, not confidence in this photo. The paired
        interval describes the candidate-minus-baseline GeoScore difference; it is not a
        location radius.
      </p>

      <div className="border-t border-border pt-3">
        <p className="font-mono text-[10px] text-foreground">{results.historicalText}</p>
        <p className="mt-1 font-mono text-[10px] text-muted-foreground">
          {results.liveSettingsText}
        </p>
        <p className="mt-1 text-[10px] text-muted-foreground/80">{results.baselineNote}</p>
      </div>

      <div className="grid gap-3 border-t border-border pt-3 sm:grid-cols-3">
        <div>
          <h4 className="text-[11px] font-medium text-foreground">
            What this does not tell you
          </h4>
          <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
            This model is built around street photos. It is not a GPS replacement.
            Distance from a known location measures error, not confidence. A changed
            estimate is not proof of privacy; visual clues can remain after editing.
          </p>
        </div>
        <div>
          <h4 className="text-[11px] font-medium text-foreground">
            What leaves your browser
          </h4>
          <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
            Metadata inspection, covers, crops and downloads happen locally. Live
            analysis sends only re-encoded pixels to server memory, without upload
            storage. OpenStreetMap tiles receive your IP address and the map area,
            not your photo.
          </p>
        </div>
        <div>
          <h4 className="text-[11px] font-medium text-foreground">
            Recorded versus live pixels
          </h4>
          <p className="mt-1 text-[10px] leading-relaxed text-muted-foreground">
            Examples are quality-90 JPEG display exports of evaluation images.
            Historical predictions were made on different source pixels. Live edit
            comparisons reanalyze the normalized display original and its edit, and
            require matching model settings. A chosen cover is an illustration, not
            aggregate evidence.
          </p>
          <p className="mt-1 text-[10px] text-muted-foreground/70">
            Research: Chiikabu Labs Chipoint v2, SigLIP2 / OpenCLIP, Meta DINOv3 and
            OSV-5M.{' '}
            <a href="/vendor/ATTRIBUTION.txt" className="underline underline-offset-2 hover:text-foreground">
              Frontend libraries & map attribution
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}

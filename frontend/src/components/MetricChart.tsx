/* MetricChart — compact baseline-vs-deployed comparison for the evidence
 * dialog. Lazily loaded with the dialog; one mini chart per metric because
 * the units and scales differ and must not share an axis. */
import { Bar, BarChart, Cell, XAxis, YAxis } from 'recharts'
import { useReducedMotion } from 'framer-motion'
import { formatNumber, isNumber } from '@/lib/geo'

interface MetricChartProps {
  baseline: Record<string, unknown>
  deployed: Record<string, unknown>
  deployedLabel: string
}

const ROWS: { key: string; label: string; unit: string }[] = [
  { key: 'geoscore', label: 'GeoScore', unit: '' },
  { key: 'acc@25km', label: 'Within 25 km', unit: '%' },
  { key: 'acc@1km', label: 'Within 1 km', unit: '%' },
]

const BASE_FILL = '#b8c5b1'
const DEPLOY_FILL = '#cc4e36'

export default function MetricChart({ baseline, deployed, deployedLabel }: MetricChartProps) {
  const reduce = useReducedMotion()
  return (
    <div className="grid gap-3 sm:grid-cols-3" role="group" aria-label="Baseline versus deployed metrics chart">
      {ROWS.map(({ key, label, unit }) => {
        const base = baseline[key]
        const dep = deployed[key]
        const declines =
          isNumber(base) && isNumber(dep) && (dep as number) < (base as number)
        const data = [
          { name: 'Baseline', value: isNumber(base) ? (base as number) : 0 },
          { name: deployedLabel, value: isNumber(dep) ? (dep as number) : 0 },
        ]
        return (
          <figure key={key} className="border border-border bg-surface-2/40 px-2 pt-2 pb-1">
            <figcaption className="flex items-baseline justify-between gap-2">
              <span className="font-mono text-[9px] tracking-[0.16em] text-muted-foreground uppercase">
                {label}
              </span>
              <span className="coord-mono text-[9px] text-foreground/80">
                {formatNumber(base, 2)}
                {unit} → {formatNumber(dep, 2)}
                {unit}
              </span>
              {declines && (
                <span className="font-mono text-[8px] tracking-[0.14em] text-[#f06548] uppercase">
                  decline
                </span>
              )}
            </figcaption>
            <BarChart
              layout="vertical"
              data={data}
              width={220}
              height={56}
              margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
            >
              <XAxis type="number" hide domain={[0, 'dataMax']} />
              <YAxis
                type="category"
                dataKey="name"
                width={62}
                tick={{ fill: '#63725c', fontSize: 10, fontFamily: 'Manrope' }}
                axisLine={false}
                tickLine={false}
              />
              <Bar
                dataKey="value"
                barSize={9}
                radius={[0, 2, 2, 0]}
                isAnimationActive={!reduce}
                animationDuration={280}
                animationEasing="ease-out"
              >
                {data.map((entry, index) => (
                  <Cell
                    key={entry.name}
                    fill={index === 0 ? BASE_FILL : DEPLOY_FILL}
                  />
                ))}
              </Bar>
            </BarChart>
          </figure>
        )
      })}
    </div>
  )
}

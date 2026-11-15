/* Geographic + formatting helpers. Direct port of demo/exposure.js. */
import type { GeoPoint, PredictPayload, StatusResponse } from './types'

export const isNumber = (n: unknown): n is number =>
  typeof n === 'number' && Number.isFinite(n)

export const point = (lat: unknown, lon: unknown): GeoPoint | null =>
  isNumber(lat) && isNumber(lon) && Math.abs(lat) <= 90 && Math.abs(lon) <= 180
    ? { lat, lon }
    : null

export const formatNumber = (value: unknown, digits = 2): string =>
  isNumber(value)
    ? value.toLocaleString('en-US', {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : '—'

export const formatPoint = (p: GeoPoint | null): string =>
  p
    ? `${Math.abs(p.lat).toFixed(4)}° ${p.lat < 0 ? 'S' : 'N'}, ${Math.abs(p.lon).toFixed(4)}° ${p.lon < 0 ? 'W' : 'E'}`
    : '—'

export const formatDistance = (km: number): string =>
  `${formatNumber(km, km < 10 ? 2 : 1)} km`

export function distance(a: GeoPoint, b: GeoPoint): number {
  const r = Math.PI / 180
  const h =
    Math.sin(((b.lat - a.lat) * r) / 2) ** 2 +
    Math.cos(a.lat * r) * Math.cos(b.lat * r) * Math.sin(((b.lon - a.lon) * r) / 2) ** 2
  return (
    6371.0088 *
    2 *
    Math.atan2(Math.sqrt(Math.max(0, Math.min(1, h))), Math.sqrt(Math.max(0, 1 - h)))
  )
}

/* ---- model configuration keys (parity checks for edit comparisons) ---- */

export function configurationKey(config: PredictPayload | undefined | null): string | null {
  if (
    !config ||
    !Number.isInteger(config.top_n) ||
    !config.tower_weights ||
    typeof config.tower_weights !== 'object' ||
    Array.isArray(config.tower_weights)
  )
    return null
  const weights = Object.entries(config.tower_weights).sort(([a], [b]) =>
    a.localeCompare(b),
  )
  if (!weights.length || weights.some(([, value]) => !isNumber(value))) return null
  return JSON.stringify([
    config.top_n,
    weights,
    config.variant ?? null,
    config.calibrated ?? null,
  ])
}

export function statusConfigurationKey(status: StatusResponse | null): string {
  const config = status?.prediction_config
  return JSON.stringify([
    configurationKey(config),
    config?.model_revision ?? null,
    config?.retrieval_k ?? null,
  ])
}

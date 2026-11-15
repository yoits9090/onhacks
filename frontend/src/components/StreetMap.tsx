import { useEffect, useRef, useState } from 'react'
import * as L from 'leaflet'
import { useReducedMotion } from 'framer-motion'
import type { MapPoint } from '@/lib/types'
import { formatPoint } from '@/lib/geo'

const COLORS = { original: '#efbe84', edited: '#80caff', truth: '#a6d7b1', baseline: '#c2b4df' }

export default function StreetMap({ points, ariaLabel }: { points: MapPoint[]; ariaLabel: string }) {
  const host = useRef<HTMLDivElement>(null)
  const map = useRef<L.Map | null>(null)
  const markers = useRef<L.LayerGroup | null>(null)
  const [tileFailure, setTileFailure] = useState(false)
  const reduceMotion = useReducedMotion()
  useEffect(() => {
    if (!host.current) return
    const view = L.map(host.current, { zoomControl: false, scrollWheelZoom: false, minZoom: 1, maxZoom: 18, worldCopyJump: true })
    map.current = view
    view.setView([26, 20], 2)
    view.attributionControl.setPrefix(false)
    L.control.zoom({ position: 'topright' }).addTo(view)
    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
      maxZoom: 19,
    }).addTo(view)
    tiles.on('tileerror', () => setTileFailure(true))
    markers.current = L.layerGroup().addTo(view)
    let active = true
    fetch('/assets/land-110m.geojson').then(response => response.json()).then(land => {
      if (!active) return
      view.createPane('local-land')
      view.getPane('local-land')!.style.zIndex = '150'
      L.geoJSON(land, { pane: 'local-land', interactive: false, style: { color: '#343b40', weight: 1, fillColor: '#1d2327', fillOpacity: 1 } }).addTo(view)
    }).catch(() => {})
    const observer = new ResizeObserver(() => view.invalidateSize({ animate: false }))
    observer.observe(host.current)
    return () => { active = false; observer.disconnect(); view.remove(); map.current = null; markers.current = null }
  }, [])
  const pointKey = JSON.stringify(points)
  useEffect(() => {
    const view = map.current
    const layer = markers.current
    if (!view || !layer) return
    layer.clearLayers()
    for (const p of points) {
      const marker = L.circleMarker([p.lat, p.lon], {
        color: '#fff', weight: 3, fillColor: COLORS[p.key], fillOpacity: 1, radius: p.key === 'truth' ? 7 : 10,
      }).addTo(layer)
      marker.bindTooltip(`${p.label}: ${formatPoint(p)}`, { className: 'street-tooltip' })
    }
    if (points.length) {
      const anchor = points[0].lon
      const positions = points.map(p => L.latLng(p.lat, p.lon + Math.round((anchor - p.lon) / 360) * 360))
      view.fitBounds(L.latLngBounds(positions), {
        paddingTopLeft: [45, 80], paddingBottomRight: [45, view.getSize().y < 500 ? 240 : 340], maxZoom: points.length === 1 ? 6 : 10,
        animate: !reduceMotion, duration: 0.55,
      })
      if (points.length > 1) L.polyline(positions, { color: '#737c74', weight: 1.5, dashArray: '5 7', opacity: 0.65 }).addTo(layer)
    } else view.setView([26, 20], 2, { animate: false })
    // Point identity changes only when an actual estimate or truth changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pointKey, reduceMotion])
  return <div className="street-map">
    <div id="map" ref={host} role="img" aria-label={ariaLabel} />
    {tileFailure && <p className="map-load-note">Some street tiles are unavailable. A local overview remains underneath.</p>}
  </div>
}

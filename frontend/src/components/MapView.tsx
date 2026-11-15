/* MapView — dark Leaflet instrument. Port of exposure.js initMap/drawMap/
 * renderGeographicLabels/toggleTiles + the radar-sweep reveal moment. */
import { Suspense, lazy, useCallback, useEffect, useRef, useState } from 'react'
import * as L from 'leaflet'
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion'
import { ExternalLink, Globe } from 'lucide-react'
import type { MapPoint } from '@/lib/types'
import { formatPoint } from '@/lib/geo'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

const GlobeView = lazy(() => import('./GlobeView'))

const PIN_COLORS: Record<MapPoint['key'], string> = {
  original: '#f0a828',
  edited: '#5fd4d0',
  truth: '#f06548',
  baseline: '#b1a1db',
}

const LAND_STYLE: L.PathOptions = {
  color: '#33465f',
  weight: 0.7,
  fillColor: '#1a2738',
  fillOpacity: 1,
}
const GRID_STYLE: L.PathOptions = { color: '#223349', weight: 0.6 }

interface MapViewProps {
  points: MapPoint[]
  ariaLabel: string
}

export default function MapView({ points, ariaLabel }: MapViewProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<L.Map | null>(null)
  const markerLayerRef = useRef<L.LayerGroup | null>(null)
  const labelLayerRef = useRef<L.LayerGroup | null>(null)
  const tilesRef = useRef<L.TileLayer | null>(null)
  const tileGenerationRef = useRef(0)
  const geographicReadyRef = useRef(false)
  const [tilesOn, setTilesOn] = useState(false)
  const tilesOnRef = useRef(false)
  const [mapReady, setMapReady] = useState(false)
  const [fallback, setFallback] = useState<string | null>(null)
  const [mapNote, setMapNote] = useState('Local world map · no external requests')
  const [sweepKey, setSweepKey] = useState(0)
  const revealedRef = useRef(false)
  const drawTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reduceMotion = useReducedMotion()
  const [globeOn, setGlobeOn] = useState(false)
  const [webglOk] = useState(() => {
    try {
      const probe = document.createElement('canvas')
      return Boolean(
        window.WebGLRenderingContext &&
          (probe.getContext('webgl2') || probe.getContext('webgl')),
      )
    } catch {
      return false
    }
  })

  const applyFallbackMap = useCallback((note: string) => {
    mapRef.current?.remove()
    mapRef.current = null
    setFallback(note)
    setMapNote(note)
  }, [])

  /* Fixed atlas context labels — never reverse-geocoded claims. */
  const renderGeographicLabels = useCallback(() => {
    const map = mapRef.current
    const layer = labelLayerRef.current
    if (!map || !layer) return
    layer.clearLayers()
    if (tilesOnRef.current) return
    const labels: [string, number, number][] =
      map.getZoom() < 3
        ? [
            ['North America', 48, -105],
            ['South America', -15, -60],
            ['Europe', 52, 16],
            ['Africa', 8, 20],
            ['Asia', 42, 100],
            ['Australia', -25, 134],
          ]
        : [
            ['Russia', 62, 85],
            ['Kazakhstan', 48, 67],
            ['China', 36, 103],
            ['India', 22, 79],
            ['Mongolia', 47, 103],
            ['Finland', 64, 26],
            ['Ukraine', 49, 32],
            ['Türkiye', 39, 35],
            ['France', 47, 2],
            ['Spain', 40, -4],
            ['Germany', 51, 10],
            ['Italy', 43, 12],
            ['Canada', 57, -110],
            ['United States', 39, -101],
            ['Mexico', 23, -103],
            ['Brazil', -12, -53],
            ['Argentina', -36, -65],
            ['Australia', -25, 134],
            ['South Africa', -29, 25],
            ['Egypt', 27, 30],
            ['Algeria', 28, 3],
          ]
    labels.forEach(([name, lat, lon]) => {
      const span = document.createElement('span')
      span.textContent = name
      L.marker([lat, lon], {
        icon: L.divIcon({
          className: 'geographic-label',
          html: span,
          iconSize: [100, 18],
          iconAnchor: [50, 9],
        }),
        interactive: false,
        keyboard: false,
      }).addTo(layer)
    })
  }, [])

  /* init once */
  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let cancelled = false
    try {
      const map = L.map(host, {
        zoomControl: true,
        attributionControl: true,
        worldCopyJump: true,
        scrollWheelZoom: false,
        minZoom: 1,
        maxZoom: 17,
      })
      map.attributionControl.setPrefix(false)
      map.setView([22, 15], 1)
      map.attributionControl.addAttribution('Natural Earth')
      // Keep the local geographic fallback beneath optional raster tiles.
      map.createPane('local-world')
      map.getPane('local-world')!.style.zIndex = '150'
      map.getPane('local-world')!.style.pointerEvents = 'none'
      mapRef.current = map
      markerLayerRef.current = L.layerGroup().addTo(map)
      labelLayerRef.current = L.layerGroup().addTo(map)
      map.on('zoomend', renderGeographicLabels)
      renderGeographicLabels()
      for (let lat = -60; lat <= 60; lat += 30) {
        L.polyline(
          [
            [lat, -180],
            [lat, 180],
          ],
          { pane: 'local-world', ...GRID_STYLE, interactive: false },
        ).addTo(map)
      }
      for (let lon = -180; lon <= 180; lon += 30) {
        L.polyline(
          [
            [-80, lon],
            [80, lon],
          ],
          { pane: 'local-world', ...GRID_STYLE, interactive: false },
        ).addTo(map)
      }
      fetch('/assets/land-110m.geojson')
        .then(response => {
          if (!response.ok) throw new Error('Local world map missing')
          return response.json()
        })
        .then(land => {
          if (cancelled || !mapRef.current) return
          L.geoJSON(land, { pane: 'local-world', style: LAND_STYLE, interactive: false })
            .addTo(map)
            .bringToBack()
          geographicReadyRef.current = true
        })
        .catch(() => {
          if (!tilesRef.current) applyFallbackMap('Local outlines unavailable · schematic world overview')
        })
      setMapReady(true)
      const observer = new ResizeObserver(() => {
        map.invalidateSize({ animate: false })
      })
      observer.observe(host)
      return () => {
        cancelled = true
        observer.disconnect()
        map.remove()
        mapRef.current = null
        markerLayerRef.current = null
        labelLayerRef.current = null
        tilesRef.current = null
      }
    } catch {
      applyFallbackMap('Map unavailable · schematic world overview')
      return () => {}
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const drawMarkers = useCallback(
    (list: MapPoint[]) => {
      const map = mapRef.current
      const markerLayer = markerLayerRef.current
      if (!map || !markerLayer) return
      markerLayer.clearLayers()
      map.invalidateSize({ animate: false })
      if (!list.length) {
        map.setView([22, 15], 1, { animate: false })
        return
      }
      // Cut at the largest longitude gap, so markers near ±180° fit across the dateline.
      const lons = list.map(p => (p.lon + 360) % 360).sort((a, b) => a - b)
      let gap = -1
      let start = lons[0]
      for (let i = 0; i < lons.length; i++) {
        const next = lons[(i + 1) % lons.length] + (i === lons.length - 1 ? 360 : 0)
        if (next - lons[i] > gap) {
          gap = next - lons[i]
          start = next % 360
        }
      }
      const positions: [number, number][] = list.map(p => {
        let lon = (p.lon + 360) % 360
        if (lon < start) lon += 360
        if (start > 180) lon -= 360
        return [Math.max(-85, Math.min(85, p.lat)), lon]
      })
      list.forEach((p, index) => {
        const marker = L.circleMarker(positions[index], {
          radius: p.key === 'original' ? 7 : p.key === 'truth' ? 5 : 6,
          color: '#e6ecf4',
          weight: 1.5,
          fillColor: PIN_COLORS[p.key],
          fillOpacity: 1,
          className: 'og-pin',
        }).addTo(markerLayer)
        const element = marker.getElement()
        if (element instanceof SVGElement)
          element.style.animationDelay = `${Math.min(index * 80, 240)}ms`
        const span = document.createElement('span')
        span.textContent = p.label
        marker.bindTooltip(span, { direction: 'top', className: 'offgrid-tip' })
      })
      // The 1:110m local outline is useful at regional scale, not as an empty street map.
      const options: L.FitBoundsOptions = {
        animate: false,
        padding: [36, 35],
        maxZoom: tilesRef.current ? 12 : 3,
      }
      if (positions.length === 1)
        map.setView(positions[0], tilesRef.current ? 8 : 3, { animate: false })
      else map.fitBounds(positions, options)
      if (list.some(p => Math.abs(p.lat) > 85))
        setMapNote('Polar marker clamped at 85° · exact coordinates shown below')
    },
    [],
  )

  /* Points → markers. First reveal delays ~600ms so the sweep lands first.
   * Keyed on serialized coordinates so unrelated renders don't redraw. */
  const pointsRef = useRef(points)
  pointsRef.current = points
  const pointsKey = JSON.stringify(
    points.map(p => [p.key, Number(p.lat.toFixed(6)), Number(p.lon.toFixed(6))]),
  )
  useEffect(() => {
    if (drawTimerRef.current) {
      clearTimeout(drawTimerRef.current)
      drawTimerRef.current = null
    }
    const list = pointsRef.current
    const has = list.length > 0
    const firstReveal = has && !revealedRef.current
    revealedRef.current = has
    if (firstReveal && !reduceMotion) {
      setSweepKey(k => k + 1)
      drawTimerRef.current = setTimeout(() => {
        drawMarkers(pointsRef.current)
        drawTimerRef.current = null
      }, 620)
    } else {
      drawMarkers(list)
    }
    return () => {
      if (drawTimerRef.current) clearTimeout(drawTimerRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pointsKey, drawMarkers, reduceMotion, mapReady])

  const toggleTiles = useCallback(() => {
    const map = mapRef.current
    if (!map) return
    const generation = ++tileGenerationRef.current
    if (tilesRef.current) {
      map.removeLayer(tilesRef.current)
      tilesRef.current = null
      tilesOnRef.current = false
      setTilesOn(false)
      renderGeographicLabels()
      setMapNote(
        geographicReadyRef.current
          ? 'Local world map · no external requests'
          : 'Local coordinate grid · no external requests',
      )
      return
    }
    tilesOnRef.current = true
    setTilesOn(true)
    setMapNote('Loading external OpenStreetMap tiles…')
    let loaded = false
    let errors = 0
    const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 17,
      attribution:
        '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a>',
      crossOrigin: false,
    })
    tiles.on('tileload', () => {
      if (generation !== tileGenerationRef.current) return
      loaded = true
      setMapNote(
        errors
          ? 'Some street tiles unavailable · coordinates remain valid'
          : 'OpenStreetMap · external tiles enabled',
      )
    })
    tiles.on('tileerror', () => {
      if (generation !== tileGenerationRef.current) return
      errors++
      setMapNote(
        loaded
          ? 'Some street tiles unavailable · coordinates remain valid'
          : 'Map tiles unavailable · local overview remains',
      )
    })
    tilesRef.current = tiles
    tiles.addTo(map)
    renderGeographicLabels()
    setTimeout(() => {
      if (generation === tileGenerationRef.current && tilesRef.current && !loaded)
        setMapNote('Map tiles unavailable · local overview remains')
    }, 10000)
  }, [renderGeographicLabels])

  /* Redraw once tiles toggle changes desired max zoom fitting. */
  useEffect(() => {
    drawMarkers(pointsRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tilesOn])

  const reduced = Boolean(reduceMotion)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="relative isolate min-h-0 flex-1 overflow-hidden border-b border-border">
        {fallback === null ? (
          <div
            id="map"
            ref={hostRef}
            className="absolute inset-0"
            role="img"
            aria-label={ariaLabel}
          />
        ) : (
          <svg
            viewBox="0 0 720 360"
            role="img"
            aria-label="Schematic world overview"
            className="absolute inset-0 h-full w-full"
            preserveAspectRatio="xMidYMid slice"
          >
            <defs>
              <pattern id="map-grid" width="60" height="60" patternUnits="userSpaceOnUse">
                <path d="M60 0H0V60" fill="none" stroke="#223349" strokeWidth=".7" />
              </pattern>
            </defs>
            <rect width="720" height="360" fill="#0d1420" />
            <rect width="720" height="360" fill="url(#map-grid)" />
            <g fill="#1a2738" stroke="#33465f" strokeWidth="1">
              <path d="m35 68 45-31 83-8 45 20 27 46-24 19-23-5-26 38-16 23-40-22-10-31-40-17zm162-39 42-15 39 12-14 51-37 19-23-24zm-44 139 44 10 29 34 4 29-32 57-13 20-19-57-18-53zM300 88l30-25 45-13 34-9 101-20 101 24 62 40-26 30-40-5-29 27-30-6-13 42-32-2-35-31-20-10-24 8-15-21-35-2-22 10-16-10-27 8-13-9zm28 42 44-8 27 27 25 29-21 50-25 24-21-13-12-45-23-23zm89 80 11-9 1 25-12 14zm95-51 26 7 11 17 28 5 15 12-43-4-24-17zm64 49 50-14 31 25 3 32-37 6-51-20zm82 58 14-17 4 14-14 18z" />
            </g>
            <g>
              {points.map(p => (
                <circle
                  key={p.key}
                  cx={(p.lon + 180) * 2}
                  cy={(90 - p.lat) * 2}
                  r="5"
                  fill={PIN_COLORS[p.key]}
                  stroke="#fff"
                  strokeWidth="1.5"
                >
                  <title>{`${p.label}: ${formatPoint(p)}`}</title>
                </circle>
              ))}
            </g>
          </svg>
        )}

        {/* radar-sweep reveal — the one loud moment */}
        <AnimatePresence>
          {sweepKey > 0 && !reduced && (
            <motion.div
              key={sweepKey}
              className="pointer-events-none absolute inset-0 z-[500]"
              initial={{ opacity: 1 }}
              animate={{ opacity: 0 }}
              transition={{ delay: 0.62, duration: 0.18 }}
            >
              <motion.div
                className="absolute inset-0"
                style={{
                  background:
                    'conic-gradient(from 0deg at 50% 50%, transparent 0deg, rgba(95,212,208,0.20) 45deg, rgba(240,168,40,0.10) 70deg, transparent 95deg)',
                }}
                initial={{ rotate: 0 }}
                animate={{ rotate: 360 }}
                transition={{ duration: 0.66, ease: 'linear' }}
              />
              <motion.div
                className="absolute top-1/2 left-1/2 h-40 w-40 -translate-x-1/2 -translate-y-1/2 rounded-full border border-[#5fd4d0]/60"
                initial={{ scale: 0.15, opacity: 1 }}
                animate={{ scale: 3.2, opacity: 0 }}
                transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
              />
              <motion.div
                className="absolute top-1/2 left-1/2 h-10 w-10 -translate-x-1/2 -translate-y-1/2"
                initial={{ opacity: 1 }}
                animate={{ opacity: 0 }}
                transition={{ duration: 0.55 }}
              >
                <span className="absolute top-1/2 left-0 h-px w-full bg-[#f0a828]/80" />
                <span className="absolute left-1/2 top-0 w-px h-full bg-[#f0a828]/80" />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* idle crosshair — instrument at rest. The label rides low in an
            ocean-quiet band so it never collides with atlas labels. */}
        {points.length === 0 && !globeOn && (
          <div className="pointer-events-none absolute inset-0 z-[400]">
            <div className="absolute top-[42%] left-1/2 h-14 w-14 -translate-x-1/2 -translate-y-1/2 opacity-70">
              <span className="absolute top-1/2 left-0 h-px w-full bg-[#8b9aac]/70" />
              <span className="absolute left-1/2 top-0 h-full w-px bg-[#8b9aac]/70" />
              <span className="absolute inset-0 rounded-full border border-[#8b9aac]/40" />
              <span className="absolute inset-[18px] rounded-full border border-[#f0a828]/50" />
            </div>
            <div className="absolute inset-x-0 bottom-12 flex justify-center">
              <div className="rounded-sm border border-border/60 bg-[#0a0d12]/75 px-3 py-1.5 text-center backdrop-blur-sm">
                <p className="font-mono text-[10px] tracking-[0.28em] text-muted-foreground uppercase">
                  Estimate hidden
                </p>
                <p className="mt-0.5 text-xs text-muted-foreground/80">
                  Every scene leaves clues.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* lazy 3D globe — flat map stays mounted underneath */}
        {globeOn && webglOk && fallback === null && (
          <div className="absolute inset-0 z-[450] bg-[#0a0d12]">
            <Suspense
              fallback={
                <p className="flex h-full items-center justify-center font-mono text-[10px] tracking-[0.24em] text-muted-foreground uppercase">
                  Loading globe…
                </p>
              }
            >
              <GlobeView points={points} />
            </Suspense>
          </div>
        )}

        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-[400] flex items-center justify-between bg-gradient-to-t from-[#0a0d12]/85 to-transparent px-3 pb-1.5 pt-6">
          <span className="font-mono text-[10px] text-muted-foreground/90">{mapNote}</span>
          <span className="font-mono text-[10px] text-muted-foreground/70" aria-hidden="true">
            ↑ N
          </span>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 px-3 py-1.5">
        <div className="flex min-w-0 items-center gap-2">
          <span className="font-mono text-[10px] tracking-[0.18em] text-muted-foreground uppercase">
            {fallback === null ? 'Tactical overview' : 'Schematic fallback'}
          </span>
          {webglOk && fallback === null && (
            <span
              className="flex items-center border border-border p-0.5"
              role="group"
              aria-label="Map projection"
            >
              {(['flat', 'globe'] as const).map(mode => (
                <button
                  key={mode}
                  type="button"
                  aria-pressed={mode === 'globe' ? globeOn : !globeOn}
                  onClick={() => setGlobeOn(mode === 'globe')}
                  className={cn(
                    'flex items-center gap-1 px-1.5 py-0.5 font-mono text-[9px] tracking-[0.14em] uppercase transition-colors',
                    'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                    (mode === 'globe') === globeOn
                      ? 'bg-[#5fd4d0]/15 text-[#5fd4d0]'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                >
                  {mode === 'globe' && <Globe className="size-2.5" aria-hidden="true" />}
                  {mode}
                </button>
              ))}
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="sm"
          className="h-6 gap-1.5 px-2 font-mono text-[10px] tracking-wide text-muted-foreground hover:text-foreground"
          aria-pressed={tilesOn}
          disabled={fallback !== null}
          onClick={toggleTiles}
          title="External OpenStreetMap tiles share your IP and map area"
          aria-label="Toggle external street map tiles; shares your IP and map area with OpenStreetMap"
        >
          <ExternalLink className="size-3" />
          {tilesOn ? 'Use local overview' : 'Enable street map'}
        </Button>
      </div>
    </div>
  )
}

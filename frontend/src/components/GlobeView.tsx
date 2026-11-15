/* GlobeView — lazy 3D companion to the flat Leaflet map. Rendered only when
 * the user asks for it; Leaflet stays mounted underneath and unchanged. */
import { useEffect, useRef } from 'react'
import Globe from 'globe.gl'
import { useReducedMotion } from 'framer-motion'
import type { MapPoint } from '@/lib/types'
import { formatPoint } from '@/lib/geo'

const PIN_COLORS: Record<MapPoint['key'], string> = {
  original: '#f0a828',
  edited: '#5fd4d0',
  truth: '#f06548',
  baseline: '#b1a1db',
}

type GlobeInstance = InstanceType<typeof Globe>

interface GlobeViewProps {
  points: MapPoint[]
}

export default function GlobeView({ points }: GlobeViewProps) {
  const hostRef = useRef<HTMLDivElement>(null)
  const globeRef = useRef<GlobeInstance | null>(null)
  const reduceMotion = useReducedMotion()
  const reduceRef = useRef(reduceMotion)
  reduceRef.current = reduceMotion

  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let disposed = false
    let globe: GlobeInstance
    try {
      globe = new Globe(host, { rendererConfig: { antialias: true, alpha: true } })
    } catch {
      host.innerHTML =
        '<p class="flex h-full items-center justify-center font-mono text-[10px] tracking-[0.24em] text-[#8b9aac] uppercase">Globe unavailable — flat map unaffected</p>'
      return
    }
    globeRef.current = globe
    globe
      .backgroundColor('rgba(10,13,18,0)')
      .width(host.clientWidth)
      .height(host.clientHeight)
      .showAtmosphere(false)
      .showGraticules(true)
      .polygonsData([])
      .polygonCapColor(() => '#1a2738')
      .polygonSideColor(() => 'rgba(0,0,0,0)')
      .polygonStrokeColor(() => '#33465f')
      .polygonAltitude(0.012)
      .pointsData([])
      .pointColor((p: object) => PIN_COLORS[(p as MapPoint).key])
      .pointAltitude(0.025)
      .pointRadius((p: object) => ((p as MapPoint).key === 'original' ? 0.8 : 0.6))
      .pointLabel(
        (p: object) => `${(p as MapPoint).label}: ${formatPoint(p as MapPoint)}`,
      )
      .ringsData([])
      .ringColor(() => (t: number) => `rgba(240,168,40,${Math.max(0, 1 - t)})`)
      .ringMaxRadius(6)
      .ringPropagationSpeed(1.3)
      .ringRepeatPeriod(1900)
      .ringAltitude(0.014)
    const material = globe.globeMaterial()
    if (material && 'color' in material)
      (material as { color: { set: (c: string) => void } }).color.set('#0e1520')

    const controls = globe.controls()
    controls.autoRotate = !reduceRef.current
    controls.autoRotateSpeed = 0.45
    controls.enableZoom = true
    controls.enablePan = false

    fetch('/assets/land-110m.geojson')
      .then(response => (response.ok ? response.json() : null))
      .then(land => {
        if (disposed || !land?.features?.length) return
        globe.polygonsData(land.features)
      })
      .catch(() => {})

    const observer = new ResizeObserver(() => {
      globe.width(host.clientWidth).height(host.clientHeight)
    })
    observer.observe(host)
    globe.pointOfView({ lat: 28, lng: 15, altitude: 2.3 }, 0)

    return () => {
      disposed = true
      observer.disconnect()
      globeRef.current = null
      try {
        globe.pauseAnimation()
        globe._destructor()
      } catch {}
      host.innerHTML = ''
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const pointsKey = JSON.stringify(
    points.map(p => [p.key, p.label, Number(p.lat.toFixed(6)), Number(p.lon.toFixed(6))]),
  )
  useEffect(() => {
    const globe = globeRef.current
    if (!globe) return
    globe.pointsData(points.map(p => ({ ...p, lng: p.lon })))
    const focus = points.find(p => p.key === 'original') ?? points[0]
    if (reduceRef.current) {
      globe.ringsData([])
    } else {
      globe.ringsData(focus ? [{ ...focus, lng: focus.lon }] : [])
    }
    if (focus)
      globe.pointOfView(
        { lat: focus.lat, lng: focus.lon, altitude: points.length > 1 ? 2.5 : 2 },
        reduceRef.current ? 0 : 800,
      )
    else globe.pointOfView({ lat: 28, lng: 15, altitude: 2.3 }, reduceRef.current ? 0 : 800)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pointsKey])

  return <div ref={hostRef} className="h-full w-full" aria-hidden="true" />
}

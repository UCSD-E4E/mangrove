import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from 'react'
import DeckGL from '@deck.gl/react'
import { MapView, FlyToInterpolator } from '@deck.gl/core'
import { ScatterplotLayer, BitmapLayer, TextLayer } from '@deck.gl/layers'
import { TileLayer } from '@deck.gl/geo-layers'
import { regions } from '../data/regions'
import type { Region } from '../data/regions'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? ''

export type GlobeViewerHandle = {
  flyToRegion(id: string): void
  resetView(): void
}

type ViewState = {
  longitude: number
  latitude: number
  zoom: number
  pitch?: number
  bearing?: number
  transitionDuration?: number | 'auto'
  transitionInterpolator?: object
}

const INITIAL_VIEW_STATE: ViewState = {
  longitude: 0,
  latitude: 20,
  zoom: 1.5,
  pitch: 0,
  bearing: 0,
}

const TRAINED_COLOR: [number, number, number] = [61, 107, 74]
const TESTING_COLOR: [number, number, number] = [170, 170, 170]

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function satelliteLayer() {
  return new TileLayer({
    id: 'satellite-base',
    data: `https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg?access_token=${MAPBOX_TOKEN}`,
    maxZoom: 19,
    minZoom: 0,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    renderSubLayers: (props: any) => {
      const [[west, south], [east, north]] = props.tile.boundingBox
      return new BitmapLayer(props, {
        data: undefined,
        image: props.data,
        bounds: [west, south, east, north],
      })
    },
  })
}

type Props = {
  onRegionClick?: (id: string) => void
}

export const GlobeViewer = forwardRef<GlobeViewerHandle, Props>(function GlobeViewer({ onRegionClick }, ref) {
  const [viewState, setViewState] = useState<ViewState>(INITIAL_VIEW_STATE)
  const rotatingRef = useRef(true)
  const rafRef = useRef<number>(0)

  useImperativeHandle(ref, () => ({
    flyToRegion(id: string) {
      const region = regions.find((r) => r.id === id)
      if (!region) {
        console.warn(`GlobeViewer: unknown region "${id}"`)
        return
      }
      rotatingRef.current = false
      setViewState((vs) => ({
        ...vs,
        longitude: region.coords.lng,
        latitude: region.coords.lat,
        zoom: region.zoom,
        pitch: 52,
        bearing: 0,
        transitionDuration: 2200,
        transitionInterpolator: new FlyToInterpolator({ speed: 1.5 }),
      }))
    },
    resetView() {
      rotatingRef.current = true
      setViewState(INITIAL_VIEW_STATE)
    },
  }))

  useEffect(() => {
    const animate = () => {
      if (rotatingRef.current) {
        setViewState((vs) => ({ ...vs, longitude: (vs.longitude + 0.025) % 360 }))
      }
      rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => cancelAnimationFrame(rafRef.current)
  }, [])

  const layers = [
    satelliteLayer(),

    // Soft outer glow behind each marker
    new ScatterplotLayer<Region>({
      id: 'region-glow',
      data: regions,
      getPosition: (d) => [d.coords.lng, d.coords.lat, 0],
      getFillColor: [255, 255, 255, 30] as [number, number, number, number],
      getRadius: 220000,
      radiusUnits: 'meters',
      radiusMinPixels: 20,
      pickable: false,
    }),

    // Solid marker dot — this is the primary interactive layer
    new ScatterplotLayer<Region>({
      id: 'regions',
      data: regions,
      getPosition: (d) => [d.coords.lng, d.coords.lat, 0],
      getFillColor: (d) => (d.status === 'trained' ? TRAINED_COLOR : TESTING_COLOR),
      getRadius: 70000,
      radiusUnits: 'meters',
      radiusMinPixels: 8,
      stroked: true,
      getLineColor: [255, 255, 255, 210] as [number, number, number, number],
      lineWidthMinPixels: 2,
      pickable: true,
      onClick: ({ object }) => {
        if (object && onRegionClick) onRegionClick(object.id)
      },
    }),

    // Region name labels below each marker
    new TextLayer<Region>({
      id: 'region-labels',
      data: regions,
      getPosition: (d) => [d.coords.lng, d.coords.lat, 0],
      getText: (d) => d.name.toUpperCase(),
      getSize: 11,
      getColor: [255, 255, 255, 200] as [number, number, number, number],
      getTextAnchor: 'middle',
      getAlignmentBaseline: 'top',
      getPixelOffset: [0, 20],
      fontFamily: 'Inter, DM Sans, sans-serif',
      fontWeight: 700,
      outlineWidth: 3,
      outlineColor: [0, 0, 0, 180] as [number, number, number, number],
      pickable: false,
    }),
  ]

  return (
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    <DeckGL
      views={[new MapView({ id: 'globe', repeat: true })]}
      viewState={viewState as any}
      onViewStateChange={({ viewState: vs }) => setViewState(vs as ViewState)}
      controller={{ scrollZoom: false }}
      layers={layers}
      style={{ position: 'absolute', inset: '0', background: '#0a0f1a' }}
    />
  )
})

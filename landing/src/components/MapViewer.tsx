import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { TileLayer } from '@deck.gl/geo-layers'
import { BitmapLayer, GeoJsonLayer } from '@deck.gl/layers'
import { PMTiles } from 'pmtiles'
import { parse } from '@loaders.gl/core'
import { MVTLoader } from '@loaders.gl/mvt'
import { regions } from '../data/regions'
import { CLASS_COLORS } from '../lib/classConfig'

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN ?? ''
const INITIAL_CENTER: [number, number] = [0, 20]
const INITIAL_ZOOM = 1.5
const RASTER_MAX_ZOOM = 12

export type MapViewerHandle = {
  flyToRegion(id: string): void
  resetView(): void
  zoomIn(): void
  zoomOut(): void
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const getClassColor = (f: any): [number, number, number, number] =>
  CLASS_COLORS[f.properties?.class_idx ?? 0] ?? CLASS_COLORS[0]

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderVectorSubLayer(props: any) {
  return new GeoJsonLayer({
    ...props,
    getFillColor: getClassColor,
    getLineWidth: 0,
    filled: true,
    stroked: false,
    pickable: false,
  })
}

const pmtilesCache = new Map<string, PMTiles>()

function buildRasterLayer(tilesUrl: string) {
  return new TileLayer({
    id: 'classification-raster',
    data: tilesUrl,
    minZoom: 0,
    maxZoom: RASTER_MAX_ZOOM,
    tileSize: 256,
    renderSubLayers: (props) => {
      if (!props.data) return null
      const { boundingBox } = props.tile
      const eps = 0.0002
      return new BitmapLayer({
        ...props,
        data: undefined,
        image: props.data,
        bounds: [
          boundingBox[0][0] - eps, boundingBox[0][1] - eps,
          boundingBox[1][0] + eps, boundingBox[1][1] + eps,
        ] as [number, number, number, number],
      })
    },
  })
}

function buildVectorLayer(pmtilesUrl: string) {
  let source = pmtilesCache.get(pmtilesUrl)
  if (!source) {
    source = new PMTiles(pmtilesUrl)
    pmtilesCache.set(pmtilesUrl, source)
  }
  const src = source

  return new TileLayer({
    id: 'classification-vector',
    getTileData: async ({ index: { x, y, z } }: { index: { x: number; y: number; z: number } }) => {
      try {
        const tile = await src.getZxy(z, x, y)
        if (!tile?.data) return null
        return await parse(tile.data as ArrayBuffer, MVTLoader, {
          mvt: {
            coordinates: 'wgs84' as const,
            tileIndex: { x, y, z },
            layers: ['landcover'],
          },
          worker: true,
        })
      } catch {
        return null
      }
    },
    renderSubLayers: renderVectorSubLayer,
    minZoom: 0,
    maxZoom: 14,
    maxRequests: 6,
    updateTriggers: { getTileData: [pmtilesUrl] },
  })
}

interface Props {
  tilesUrl?: string
  pmtilesUrl?: string
  defaultCenter?: [number, number]
  defaultZoom?: number
  defaultPitch?: number
}

export const MapViewer = forwardRef<MapViewerHandle, Props>(
  function MapViewer({
    tilesUrl = '',
    pmtilesUrl = '',
    defaultCenter = INITIAL_CENTER,
    defaultZoom = INITIAL_ZOOM,
    defaultPitch = 0,
  }, ref) {
    const containerRef = useRef<HTMLDivElement>(null)
    const mapRef = useRef<maplibregl.Map | null>(null)
    const overlayRef = useRef<MapboxOverlay | null>(null)
    const aboveThresholdRef = useRef(defaultZoom > RASTER_MAX_ZOOM)
    const applyLayersRef = useRef<() => void>(() => {})

    useImperativeHandle(ref, () => ({
      flyToRegion(id: string) {
        const region = regions.find((r) => r.id === id)
        if (!region || !mapRef.current) return
        mapRef.current.flyTo({
          center: [region.coords.lng, region.coords.lat],
          zoom: region.zoom,
          pitch: region.pitch,
          duration: 2200,
        })
      },
      resetView() {
        mapRef.current?.flyTo({ center: INITIAL_CENTER, zoom: INITIAL_ZOOM, pitch: 0, duration: 1500 })
      },
      zoomIn() { mapRef.current?.zoomIn() },
      zoomOut() { mapRef.current?.zoomOut() },
    }))

    useEffect(() => {
      if (!containerRef.current || mapRef.current) return

      const map = new maplibregl.Map({
        container: containerRef.current,
        style: {
          version: 8,
          sources: {
            satellite: {
              type: 'raster',
              tiles: [
                `https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.jpg?access_token=${MAPBOX_TOKEN}`,
              ],
              tileSize: 256,
              attribution: '© Mapbox © OpenStreetMap',
            },
          },
          layers: [{ id: 'satellite', type: 'raster', source: 'satellite' }],
        },
        center: defaultCenter,
        zoom: defaultZoom,
        pitch: defaultPitch,
        bearing: 0,
        scrollZoom: true,
      })

      const overlay = new MapboxOverlay({ interleaved: true, layers: [] })
      map.addControl(overlay as unknown as maplibregl.IControl)
      mapRef.current = map
      overlayRef.current = overlay

      map.on('zoom', () => {
        const above = map.getZoom() > RASTER_MAX_ZOOM
        if (above !== aboveThresholdRef.current) {
          aboveThresholdRef.current = above
          applyLayersRef.current()
        }
      })

      return () => {
        overlay.finalize()
        map.remove()
        mapRef.current = null
        overlayRef.current = null
      }
    }, [])

    useEffect(() => {
      if (!overlayRef.current) return

      applyLayersRef.current = () => {
        if (!overlayRef.current) return
        const above = aboveThresholdRef.current
        const layer = above
          ? (pmtilesUrl ? buildVectorLayer(pmtilesUrl) : null)
          : (tilesUrl    ? buildRasterLayer(tilesUrl)  : null)
        overlayRef.current.setProps({ layers: layer ? [layer] : [] })
      }

      applyLayersRef.current()
    }, [tilesUrl, pmtilesUrl])

    return <div ref={containerRef} data-testid="map-viewer" style={{ width: '100%', height: '100%' }} />
  },
)

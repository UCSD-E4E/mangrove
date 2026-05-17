import { useParams, useNavigate } from 'react-router-dom'
import { useRef, useState, useEffect } from 'react'
import { MapViewer } from '../components/MapViewer'
import type { MapViewerHandle } from '../components/MapViewer'
import { TerrainLegend } from '../components/TerrainLegend'
import { TerrainControls } from '../components/TerrainControls'
import { RegionStrip } from '../components/RegionStrip'
import { regions } from '../data/regions'

export function RegionPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const mapRef = useRef<MapViewerHandle>(null)
  const region = regions.find((r) => r.id === id) ?? regions[0]

  // Entry fade — resets each time the region changes
  const [entered, setEntered] = useState(false)
  useEffect(() => {
    setEntered(false)
    const raf = requestAnimationFrame(() => setEntered(true))
    return () => cancelAnimationFrame(raf)
  }, [id])

  // Fly to the region every time id changes (including first mount)
  useEffect(() => {
    mapRef.current?.flyToRegion(region.id)
  }, [region.id])

  return (
    <div
      style={{
        background: '#0a0f1a',
        opacity: entered ? 1 : 0,
        transform: entered ? 'none' : 'scale(0.985)',
        transition: 'opacity 0.55s ease, transform 0.55s ease',
        transformOrigin: 'center top',
      }}
    >
      <div style={{ position: 'relative', width: '100%', height: '100vh', overflow: 'hidden' }}>
        {/* Map starts at world zoom — flyToRegion zooms in to the region */}
        <MapViewer
          ref={mapRef}
          tilesUrl={region.tilesUrl}
          pmtilesUrl={region.pmtilesUrl}
          defaultZoom={2}
          defaultPitch={0}
        />

        {/* Top gradient for legibility */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: '160px',
          background: 'linear-gradient(rgba(10,15,26,0.75) 0%, transparent 100%)',
          pointerEvents: 'none',
        }} />

        {/* Top bar: back button + region name */}
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0,
          display: 'flex', alignItems: 'center', gap: '16px',
          padding: '20px 24px',
          zIndex: 20,
        }}>
          <button
            onClick={() => navigate('/')}
            style={{
              background: 'rgba(255,255,255,0.1)',
              border: '1px solid rgba(255,255,255,0.2)',
              borderRadius: '8px',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.8rem',
              padding: '7px 14px',
              backdropFilter: 'blur(8px)',
              WebkitBackdropFilter: 'blur(8px)',
              whiteSpace: 'nowrap',
              flexShrink: 0,
            }}
          >
            ← Back
          </button>

          <div>
            <p style={{
              margin: 0,
              fontSize: '0.65rem',
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              color: 'rgba(255,255,255,0.45)',
              fontWeight: 600,
            }}>
              Mangrove Region
            </p>
            <h2 style={{ margin: 0, color: '#fff', fontSize: '1.15rem', fontWeight: 600 }}>
              {region.displayName}
            </h2>
          </div>
        </div>

        <TerrainLegend />
        <TerrainControls
          onZoomIn={() => mapRef.current?.zoomIn()}
          onZoomOut={() => mapRef.current?.zoomOut()}
          onReset={() => mapRef.current?.flyToRegion(region.id)}
        />

        <RegionStrip
          activeRegionId={region.id}
          onSelectRegion={(newId) => navigate(`/region/${newId}`)}
        />
      </div>
    </div>
  )
}

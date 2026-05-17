import { useNavigate } from 'react-router-dom'
import { useRef } from 'react'
import { GlobeViewer } from '../components/GlobeViewer'
import { Nav } from '../components/Nav'
import { regions } from '../data/regions'

export function VisualizerPage() {
  const navigate = useNavigate()
  const globeRef = useRef<{ flyToRegion: (id: string) => void; resetView: () => void } | null>(null)

  return (
    <div style={{ height: '100vh', background: '#0a1a0c', display: 'flex', flexDirection: 'column' }} className="page-enter">
      <Nav />

      <div style={{ flex: 1, position: 'relative', paddingTop: '56px' }}>
        {/* Globe */}
        <GlobeViewer
          ref={globeRef}
          onRegionClick={(id) => navigate(`/region/${id}`)}
        />

        {/* Header overlay */}
        <div
          style={{
            position: 'absolute',
            top: '80px',
            left: '48px',
            zIndex: 10,
            pointerEvents: 'none',
          }}
        >
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(200,220,200,0.5)',
              marginBottom: '8px',
            }}
          >
            Global visualizer
          </p>
          <h1
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontWeight: 400,
              fontSize: 'clamp(1.6rem, 3vw, 2.4rem)',
              color: 'rgba(240,234,216,0.9)',
              lineHeight: 1.1,
            }}
          >
            Select a region
          </h1>
        </div>

        {/* Region list */}
        <div
          style={{
            position: 'absolute',
            bottom: '40px',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            gap: '8px',
            flexWrap: 'wrap',
            justifyContent: 'center',
            zIndex: 10,
            padding: '0 24px',
          }}
        >
          {regions.map(r => (
            <button
              key={r.id}
              onClick={() => navigate(`/region/${r.id}`)}
              style={{
                background: 'rgba(6,14,7,0.75)',
                backdropFilter: 'blur(12px)',
                border: `1px solid ${r.status === 'trained' ? 'rgba(61,107,74,0.5)' : 'rgba(255,255,255,0.1)'}`,
                color: r.status === 'trained' ? 'rgba(200,220,200,0.85)' : 'rgba(200,220,200,0.35)',
                padding: '8px 18px',
                borderRadius: '999px',
                cursor: r.status === 'trained' ? 'pointer' : 'default',
                fontFamily: "'DM Sans', sans-serif",
                fontSize: '0.78rem',
                letterSpacing: '0.03em',
                display: 'flex',
                alignItems: 'center',
                gap: '6px',
                transition: 'background 0.2s ease, border-color 0.2s ease',
              }}
            >
              <span
                style={{
                  width: '6px',
                  height: '6px',
                  borderRadius: '50%',
                  background: r.status === 'trained' ? '#3d6b4a' : 'rgba(200,200,200,0.3)',
                  flexShrink: 0,
                }}
              />
              {r.name}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

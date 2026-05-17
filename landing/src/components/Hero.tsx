import { useRef } from 'react'
import { GlobeViewer } from './GlobeViewer'
import type { GlobeViewerHandle } from './GlobeViewer'
import { regions } from '../data/regions'

export type { GlobeViewerHandle }

type Props = {
  onRegionClick?: (id: string) => void
}

export function Hero({ onRegionClick }: Props) {
  const globeRef = useRef<GlobeViewerHandle>(null)

  return (
    <section
      data-testid="hero"
      aria-label="Hero"
      style={{
        height: '100vh',
        backgroundColor: '#0a1a0c',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Globe — fills section */}
      <div id="globe-root" style={{ position: 'absolute', inset: '0' }}>
        <GlobeViewer ref={globeRef} onRegionClick={onRegionClick} />
      </div>

      {/* Text overlay — top left */}
      <div
        data-testid="hero-overlay"
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

      {/* Region pills — bottom center */}
      <div
        data-testid="region-pills"
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
            onClick={() => onRegionClick?.(r.id)}
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
    </section>
  )
}

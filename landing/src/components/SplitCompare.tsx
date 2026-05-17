import { useRef, useState } from 'react'

const METADATA = [
  { label: 'Source', value: 'Sentinel-2 + Drone Survey' },
  { label: 'Model', value: 'SegFormer + SR-UNet' },
  { label: 'Output', value: '0.35 m GSD' },
]

interface SplitCompareProps {
  leftSrc?: string
  rightSrc?: string
}

export function SplitCompare({
  leftSrc = import.meta.env.VITE_SPLIT_LEFT_URL,
  rightSrc = import.meta.env.VITE_SPLIT_RIGHT_URL,
}: SplitCompareProps = {}) {
  const [divider, setDivider] = useState(42)
  const [dragging, setDragging] = useState(false)
  const viewerRef = useRef<HTMLDivElement>(null)

  const clamp = (pct: number) => Math.min(95, Math.max(5, pct))

  const startDrag = () => {
    setDragging(true)

    const handleDocMove = (ev: PointerEvent) => {
      const rect = viewerRef.current?.getBoundingClientRect()
      if (!rect || rect.width === 0) return
      setDivider(clamp(((ev.clientX - rect.left) / rect.width) * 100))
    }

    const handleDocUp = () => {
      setDragging(false)
      document.removeEventListener('pointermove', handleDocMove)
      document.removeEventListener('pointerup', handleDocUp)
    }

    document.addEventListener('pointermove', handleDocMove)
    document.addEventListener('pointerup', handleDocUp)
  }

  return (
    <section
      data-testid="split-compare"
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        padding: '80px 24px',
        backgroundColor: 'var(--bg-raised)',
      }}
    >
      <div style={{ width: '100%', maxWidth: '920px' }}>
        <p className="overline" style={{ marginBottom: '1rem' }}>Resolution uplift</p>

        <h2 style={{ marginBottom: '1rem' }}>
          From 10 meters to{' '}
          <em className="serif-italic">0.35 meters</em>
        </h2>

        <p
          style={{
            color: 'var(--text-secondary)',
            lineHeight: 1.8,
            maxWidth: '560px',
            marginBottom: '2rem',
          }}
        >
          Our super-resolution pipeline upsamples Sentinel-2 land cover predictions by 28×,
          producing sub-meter canopy maps suitable for individual tree monitoring and
          accurate biomass estimation.
        </p>

        {/* Viewer */}
        <div
          data-testid="viewer"
          ref={viewerRef}
          style={{
            position: 'relative',
            height: '500px',
            borderRadius: '6px',
            overflow: 'hidden',
            cursor: dragging ? 'ew-resize' : 'default',
            userSelect: 'none',
            marginBottom: '1.5rem',
          }}
        >
          {/* Right image — 0.35m SR output (full width, behind left) */}
          {rightSrc ? (
            <img
              data-testid="image-right"
              src={rightSrc}
              alt="0.35m SR output"
              loading="lazy"
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover' }}
            />
          ) : (
            <div
              data-testid="image-right"
              style={{
                position: 'absolute',
                inset: 0,
                background: 'linear-gradient(135deg, #3d6b4a 0%, #7a9a6a 50%, #4a90b8 100%)',
              }}
            />
          )}

          {/* Left image — 10m coarse input (clipped at divider) */}
          {leftSrc ? (
            <img
              data-testid="image-left"
              src={leftSrc}
              alt="10m input"
              loading="lazy"
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: 'cover',
                clipPath: `inset(0 ${100 - divider}% 0 0)`,
              }}
            />
          ) : (
            <div
              data-testid="image-left"
              style={{
                position: 'absolute',
                inset: 0,
                background: 'linear-gradient(135deg, #c8c0b0 0%, #aaa 50%, #8a8070 100%)',
                clipPath: `inset(0 ${100 - divider}% 0 0)`,
              }}
            />
          )}

          {/* Divider line + handle */}
          <div
            data-testid="divider"
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${divider}%`,
              width: '1px',
              background: 'rgba(255,255,255,0.9)',
              transform: 'translateX(-50%)',
              zIndex: 5,
            }}
          >
            <div
              data-testid="divider-handle"
              onPointerDown={startDrag}
              style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                background: 'white',
                boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
                cursor: 'ew-resize',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
              }}
            >
              ⟺
            </div>
          </div>

          {/* Labels */}
          <div
            style={{
              position: 'absolute',
              bottom: '12px',
              left: '12px',
              background: 'rgba(0,0,0,0.55)',
              color: '#fff',
              borderRadius: '999px',
              padding: '3px 10px',
              fontSize: '0.7rem',
              fontFamily: "'DM Mono', monospace",
              zIndex: 6,
            }}
          >
            10m input
          </div>
          <div
            style={{
              position: 'absolute',
              bottom: '12px',
              right: '12px',
              background: 'rgba(61,107,74,0.8)',
              color: '#fff',
              borderRadius: '999px',
              padding: '3px 10px',
              fontSize: '0.7rem',
              fontFamily: "'DM Mono', monospace",
              zIndex: 6,
            }}
          >
            0.35m SR output
          </div>
        </div>

        {/* Metadata row */}
        <div
          data-testid="metadata-row"
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: '1px',
            background: 'var(--border)',
            border: '1px solid var(--border)',
            borderRadius: '6px',
            overflow: 'hidden',
          }}
        >
          {METADATA.map((item) => (
            <div
              key={item.label}
              data-testid="metadata-cell"
              style={{
                background: 'var(--bg-panel)',
                padding: '14px 20px',
              }}
            >
              <p className="caption" style={{ marginBottom: '2px' }}>{item.label}</p>
              <p style={{ fontFamily: "'DM Sans', sans-serif", fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                {item.value}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

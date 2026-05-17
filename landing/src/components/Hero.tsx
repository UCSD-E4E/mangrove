import { useRef, useEffect } from 'react'
import { GlobeViewer } from './GlobeViewer'
import type { GlobeViewerHandle } from './GlobeViewer'

export type { GlobeViewerHandle }

type Props = {
  onRegionClick?: (id: string) => void
}

export function Hero({ onRegionClick }: Props) {
  const sectionRef = useRef<HTMLElement>(null)
  const globeRef = useRef<GlobeViewerHandle>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const gradientRef = useRef<HTMLDivElement>(null)
  const textRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let ticking = false
    const update = () => {
      const progress = Math.min(window.scrollY / (window.innerHeight * 0.65), 1)
      const opacity = String(1 - progress)
      if (overlayRef.current) overlayRef.current.style.opacity = opacity
      if (gradientRef.current) gradientRef.current.style.opacity = opacity
      if (textRef.current) textRef.current.style.opacity = opacity
      ticking = false
    }
    const onScroll = () => {
      if (!ticking) { requestAnimationFrame(update); ticking = true }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    // Wrapper provides the extra scroll space while the hero is pinned via sticky
    <div style={{ height: '230vh' }}>
      <section
        ref={sectionRef}
        data-testid="hero"
        aria-label="Hero"
        style={{
          position: 'sticky',
          top: 0,
          height: '100vh',
          overflow: 'hidden',
          backgroundColor: '#f0ede6',
        }}
      >
        {/* Globe — fills section */}
        <div
          id="globe-root"
          style={{ position: 'absolute', inset: '0' }}
        >
          <GlobeViewer ref={globeRef} onRegionClick={onRegionClick} />
        </div>

        {/* Frosted blur overlay — obscures globe so text is legible; fades out on scroll */}
        <div
          ref={overlayRef}
          style={{
            position: 'absolute',
            inset: '0',
            backdropFilter: 'blur(14px)',
            WebkitBackdropFilter: 'blur(14px)',
            background: 'rgba(240, 237, 230, 0.45)',
            pointerEvents: 'none',
          }}
        />

        {/* Bottom gradient — fades globe into page background */}
        <div
          ref={gradientRef}
          data-testid="gradient-overlay"
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            height: '65%',
            backgroundImage:
              'linear-gradient(transparent 0%, rgba(240,237,230,0.5) 40%, rgba(240,237,230,0.97) 100%)',
            pointerEvents: 'none',
          }}
        />

        {/* Text block — bottom-left over gradient; fades on scroll */}
        <div
          ref={textRef}
          style={{
            position: 'absolute',
            bottom: '4rem',
            left: '4rem',
            maxWidth: '600px',
            zIndex: 1,
          }}
        >
          <p className="overline" style={{ marginBottom: '1.25rem' }}>
            Engineers for Exploration · UC San Diego
          </p>

          <h1 style={{ marginBottom: '1.25rem' }}>
            A living map of Earth's{' '}
            <em className="serif-italic">mangroves</em>
          </h1>

          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem', lineHeight: 1.7, maxWidth: '480px' }}>
            Drag the globe. Click any region. Explore a 3D terrain view of our
            land cover predictions at sub-meter resolution.
          </p>
        </div>

        {/* Scroll cue — bottom-right */}
        <div
          data-testid="scroll-cue"
          style={{
            position: 'absolute',
            bottom: '4rem',
            right: '4rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: '0.75rem',
            zIndex: 1,
          }}
        >
          <span className="overline">scroll</span>
          <div
            style={{
              width: '1px',
              height: '48px',
              background: 'linear-gradient(var(--accent), transparent)',
            }}
          />
        </div>
      </section>
    </div>
  )
}

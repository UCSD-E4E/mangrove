import { useRef, useState, useEffect } from 'react'

const PILLARS = [
  {
    num: '01',
    title: 'Satellite Imagery',
    body:
      'We ingest multispectral satellite imagery with 10m resolution across every mangrove coastline on Earth.',
    detail: 'via Sentinel-2',
  },
  {
    num: '02',
    title: 'ML Segmentation',
    body:
      'An in-house model trained on composite imagery through a custom pipeline classifies mangroves, water, built-up land, and vegetation pixel by pixel.',
    detail: '10 classes · 94% accuracy',
  },
  {
    num: '03',
    title: 'Super Resolution',
    body:
      'We up-sample predictions to sub-meter detail, resolving individual canopy gaps and tidal channels invisible at native satellite resolution.',
    detail: '16× upscaling · hybrid transformer CNN architecture',
  },
]

export function SolutionSection() {
  const ref = useRef<HTMLElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true) },
      { threshold: 0.15 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <section
      ref={ref}
      style={{
        background: 'linear-gradient(180deg, #060e07 0%, #0a1a0c 50%, #060e07 100%)',
        padding: 'clamp(80px, 12vh, 140px) clamp(24px, 8vw, 120px)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Faint grid overlay */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'linear-gradient(rgba(61,107,74,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(61,107,74,0.04) 1px, transparent 1px)',
          backgroundSize: '80px 80px',
          pointerEvents: 'none',
        }}
      />

      <div
        style={{
          maxWidth: '1100px',
          margin: '0 auto',
          position: 'relative',
          zIndex: 1,
        }}
      >
        {/* Header */}
        <div
          style={{
            marginBottom: 'clamp(48px, 8vh, 96px)',
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(20px)',
            transition: 'opacity 0.7s ease, transform 0.7s ease',
          }}
        >
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(61,107,74,0.8)',
              marginBottom: '1.5rem',
            }}
          >
            Our solution
          </p>
          <h2
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontWeight: 400,
              fontSize: 'clamp(2rem, 5vw, 4.5rem)',
              color: '#e8e0d0',
              lineHeight: 1.1,
              maxWidth: '740px',
            }}
          >
            Machine learning to monitor every
            mangrove forest{' '}
            <em style={{ fontStyle: 'italic', color: 'rgba(100,180,120,0.85)' }}>
              on Earth.
            </em>
          </h2>
          <p
            style={{
              marginTop: '1.5rem',
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 'clamp(0.9rem, 1.5vw, 1.05rem)',
              color: 'rgba(232,224,208,0.55)',
              maxWidth: '580px',
              lineHeight: 1.8,
            }}
          >
            The UCSD Engineers for Exploration team built an end-to-end pipeline
            that turns raw satellite data into high-resolution, classified maps
            that're updated continuously, globally.
          </p>
        </div>

        {/* Pillars */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1px',
            background: 'rgba(255,255,255,0.04)',
          }}
        >
          {PILLARS.map((p, i) => (
            <div
              key={i}
              style={{
                padding: 'clamp(32px, 5vh, 56px) clamp(24px, 4vw, 40px)',
                background: '#060e07',
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateY(0)' : 'translateY(28px)',
                transition: `opacity 0.7s ease ${0.1 + i * 0.15}s, transform 0.7s ease ${0.1 + i * 0.15}s`,
              }}
            >
              <div
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: '0.68rem',
                  color: 'rgba(61,107,74,0.6)',
                  letterSpacing: '0.12em',
                  marginBottom: '20px',
                }}
              >
                {p.num}
              </div>
              <h3
                style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontSize: 'clamp(1.3rem, 2.5vw, 1.8rem)',
                  fontWeight: 400,
                  color: '#e8e0d0',
                  marginBottom: '16px',
                  lineHeight: 1.15,
                }}
              >
                {p.title}
              </h3>
              <p
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  fontSize: '0.9rem',
                  color: 'rgba(232,224,208,0.55)',
                  lineHeight: 1.8,
                  marginBottom: '20px',
                }}
              >
                {p.body}
              </p>
              <span
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: '0.7rem',
                  color: 'rgba(61,107,74,0.65)',
                  letterSpacing: '0.06em',
                }}
              >
                {p.detail}
              </span>
            </div>
          ))}
        </div>

      </div>
    </section>
  )
}

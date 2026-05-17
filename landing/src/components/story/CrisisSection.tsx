import { useRef, useState, useEffect } from 'react'

const STATS = [
  {
    value: '35%',
    label: 'of the world\'s mangroves',
    sublabel: 'lost since 1980',
    color: '#c44a2a',
  },
  {
    value: '340M',
    label: 'people depend on mangroves',
    sublabel: 'for coastal storm protection',
    color: '#c89a2a',
  },
  {
    value: '1%',
    label: 'of remaining forests',
    sublabel: 'disappear every single year',
    color: '#c44a2a',
  },
]

export function CrisisSection() {
  const ref = useRef<HTMLElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true) },
      { threshold: 0.2 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return (
    <section
      ref={ref}
      style={{
        background: '#0a0805',
        padding: 'clamp(80px, 12vh, 140px) clamp(24px, 8vw, 120px)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Faint background text */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          fontFamily: "'Instrument Serif', serif",
          fontSize: 'clamp(10rem, 30vw, 28rem)',
          fontWeight: 400,
          color: 'rgba(196,74,42,0.04)',
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
          userSelect: 'none',
          lineHeight: 1,
        }}
      >
        crisis
      </div>

      <div style={{ maxWidth: '1100px', margin: '0 auto', position: 'relative', zIndex: 1 }}>
        {/* Header */}
        <div style={{ marginBottom: 'clamp(48px, 8vh, 96px)' }}>
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(196,74,42,0.7)',
              marginBottom: '1.5rem',
            }}
          >
            The situation
          </p>
          <h2
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontWeight: 400,
              fontSize: 'clamp(2rem, 5vw, 4.5rem)',
              color: '#e8e0d0',
              lineHeight: 1.1,
              maxWidth: '700px',
            }}
          >
            We are losing one of Earth's
            most vital ecosystems,{' '}
            <em style={{ fontStyle: 'italic', color: 'rgba(196,74,42,0.9)' }}>
              faster than we can study it.
            </em>
          </h2>
        </div>

        {/* Stats grid */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '2px',
          }}
        >
          {STATS.map((stat, i) => (
            <div
              key={i}
              style={{
                padding: 'clamp(32px, 5vh, 56px) clamp(24px, 4vw, 48px)',
                background: 'rgba(255,255,255,0.02)',
                borderLeft: `3px solid ${stat.color}`,
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateY(0)' : 'translateY(24px)',
                transition: `opacity 0.7s ease ${i * 0.15}s, transform 0.7s ease ${i * 0.15}s`,
              }}
            >
              <div
                style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontSize: 'clamp(4rem, 9vw, 8rem)',
                  color: stat.color,
                  lineHeight: 1,
                  fontWeight: 400,
                  marginBottom: '16px',
                }}
              >
                {stat.value}
              </div>
              <p
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  fontSize: 'clamp(0.9rem, 1.5vw, 1.05rem)',
                  color: '#e8e0d0',
                  fontWeight: 500,
                  lineHeight: 1.4,
                  marginBottom: '6px',
                }}
              >
                {stat.label}
              </p>
              <p
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: '0.75rem',
                  color: 'rgba(232,224,208,0.4)',
                  letterSpacing: '0.04em',
                }}
              >
                {stat.sublabel}
              </p>
            </div>
          ))}
        </div>

        {/* Pull quote */}
        <div
          style={{
            marginTop: 'clamp(48px, 8vh, 80px)',
            paddingTop: 'clamp(32px, 5vh, 48px)',
            borderTop: '1px solid rgba(255,255,255,0.06)',
            display: 'flex',
            flexDirection: 'column',
            gap: '12px',
          }}
        >
          <p
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontStyle: 'italic',
              fontSize: 'clamp(1rem, 2vw, 1.35rem)',
              color: 'rgba(232,224,208,0.65)',
              lineHeight: 1.6,
              maxWidth: '680px',
            }}
          >
            "Mangroves are among the most carbon-dense forests on the planet, yet they
            receive a fraction of the conservation attention they deserve."
          </p>
          <span
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.72rem',
              color: 'rgba(232,224,208,0.3)',
              letterSpacing: '0.06em',
            }}
          >
            — IPCC Sixth Assessment Report, 2022
          </span>
        </div>
      </div>
    </section>
  )
}

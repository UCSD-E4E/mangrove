import { useRef, useState, useEffect } from 'react'

export function SuperResSection() {
  const ref = useRef<HTMLElement>(null)
  const [visible, setVisible] = useState(false)
  const [hover, setHover] = useState(false)

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
        background: '#060e07',
        padding: 'clamp(80px, 12vh, 140px) clamp(24px, 8vw, 120px)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <div style={{ maxWidth: '1100px', margin: '0 auto', position: 'relative', zIndex: 1 }}>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
            gap: 'clamp(40px, 8vw, 100px)',
            alignItems: 'center',
          }}
        >
          {/* Left: text */}
          <div
            style={{
              opacity: visible ? 1 : 0,
              transform: visible ? 'translateX(0)' : 'translateX(-24px)',
              transition: 'opacity 0.8s ease, transform 0.8s ease',
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
              Super resolution
            </p>
            <h2
              style={{
                fontFamily: "'Instrument Serif', serif",
                fontWeight: 400,
                fontSize: 'clamp(1.8rem, 4vw, 3.5rem)',
                color: '#e8e0d0',
                lineHeight: 1.1,
                marginBottom: '1.5rem',
              }}
            >
              Seeing what satellites
              <br />
              <em style={{ fontStyle: 'italic', color: 'rgba(100,180,120,0.85)' }}>
                can't.
              </em>
            </h2>
            <p
              style={{
                fontFamily: "'DM Sans', sans-serif",
                fontSize: 'clamp(0.88rem, 1.4vw, 1rem)',
                color: 'rgba(232,224,208,0.55)',
                lineHeight: 1.85,
                marginBottom: '1.25rem',
              }}
            >
              Standard Sentinel-2 imagery is 10 meters per pixel, which is enough
              to map forests, but not to resolve the tidal channels, root systems,
              and fragmented edges that determine ecosystem health.
            </p>
            <p
              style={{
                fontFamily: "'DM Sans', sans-serif",
                fontSize: 'clamp(0.88rem, 1.4vw, 1rem)',
                color: 'rgba(232,224,208,0.55)',
                lineHeight: 1.85,
              }}
            >
              Our super-resolution model upsamples predictions 16×, sharpening
              boundaries, revealing sub-canopy structure, and enabling measurement
              of individual mangrove stands that were previously invisible.
            </p>

            <div
              style={{
                marginTop: '2rem',
                display: 'flex',
                gap: '40px',
              }}
            >
              {[
                { val: '10m', label: 'Native resolution' },
                { val: '0.6m', label: 'After super-res' },
                { val: '16×', label: 'Upscaling factor' },
              ].map(({ val, label }) => (
                <div key={label}>
                  <div
                    style={{
                      fontFamily: "'Instrument Serif', serif",
                      fontSize: 'clamp(1.4rem, 3vw, 2rem)',
                      color: 'rgba(100,180,120,0.85)',
                      fontWeight: 400,
                    }}
                  >
                    {val}
                  </div>
                  <div
                    style={{
                      fontFamily: "'DM Mono', monospace",
                      fontSize: '0.68rem',
                      color: 'rgba(232,224,208,0.35)',
                      letterSpacing: '0.06em',
                      marginTop: '4px',
                    }}
                  >
                    {label}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: visual comparison */}
          <div
            style={{
              opacity: visible ? 1 : 0,
              transform: visible ? 'translateX(0)' : 'translateX(24px)',
              transition: 'opacity 0.8s ease 0.2s, transform 0.8s ease 0.2s',
            }}
          >
            <div
              style={{
                position: 'relative',
                borderRadius: '6px',
                overflow: 'hidden',
                border: '1px solid rgba(61,107,74,0.2)',
                cursor: 'crosshair',
              }}
              onMouseEnter={() => setHover(true)}
              onMouseLeave={() => setHover(false)}
            >
              {/* Simulated low-res tile */}
              <div
                style={{
                  width: '100%',
                  aspectRatio: '1',
                  background: `
                    repeating-conic-gradient(
                      rgba(13,46,20,0.9) 0% 25%, rgba(9,26,11,0.9) 25% 50%
                    ) 0 0 / 40px 40px
                  `,
                  filter: hover ? 'none' : 'blur(3px)',
                  transition: 'filter 0.4s ease',
                  position: 'relative',
                }}
              >
                {/* Simulated classified patches */}
                {[
                  { x: '10%', y: '15%', w: '30%', h: '25%', c: 'rgba(61,107,74,0.85)' },
                  { x: '50%', y: '10%', w: '40%', h: '20%', c: 'rgba(61,107,74,0.7)' },
                  { x: '5%', y: '55%', w: '45%', h: '30%', c: 'rgba(61,107,74,0.9)' },
                  { x: '55%', y: '50%', w: '35%', h: '35%', c: 'rgba(74,144,184,0.75)' },
                  { x: '60%', y: '35%', w: '15%', h: '12%', c: 'rgba(138,106,74,0.8)' },
                ].map((patch, i) => (
                  <div
                    key={i}
                    style={{
                      position: 'absolute',
                      left: patch.x,
                      top: patch.y,
                      width: patch.w,
                      height: patch.h,
                      background: patch.c,
                      borderRadius: hover ? '2px' : '0',
                      transition: 'border-radius 0.4s ease',
                    }}
                  />
                ))}
              </div>

              {/* Label overlay */}
              <div
                style={{
                  position: 'absolute',
                  inset: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'space-between',
                  padding: '16px',
                  pointerEvents: 'none',
                }}
              >
                <div
                  style={{
                    alignSelf: 'flex-start',
                    background: 'rgba(4,11,5,0.75)',
                    backdropFilter: 'blur(8px)',
                    padding: '4px 10px',
                    borderRadius: '3px',
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.68rem',
                    color: 'rgba(232,224,208,0.7)',
                    letterSpacing: '0.06em',
                  }}
                >
                  {hover ? 'Super-res · 0.6m/px' : 'Native · 10m/px'}
                </div>
                <div
                  style={{
                    alignSelf: 'flex-end',
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.65rem',
                    color: 'rgba(232,224,208,0.35)',
                  }}
                >
                  hover to compare
                </div>
              </div>
            </div>

            <p
              style={{
                marginTop: '12px',
                fontFamily: "'DM Mono', monospace",
                fontSize: '0.68rem',
                color: 'rgba(232,224,208,0.3)',
                letterSpacing: '0.06em',
                textAlign: 'center',
              }}
            >
              Simulated classification · Florida mangrove coast
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

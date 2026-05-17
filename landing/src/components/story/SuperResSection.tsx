import { useRef, useState, useEffect } from 'react'

const BASE = import.meta.env.VITE_TILES_BASE_URL ?? ''

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
                flexWrap: 'wrap',
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

          {/* Right: real image comparison */}
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
                aspectRatio: '1',
              }}
              onMouseEnter={() => setHover(true)}
              onMouseLeave={() => setHover(false)}
            >
              {/* Sentinel-2 10m — base layer */}
              <img
                src={`${BASE}/sentinel2_10m.png`}
                alt="Sentinel-2 10m resolution"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block',
                  imageRendering: 'pixelated',
                }}
              />

              {/* High-res 0.6m — fades in on hover */}
              <img
                src={`${BASE}/naip_0.6m.png`}
                alt="Super-resolution 0.6m"
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  display: 'block',
                  opacity: hover ? 1 : 0,
                  transition: 'opacity 0.55s ease',
                }}
              />

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
                    background: 'rgba(4,11,5,0.72)',
                    backdropFilter: 'blur(8px)',
                    padding: '4px 10px',
                    borderRadius: '3px',
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.68rem',
                    color: 'rgba(232,224,208,0.8)',
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
              Ten Thousand Islands · Florida mangrove coast
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}

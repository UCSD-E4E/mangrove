import { useNavigate } from 'react-router-dom'

export function JoinCTA() {
  const navigate = useNavigate()
  return (
    <section
      data-testid="join-cta"
      style={{
        background: '#f7f4ee',
        padding: 'clamp(80px, 14vh, 160px) clamp(24px, 8vw, 120px)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Decorative rule — top */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: 0,
          left: 'clamp(24px, 8vw, 120px)',
          right: 'clamp(24px, 8vw, 120px)',
          height: '1px',
          background: 'rgba(61,107,74,0.15)',
        }}
      />

      <div
        style={{
          maxWidth: '1100px',
          margin: '0 auto',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: 'clamp(48px, 8vw, 100px)',
          alignItems: 'center',
        }}
      >
        {/* Left — headline */}
        <div>
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
              color: 'rgba(61,107,74,0.7)',
              marginBottom: '1.5rem',
            }}
          >
            Open science
          </p>

          <h2
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontWeight: 400,
              fontSize: 'clamp(2.4rem, 5vw, 4.2rem)',
              color: '#1a1a18',
              lineHeight: 1.06,
              marginBottom: '1.5rem',
            }}
          >
            Have data,
            <br />
            expertise,
            <br />
            or{' '}
            <em style={{ fontStyle: 'italic', color: '#3d6b4a' }}>ideas?</em>
          </h2>

          <p
            style={{
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 'clamp(0.9rem, 1.4vw, 1.05rem)',
              color: '#666660',
              lineHeight: 1.85,
              maxWidth: '380px',
            }}
          >
            We partner with conservation organizations, remote sensing labs, satellite
            data providers, and NGOs worldwide.
          </p>
        </div>

        {/* Right — contact block */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}>
          {/* Email CTA */}
          <div
            style={{
              border: '1px solid rgba(61,107,74,0.2)',
              borderRadius: '8px',
              padding: 'clamp(28px, 4vh, 40px) clamp(24px, 3vw, 36px)',
              background: '#fff',
            }}
          >
            <p
              style={{
                fontFamily: "'DM Mono', monospace",
                fontSize: '0.65rem',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
                color: 'rgba(61,107,74,0.6)',
                marginBottom: '12px',
              }}
            >
              Get in touch
            </p>
            <p
              style={{
                fontFamily: "'Instrument Serif', serif",
                fontSize: 'clamp(1.1rem, 2vw, 1.4rem)',
                color: '#1a1a18',
                lineHeight: 1.3,
                marginBottom: '20px',
              }}
            >
              We respond within a few business days.
            </p>
            <button
              data-testid="cta-button"
              onClick={() => navigate('/collaborate')}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '12px 28px',
                background: '#3d6b4a',
                color: '#fff',
                borderRadius: '4px',
                border: 'none',
                cursor: 'pointer',
                fontFamily: "'DM Sans', sans-serif",
                fontSize: '0.88rem',
                fontWeight: 500,
                letterSpacing: '0.03em',
                transition: 'opacity 0.2s ease',
              }}
              onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.opacity = '0.85')}
              onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.opacity = '1')}
            >
              Get in touch →
            </button>
          </div>

          {/* E4E link */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '20px 28px',
              border: '1px solid rgba(0,0,0,0.06)',
              borderRadius: '8px',
              background: '#f2efe8',
            }}
          >
            <div>
              <p
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: '0.64rem',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: '#aaa',
                  marginBottom: '4px',
                }}
              >
                About the lab
              </p>
              <p
                style={{
                  fontFamily: "'DM Sans', sans-serif",
                  fontSize: '0.88rem',
                  color: '#1a1a18',
                  fontWeight: 500,
                }}
              >
                Engineers for Exploration
              </p>
            </div>
            <a
              href="https://e4e.ucsd.edu"
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontFamily: "'DM Mono', monospace",
                fontSize: '0.72rem',
                color: '#3d6b4a',
                textDecoration: 'none',
                letterSpacing: '0.04em',
              }}
            >
              e4e.ucsd.edu ↗
            </a>
          </div>
        </div>
      </div>
    </section>
  )
}

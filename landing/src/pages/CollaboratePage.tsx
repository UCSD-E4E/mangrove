import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'

const WAYS = [
  {
    title: 'Share data',
    body: 'Have UAV imagery, field surveys, or satellite archives? We can integrate new data sources into our training pipeline.',
  },
  {
    title: 'Join as a researcher',
    body: 'We welcome remote sensing scientists, ML engineers, and conservation biologists. UC San Diego affiliation is not required.',
  },
  {
    title: 'Partner on a region',
    body: 'Working in a mangrove area not yet covered? We can prioritize new regions with local partner support.',
  },
]

export function CollaboratePage() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-page)' }} className="page-enter">
      <Nav />

      <main style={{ paddingTop: '56px' }}>
        {/* Hero */}
        <div
          style={{
            background: '#060e07',
            padding: 'clamp(80px, 14vh, 140px) clamp(24px, 8vw, 120px)',
            position: 'relative',
            overflow: 'hidden',
          }}
        >
          <div className="mangrove-rays" />
          <div style={{ maxWidth: '720px', position: 'relative', zIndex: 1 }}>
            <p
              style={{
                fontFamily: "'DM Mono', monospace",
                fontSize: '0.68rem',
                letterSpacing: '0.18em',
                textTransform: 'uppercase',
                color: 'rgba(61,107,74,0.75)',
                marginBottom: '1.5rem',
              }}
            >
              Open science
            </p>
            <h1
              style={{
                fontFamily: "'Instrument Serif', serif",
                fontWeight: 400,
                fontSize: 'clamp(2.2rem, 5.5vw, 4.5rem)',
                color: '#e8e0d0',
                lineHeight: 1.08,
                marginBottom: '1.5rem',
              }}
            >
              Let's work{' '}
              <em style={{ fontStyle: 'italic', color: 'rgba(100,180,120,0.85)' }}>
                together.
              </em>
            </h1>
            <p
              style={{
                fontFamily: "'DM Sans', sans-serif",
                fontSize: 'clamp(0.9rem, 1.5vw, 1.05rem)',
                color: 'rgba(232,224,208,0.6)',
                lineHeight: 1.85,
                maxWidth: '560px',
              }}
            >
              We partner with conservation organizations, remote sensing labs, satellite
              data providers, and NGOs. If you have data, expertise, funding, or ideas —
              reach out.
            </p>
          </div>
        </div>

        <div
          style={{
            maxWidth: '900px',
            margin: '0 auto',
            padding: 'clamp(60px, 10vh, 100px) clamp(24px, 8vw, 120px)',
          }}
        >
          {/* Ways to collaborate */}
          <p className="overline" style={{ marginBottom: '2rem' }}>Ways to collaborate</p>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
              gap: '1px',
              background: 'var(--border)',
              marginBottom: 'clamp(60px, 10vh, 100px)',
            }}
          >
            {WAYS.map((w, i) => (
              <div
                key={i}
                style={{
                  background: 'var(--bg-page)',
                  padding: 'clamp(28px, 4vh, 40px) clamp(20px, 3vw, 32px)',
                }}
              >
                <h3
                  style={{
                    fontFamily: "'Instrument Serif', serif",
                    fontWeight: 400,
                    fontSize: 'clamp(1.1rem, 2vw, 1.4rem)',
                    color: 'var(--text-primary)',
                    marginBottom: '12px',
                  }}
                >
                  {w.title}
                </h3>
                <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  {w.body}
                </p>
              </div>
            ))}
          </div>

          {/* Contact */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: 'clamp(32px, 5vw, 60px)',
              alignItems: 'start',
            }}
          >
            <div>
              <p className="overline" style={{ marginBottom: '1rem' }}>Get in touch</p>
              <h2
                style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontWeight: 400,
                  fontSize: 'clamp(1.5rem, 3vw, 2.2rem)',
                  color: 'var(--text-primary)',
                  lineHeight: 1.15,
                  marginBottom: '1rem',
                }}
              >
                Email us directly
              </h2>
              <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.8, marginBottom: '2rem' }}>
                The quickest way to start a conversation. We respond within a few business days.
              </p>
              <a
                href="mailto:e4e@ucsd.edu"
                style={{
                  display: 'inline-block',
                  padding: '12px 32px',
                  background: 'var(--accent)',
                  color: '#fff',
                  borderRadius: '4px',
                  textDecoration: 'none',
                  fontFamily: "'DM Sans', sans-serif",
                  fontSize: '0.88rem',
                  fontWeight: 500,
                  letterSpacing: '0.03em',
                  transition: 'opacity 0.2s ease',
                }}
                onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.opacity = '0.85')}
                onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.opacity = '1')}
              >
                e4e@ucsd.edu →
              </a>
            </div>

            {/* E4E link */}
            <div
              style={{
                background: 'var(--bg-raised)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '28px 24px',
              }}
            >
              <p
                style={{
                  fontFamily: "'DM Mono', monospace",
                  fontSize: '0.68rem',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  color: 'var(--text-muted)',
                  marginBottom: '12px',
                }}
              >
                About the lab
              </p>
              <p
                style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontSize: '1.1rem',
                  color: 'var(--text-primary)',
                  lineHeight: 1.3,
                  marginBottom: '12px',
                }}
              >
                Engineers for Exploration
              </p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.75, marginBottom: '20px' }}>
                A student research group at UC San Diego building technology for
                scientific exploration, conservation, and cultural heritage preservation.
              </p>
              <a
                href="https://e4e.ucsd.edu"
                target="_blank"
                rel="noopener noreferrer"
                style={{
                  fontSize: '0.82rem',
                  color: 'var(--accent)',
                  textDecoration: 'none',
                  fontFamily: "'DM Sans', sans-serif",
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                }}
              >
                Visit e4e.ucsd.edu ↗
              </a>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

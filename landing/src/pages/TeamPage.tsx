import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'

const BASE = import.meta.env.VITE_TILES_BASE_URL ?? ''

const LEADS = [
  {
    id: 'cheung',
    name: 'Jason Cheung',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    bio: 'Leads the ML pipeline and model architecture. Developing SegFormer-based segmentation for multispectral satellite imagery and super-resolution upscaling.',
    photo: `${BASE}/team/jason-cheung.jpg`,
  },
  {
    id: 'swetlin',
    name: 'Nick Swetlin',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    bio: 'Leads satellite data ingestion and processing via Google Earth Engine. Handles multi-temporal compositing, cloud masking, and spectral normalization.',
    photo: `${BASE}/team/nick-swetlin.jpg`,
  },
  {
    id: 'zhou',
    name: 'Andrew Zhou',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    bio: 'Leads the tile generation pipeline, PMTiles infrastructure, and the Cloudflare R2 delivery stack powering the visualization platform.',
    photo: `${BASE}/team/andrew-zhou.svg`,
  },
]

const ENGINEERS = [
  { id: 'fernandez', name: 'Kate Fernandez', role: 'Engineer', photo: `${BASE}/team/kate-fernandez.svg` },
  { id: 'dsouza',    name: 'Dylan Dsouza',   role: 'Engineer', photo: `${BASE}/team/dylan-dsouza.svg`   },
  { id: 'dong',      name: 'Michelle Dong',  role: 'Engineer', photo: `${BASE}/team/michelle-dong.svg`  },
  { id: 'mei',       name: 'Jack Mei',       role: 'Engineer', photo: `${BASE}/team/jack-mei.svg`       },
  { id: 'jain',      name: 'Prisha Jain',    role: 'Engineer', photo: `${BASE}/team/prisha-jain.svg`    },
]

const ADVISORS = [
  { name: 'Prof. Ryan Kastner',   role: 'Faculty Advisor', affiliation: 'UC San Diego CSE' },
  { name: 'Prof. Curt Schurgers', role: 'Faculty Advisor', affiliation: 'UC San Diego ECE' },
]

export function TeamPage() {
  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-page)' }} className="page-enter">
      <Nav />

      <main style={{ paddingTop: '56px' }}>
        {/* Header */}
        <div
          style={{
            padding: 'clamp(60px, 10vh, 100px) clamp(24px, 8vw, 120px) clamp(40px, 6vh, 60px)',
            borderBottom: '1px solid var(--border)',
          }}
        >
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(61,107,74,0.7)',
              marginBottom: '1rem',
            }}
          >
            The researchers
          </p>
          <h1
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontWeight: 400,
              fontSize: 'clamp(2.2rem, 5vw, 4rem)',
              color: 'var(--text-primary)',
              lineHeight: 1.1,
              marginBottom: '1rem',
            }}
          >
            Meet the{' '}
            <em style={{ fontStyle: 'italic', color: '#3d6b4a' }}>team</em>
          </h1>
          <p style={{ maxWidth: '520px', fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            Engineers for Exploration is a student research group at UC San Diego dedicated to
            building technology for scientific exploration and conservation.
          </p>
        </div>

        <div style={{ maxWidth: '1100px', margin: '0 auto', padding: 'clamp(48px, 8vh, 96px) clamp(24px, 8vw, 120px)' }}>

          {/* Project Leads — portrait cards */}
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(61,107,74,0.7)',
              marginBottom: '2rem',
            }}
          >
            Project leads
          </p>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
              gap: 'clamp(24px, 4vw, 40px)',
              marginBottom: 'clamp(64px, 12vh, 112px)',
            }}
          >
            {LEADS.map(m => (
              <div
                key={m.id}
                data-testid="team-card"
                style={{ display: 'flex', flexDirection: 'column' }}
              >
                {/* Portrait photo */}
                <div
                  data-testid="team-avatar"
                  style={{
                    aspectRatio: '3 / 4',
                    overflow: 'hidden',
                    borderRadius: '0',
                    marginBottom: '20px',
                  }}
                >
                  <img
                    src={m.photo}
                    alt={m.name}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block',
                      transition: 'transform 0.5s ease',
                    }}
                    onMouseEnter={e => ((e.currentTarget as HTMLImageElement).style.transform = 'scale(1.03)')}
                    onMouseLeave={e => ((e.currentTarget as HTMLImageElement).style.transform = 'scale(1)')}
                  />
                </div>

                <p
                  style={{
                    fontFamily: "'Instrument Serif', serif",
                    fontSize: '1.25rem',
                    color: 'var(--text-primary)',
                    lineHeight: 1.2,
                    marginBottom: '4px',
                  }}
                >
                  {m.name}
                </p>
                <p
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.65rem',
                    letterSpacing: '0.1em',
                    textTransform: 'uppercase',
                    color: '#3d6b4a',
                    marginBottom: '12px',
                  }}
                >
                  {m.role} · {m.affiliation}
                </p>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
                  {m.bio}
                </p>
              </div>
            ))}
          </div>

          {/* Engineers — square grid */}
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(61,107,74,0.7)',
              marginBottom: '2rem',
            }}
          >
            Engineers
          </p>
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))',
              gap: 'clamp(16px, 3vw, 28px)',
              marginBottom: 'clamp(64px, 12vh, 112px)',
            }}
          >
            {ENGINEERS.map(m => (
              <div key={m.id} style={{ display: 'flex', flexDirection: 'column' }}>
                {/* Square photo */}
                <div
                  style={{
                    aspectRatio: '1 / 1',
                    overflow: 'hidden',
                    borderRadius: '0',
                    marginBottom: '12px',
                  }}
                >
                  <img
                    src={m.photo}
                    alt={m.name}
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block',
                      transition: 'transform 0.4s ease',
                    }}
                    onMouseEnter={e => ((e.currentTarget as HTMLImageElement).style.transform = 'scale(1.04)')}
                    onMouseLeave={e => ((e.currentTarget as HTMLImageElement).style.transform = 'scale(1)')}
                  />
                </div>
                <p
                  style={{
                    fontFamily: "'Instrument Serif', serif",
                    fontSize: '0.95rem',
                    color: 'var(--text-primary)',
                    lineHeight: 1.25,
                  }}
                >
                  {m.name}
                </p>
                <p
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.62rem',
                    letterSpacing: '0.08em',
                    textTransform: 'uppercase',
                    color: 'var(--text-muted)',
                    marginTop: '3px',
                  }}
                >
                  {m.role}
                </p>
              </div>
            ))}
          </div>

          {/* Advisors */}
          <p
            style={{
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.68rem',
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: 'rgba(61,107,74,0.7)',
              marginBottom: '2rem',
            }}
          >
            Faculty advisors
          </p>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '1px',
              background: 'var(--border)',
              maxWidth: '560px',
            }}
          >
            {ADVISORS.map(a => (
              <div
                key={a.name}
                style={{
                  background: 'var(--bg-page)',
                  padding: '20px 24px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  flexWrap: 'wrap',
                  gap: '8px',
                }}
              >
                <div>
                  <p style={{ fontWeight: 500, fontSize: '0.95rem', color: 'var(--text-primary)' }}>{a.name}</p>
                  <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{a.role}</p>
                </div>
                <span
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.65rem',
                    color: 'var(--text-muted)',
                    letterSpacing: '0.06em',
                  }}
                >
                  {a.affiliation}
                </span>
              </div>
            ))}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

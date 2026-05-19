import { useNavigate } from 'react-router-dom'

const BASE = import.meta.env.VITE_TILES_BASE_URL ?? ''

const LEADS = [
  {
    id: 'cheung',
    name: 'Jason Cheung',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    focus: 'Drove the expansion to planetary-scale satellite inference, architecting the end-to-end ML system spanning segmentation, super-resolution, continual learning, and real-time visualization.',
    photo: `${BASE}/team/jason-cheung.jpg`,
  },
  {
    id: 'wyre',
    name: 'Gage Wrye',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    focus: 'Led the team\'s technical vision and independently developed the project\'s core infrastructure, including data processing pipelines, deep learning models for aerial segmentation, custom loss functions, and ArcGIS toolbox integration.',
    photo: `${BASE}/team/gage-wyre.png`,
  },
  {
    id: 'swetlin',
    name: 'Nick Swetlin',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    focus: 'Orchestrates project coordination, data sourcing workflows, and stakeholder outreach.',
    photo: `${BASE}/team/nick-swetlin.jpg`,
  },
  {
    id: 'zhou',
    name: 'Andrew Zhou',
    role: 'Project Lead',
    affiliation: 'UC San Diego',
    focus: 'Shaped early project scope and contributed to the initial research direction.',
    photo: `${BASE}/team/andrew-zhou.svg`,
  },
]

export function Team() {
  const navigate = useNavigate()

  return (
    <section
      data-testid="team"
      style={{
        background: '#0e1a0f',
        padding: 'clamp(80px, 12vh, 140px) clamp(24px, 8vw, 120px)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Watermark */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: '-0.05em',
          right: 'clamp(24px, 6vw, 80px)',
          fontFamily: "'Instrument Serif', serif",
          fontSize: 'clamp(160px, 22vw, 320px)',
          color: 'rgba(61,107,74,0.06)',
          lineHeight: 1,
          userSelect: 'none',
          pointerEvents: 'none',
        }}
      >
        E4E
      </div>

      <div style={{ maxWidth: '1100px', margin: '0 auto', position: 'relative', zIndex: 1 }}>

        {/* Header row */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-end',
            flexWrap: 'wrap',
            gap: '2rem',
            marginBottom: 'clamp(48px, 8vh, 80px)',
          }}
        >
          <div>
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
            <h2
              style={{
                fontFamily: "'Instrument Serif', serif",
                fontWeight: 400,
                fontSize: 'clamp(2rem, 4.5vw, 4rem)',
                color: '#e8e0d0',
                lineHeight: 1.08,
              }}
            >
              Built by students,{' '}
              <em style={{ fontStyle: 'italic', color: 'rgba(100,180,120,0.85)' }}>
                driven by curiosity.
              </em>
            </h2>
          </div>

          <button
            onClick={() => navigate('/team')}
            style={{
              background: 'none',
              border: '1px solid rgba(232,224,208,0.2)',
              color: 'rgba(232,224,208,0.6)',
              padding: '10px 28px',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '0.78rem',
              fontFamily: "'DM Sans', sans-serif",
              letterSpacing: '0.04em',
              transition: 'border-color 0.2s ease, color 0.2s ease',
              whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(100,180,120,0.5)'
              ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(100,180,120,0.9)'
            }}
            onMouseLeave={e => {
              (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(232,224,208,0.2)'
              ;(e.currentTarget as HTMLButtonElement).style.color = 'rgba(232,224,208,0.6)'
            }}
          >
            Meet the full team →
          </button>
        </div>

        {/* Portrait cards */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 260px), 1fr))',
            gap: '2px',
          }}
        >
          {LEADS.map((m, i) => (
            <div
              key={m.id}
              data-testid="team-card"
              style={{
                display: 'flex',
                flexDirection: 'column',
                animation: `pageFadeIn 0.7s ease ${0.1 + i * 0.12}s both`,
              }}
            >
              {/* Photo */}
              <div
                data-testid="team-avatar"
                style={{
                  aspectRatio: '3 / 4',
                  overflow: 'hidden',
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
                    filter: 'grayscale(0.25) brightness(0.9)',
                    transition: 'filter 0.4s ease, transform 0.5s ease',
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLImageElement).style.filter = 'grayscale(0) brightness(1)'
                    ;(e.currentTarget as HTMLImageElement).style.transform = 'scale(1.03)'
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLImageElement).style.filter = 'grayscale(0.25) brightness(0.9)'
                    ;(e.currentTarget as HTMLImageElement).style.transform = 'scale(1)'
                  }}
                />
              </div>

              {/* Text below photo */}
              <div
                style={{
                  padding: 'clamp(20px, 3vh, 28px) 0',
                  borderTop: '1px solid rgba(255,255,255,0.06)',
                }}
              >
                <p
                  style={{
                    fontFamily: "'Instrument Serif', serif",
                    fontSize: 'clamp(1.1rem, 1.8vw, 1.35rem)',
                    color: '#e8e0d0',
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
                    color: 'rgba(61,107,74,0.7)',
                    marginBottom: '12px',
                  }}
                >
                  {m.role}
                </p>
                <p
                  style={{
                    fontFamily: "'DM Sans', sans-serif",
                    fontSize: '0.82rem',
                    color: 'rgba(232,224,208,0.4)',
                    lineHeight: 1.7,
                  }}
                >
                  {m.focus}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

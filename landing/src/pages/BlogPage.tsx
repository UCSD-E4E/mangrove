import { useNavigate } from 'react-router-dom'
import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'
import { POSTS, TAG_COLORS } from '../data/posts'

export function BlogPage() {
  const navigate = useNavigate()

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
          <p className="overline" style={{ marginBottom: '1rem' }}>Field notes</p>
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
            Blog
          </h1>
          <p style={{ maxWidth: '480px', fontSize: '0.95rem', color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            Research notes, engineering write-ups, and science explainers from the E4E Mangrove team.
          </p>
        </div>

        {/* Posts grid */}
        <div
          style={{
            maxWidth: '1100px',
            margin: '0 auto',
            padding: 'clamp(40px, 6vh, 60px) clamp(24px, 8vw, 120px)',
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
            gap: '1px',
            background: 'var(--border)',
          }}
        >
          {POSTS.map(post => (
            <article
              key={post.slug}
              className="blog-card"
              onClick={() => navigate(`/blog/${post.slug}`)}
              style={{
                background: 'var(--bg-page)',
                padding: 'clamp(28px, 4vh, 40px) clamp(24px, 3vw, 36px)',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                gap: '12px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span
                  style={{
                    display: 'inline-block',
                    padding: '3px 10px',
                    borderRadius: '3px',
                    fontSize: '0.68rem',
                    fontFamily: "'DM Mono', monospace",
                    letterSpacing: '0.08em',
                    background: `${TAG_COLORS[post.tag] ?? '#3d6b4a'}18`,
                    color: TAG_COLORS[post.tag] ?? '#3d6b4a',
                  }}
                >
                  {post.tag}
                </span>
                <span className="caption">{post.date}</span>
              </div>

              <h2
                style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontWeight: 400,
                  fontSize: 'clamp(1.1rem, 2vw, 1.4rem)',
                  color: 'var(--text-primary)',
                  lineHeight: 1.25,
                }}
              >
                {post.title}
              </h2>

              <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', lineHeight: 1.75, flex: 1 }}>
                {post.excerpt}
              </p>

              <span className="caption">{post.readTime}</span>
            </article>
          ))}
        </div>
      </main>

      <Footer />
    </div>
  )
}

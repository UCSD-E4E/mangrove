import { useParams, useNavigate } from 'react-router-dom'
import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'
import { POSTS, TAG_COLORS } from '../data/posts'
import type { Block } from '../data/posts'

const BASE = import.meta.env.VITE_TILES_BASE_URL ?? ''

function ImageBlock({ src, alt, caption }: { src?: string; alt: string; caption: string }) {
  if (src) {
    return (
      <figure style={{ margin: 'clamp(32px, 6vh, 56px) 0' }}>
        <img
          src={src.startsWith('http') ? src : `${BASE}${src}`}
          alt={alt}
          style={{ width: '100%', display: 'block', borderRadius: '4px' }}
        />
        <figcaption
          style={{
            marginTop: '12px',
            fontFamily: "'DM Mono', monospace",
            fontSize: '0.72rem',
            color: 'var(--text-muted)',
            letterSpacing: '0.04em',
            lineHeight: 1.6,
          }}
        >
          {caption}
        </figcaption>
      </figure>
    )
  }

  return (
    <figure style={{ margin: 'clamp(32px, 6vh, 56px) 0' }}>
      <div
        style={{
          width: '100%',
          aspectRatio: '16 / 9',
          background: 'var(--bg-raised)',
          border: '1px dashed var(--border)',
          borderRadius: '4px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '8px',
        }}
      >
        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: '0.7rem', color: 'var(--text-muted)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          Image pending
        </span>
        <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: '0.82rem', color: 'var(--text-muted)', maxWidth: '380px', textAlign: 'center', lineHeight: 1.5 }}>
          {alt}
        </span>
      </div>
      <figcaption
        style={{
          marginTop: '12px',
          fontFamily: "'DM Mono', monospace",
          fontSize: '0.72rem',
          color: 'var(--text-muted)',
          letterSpacing: '0.04em',
          lineHeight: 1.6,
        }}
      >
        {caption}
      </figcaption>
    </figure>
  )
}

function renderBlock(block: Block, i: number) {
  switch (block.type) {
    case 'p':
      return (
        <p
          key={i}
          style={{
            fontFamily: "'DM Sans', sans-serif",
            fontSize: 'clamp(1rem, 1.5vw, 1.1rem)',
            color: 'var(--text-secondary)',
            lineHeight: 1.9,
            marginBottom: '1.4em',
          }}
        >
          {block.text}
        </p>
      )

    case 'h2':
      return (
        <h2
          key={i}
          style={{
            fontFamily: "'Instrument Serif', serif",
            fontWeight: 400,
            fontSize: 'clamp(1.4rem, 2.5vw, 2rem)',
            color: 'var(--text-primary)',
            lineHeight: 1.2,
            marginTop: 'clamp(40px, 7vh, 64px)',
            marginBottom: '1rem',
          }}
        >
          {block.text}
        </h2>
      )

    case 'h3':
      return (
        <h3
          key={i}
          style={{
            fontFamily: "'DM Sans', sans-serif",
            fontWeight: 500,
            fontSize: 'clamp(0.95rem, 1.5vw, 1.1rem)',
            color: 'var(--text-primary)',
            letterSpacing: '0.02em',
            marginTop: '2rem',
            marginBottom: '0.75rem',
          }}
        >
          {block.text}
        </h3>
      )

    case 'pullquote':
      return (
        <blockquote
          key={i}
          style={{
            margin: 'clamp(32px, 6vh, 56px) 0',
            padding: '0 0 0 clamp(20px, 4vw, 36px)',
            borderLeft: '2px solid var(--accent)',
          }}
        >
          <p
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontStyle: 'italic',
              fontSize: 'clamp(1.1rem, 2vw, 1.4rem)',
              color: 'var(--text-primary)',
              lineHeight: 1.6,
            }}
          >
            {block.text}
          </p>
        </blockquote>
      )

    case 'image':
      return <ImageBlock key={i} src={block.src} alt={block.alt} caption={block.caption} />

    default:
      return null
  }
}

export function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>()
  const navigate = useNavigate()
  const post = POSTS.find(p => p.slug === slug)

  if (!post) {
    return (
      <div style={{ minHeight: '100vh', background: 'var(--bg-page)' }}>
        <Nav />
        <main style={{ paddingTop: '56px', padding: 'clamp(60px, 10vh, 120px) clamp(24px, 8vw, 120px)' }}>
          <p style={{ color: 'var(--text-muted)' }}>Post not found.</p>
        </main>
        <Footer />
      </div>
    )
  }

  const tagColor = TAG_COLORS[post.tag] ?? '#3d6b4a'

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-page)' }} className="page-enter">
      <Nav />

      <main style={{ paddingTop: '56px' }}>
        {/* Article header */}
        <header
          style={{
            padding: 'clamp(60px, 10vh, 100px) clamp(24px, 8vw, 120px) clamp(40px, 6vh, 60px)',
            borderBottom: '1px solid var(--border)',
            maxWidth: '1100px',
            margin: '0 auto',
          }}
        >
          {/* Back link */}
          <button
            onClick={() => navigate('/blog')}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.72rem',
              color: 'var(--text-muted)',
              letterSpacing: '0.06em',
              padding: 0,
              marginBottom: '2.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
            }}
          >
            ← Field notes
          </button>

          {/* Meta row */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
            <span
              style={{
                display: 'inline-block',
                padding: '3px 10px',
                borderRadius: '3px',
                fontSize: '0.68rem',
                fontFamily: "'DM Mono', monospace",
                letterSpacing: '0.08em',
                background: `${tagColor}18`,
                color: tagColor,
              }}
            >
              {post.tag}
            </span>
            <span className="caption">{post.date}</span>
            <span className="caption">{post.readTime}</span>
          </div>

          <h1
            style={{
              fontFamily: "'Instrument Serif', serif",
              fontWeight: 400,
              fontSize: 'clamp(2rem, 5vw, 4rem)',
              color: 'var(--text-primary)',
              lineHeight: 1.1,
              maxWidth: '760px',
            }}
          >
            {post.title}
          </h1>

          <p
            style={{
              marginTop: '1.5rem',
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 'clamp(0.95rem, 1.4vw, 1.1rem)',
              color: 'var(--text-secondary)',
              lineHeight: 1.8,
              maxWidth: '580px',
            }}
          >
            {post.excerpt}
          </p>
        </header>

        {/* Article body */}
        <article
          style={{
            maxWidth: '720px',
            margin: '0 auto',
            padding: 'clamp(48px, 8vh, 80px) clamp(24px, 8vw, 120px)',
          }}
        >
          {post.body.length > 0 ? (
            post.body.map((block, i) => renderBlock(block, i))
          ) : (
            <p style={{ color: 'var(--text-muted)', fontFamily: "'DM Mono', monospace", fontSize: '0.82rem' }}>
              Full article coming soon.
            </p>
          )}
        </article>

        {/* Divider */}
        <div style={{ maxWidth: '720px', margin: '0 auto clamp(48px, 8vh, 80px)', padding: '0 clamp(24px, 8vw, 120px)' }}>
          <div style={{ height: '1px', background: 'var(--border)' }} />
          <button
            onClick={() => navigate('/blog')}
            style={{
              marginTop: '32px',
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              fontFamily: "'DM Mono', monospace",
              fontSize: '0.72rem',
              color: 'var(--accent)',
              letterSpacing: '0.06em',
              padding: 0,
            }}
          >
            ← Back to all posts
          </button>
        </div>
      </main>

      <Footer />
    </div>
  )
}

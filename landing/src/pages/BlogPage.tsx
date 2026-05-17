import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'

const POSTS = [
  {
    slug: 'super-resolution-pipeline',
    date: 'Apr 2026',
    tag: 'ML',
    title: 'Recovering Boundaries: Our Super-Resolution Pipeline',
    excerpt:
      'Standard 10m Sentinel-2 predictions blur the tidal channels and fragmented edges that define mangrove ecosystem health. Our hybrid transformer-CNN model upsamples predictions 16 times, resolving structure that was previously invisible.',
    readTime: '8 min read',
  },
  {
    slug: 'continual-learning',
    date: 'Mar 2026',
    tag: 'Engineering',
    title: 'Continual Learning Across Regions',
    excerpt:
      'A model trained on Florida tends to forget Florida when fine-tuned on Yucatan. We fixed this with a replay buffer that keeps 10 percent of prior region samples in every training batch.',
    readTime: '6 min read',
  },
  {
    slug: 'satellite-embeddings',
    date: 'Jan 2026',
    tag: 'ML',
    title: 'Why Satellite Embeddings Are a Game Changer',
    excerpt:
      'Google\'s DINO-based satellite embeddings compress rich spectral and textural context into 64 dimensions per pixel. Prepending them to our SegFormer input lifted validation IoU by 11 points without adding labeled data.',
    readTime: '8 min read',
  },
  {
    slug: 'scaling-to-satellites',
    date: 'Dec 2025',
    tag: 'Research',
    title: 'Scaling to Satellites: What We Gain and What We Lose',
    excerpt:
      'Moving from drone to Sentinel-2 imagery multiplies coverage but trades spatial detail for global reach. We walk through the trade-offs and why 10m resolution is still useful for coastline-scale monitoring.',
    readTime: '6 min read',
  },
  {
    slug: 'segmenting-drone-imagery',
    date: 'Nov 2025',
    tag: 'ML',
    title: 'Segmenting Mangroves from Drone Imagery',
    excerpt:
      'We trained a SegFormer model on aerial drone imagery to classify mangroves, built-up land, and water at centimeter resolution. This post covers our data pipeline and the loss weighting choices that mattered most.',
    readTime: '7 min read',
  },
  {
    slug: 'why-mangroves-matter',
    date: 'Sep 2025',
    tag: 'Science',
    title: 'Why Mangroves Matter',
    excerpt:
      'Mangroves store 3 to 5 times more carbon per hectare than any terrestrial forest, filter coastal runoff, and buffer communities against storm surge. Yet we are losing them faster than we can map them.',
    readTime: '5 min read',
  },
]

const TAG_COLORS: Record<string, string> = {
  Research: '#3d6b4a',
  ML: '#4a6b8a',
  Engineering: '#6b4a3d',
  Science: '#6b5a3d',
}

export function BlogPage() {
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
              onClick={() => {}} // placeholder — wire to real post routes
              style={{
                background: 'var(--bg-page)',
                padding: 'clamp(28px, 4vh, 40px) clamp(24px, 3vw, 36px)',
                cursor: 'default',
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

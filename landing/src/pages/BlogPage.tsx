import { useNavigate } from 'react-router-dom'
import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'

const POSTS = [
  {
    slug: 'mapping-florida-mangroves',
    date: 'May 2026',
    tag: 'Research',
    title: 'Mapping Florida\'s Mangroves from Sentinel-2',
    excerpt:
      'How we processed 500 multispectral GeoTIFFs to build a high-resolution mangrove classification layer covering the entire Florida coastline and the Everglades.',
    readTime: '6 min read',
  },
  {
    slug: 'super-resolution-mangroves',
    date: 'Apr 2026',
    tag: 'ML',
    title: 'Super-Resolution for Coastal Ecosystems',
    excerpt:
      'Upsampling SegFormer predictions 4× reveals tidal channels and canopy gaps invisible at native 10m Sentinel-2 resolution. A deep-dive into our ESRGAN-based pipeline.',
    readTime: '8 min read',
  },
  {
    slug: 'pmtiles-for-science',
    date: 'Mar 2026',
    tag: 'Engineering',
    title: 'Why We Chose PMTiles for Global Tile Delivery',
    excerpt:
      'Serving 340 million tile requests from a single file. PMTiles\' HTTP range-request architecture lets us deploy worldwide without a tile server.',
    readTime: '4 min read',
  },
  {
    slug: 'gee-pipeline',
    date: 'Feb 2026',
    tag: 'Research',
    title: 'Building a Global Sentinel-2 Ingestion Pipeline with GEE',
    excerpt:
      'Google Earth Engine lets us pull cloud-free composites for any coastline on Earth. Here\'s how we structure the export pipeline for six regions.',
    readTime: '7 min read',
  },
  {
    slug: 'mangrove-carbon',
    date: 'Jan 2026',
    tag: 'Science',
    title: 'Why Mangrove Carbon Matters More Than You Think',
    excerpt:
      'Mangroves store 3–5× more carbon per hectare than terrestrial forests. Mapping their extent precisely is a prerequisite for any credible carbon accounting.',
    readTime: '5 min read',
  },
  {
    slug: 'segformer-remote-sensing',
    date: 'Dec 2025',
    tag: 'ML',
    title: 'Adapting SegFormer to 69-Band Satellite Imagery',
    excerpt:
      'SegFormer was designed for RGB images. Here\'s how we adapted its input projection to handle Google\'s 64-dimensional DINO satellite embeddings.',
    readTime: '9 min read',
  },
]

const TAG_COLORS: Record<string, string> = {
  Research: '#3d6b4a',
  ML: '#4a6b8a',
  Engineering: '#6b4a3d',
  Science: '#6b5a3d',
}

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

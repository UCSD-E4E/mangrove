import { useNavigate } from 'react-router-dom'

const NAV_LINKS = [
  { label: 'Visualizer', href: '/visualizer', internal: true },
  { label: 'Blog', href: '/blog', internal: true },
  { label: 'Team', href: '/team', internal: true },
  { label: 'Collaborate', href: '/collaborate', internal: true },
  { label: 'GitHub', href: 'https://github.com/UCSD-E4E', internal: false },
]

export function Footer() {
  const navigate = useNavigate()

  return (
    <footer
      data-testid="footer"
      style={{
        padding: '28px 24px',
        backgroundColor: 'var(--bg-page)',
        borderTop: '1px solid var(--border)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '12px',
      }}
    >
      <p
        data-testid="copyright"
        style={{
          fontSize: '0.78rem',
          color: 'var(--text-muted)',
          fontFamily: "'DM Mono', monospace",
        }}
      >
        © 2025 Engineers for Exploration, UC San Diego
      </p>

      <nav style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
        {NAV_LINKS.map(link => (
          link.internal ? (
            <button
              key={link.label}
              onClick={() => navigate(link.href)}
              style={{
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                padding: 0,
                fontFamily: "'DM Sans', sans-serif",
              }}
            >
              {link.label}
            </button>
          ) : (
            <a
              key={link.label}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              style={{
                fontSize: '0.78rem',
                color: 'var(--text-muted)',
                textDecoration: 'none',
              }}
            >
              {link.label}
            </a>
          )
        ))}
      </nav>
    </footer>
  )
}

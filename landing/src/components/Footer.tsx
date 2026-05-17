const NAV_LINKS = [
  { label: 'Research', href: '#research' },
  { label: 'Regions', href: '#regions' },
  { label: 'GitHub', href: 'https://github.com/UCSD-E4E' },
]

export function Footer() {
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

      <nav style={{ display: 'flex', gap: '24px' }}>
        {NAV_LINKS.map(link => (
          <a
            key={link.label}
            href={link.href}
            style={{
              fontSize: '0.78rem',
              color: 'var(--text-muted)',
              textDecoration: 'none',
            }}
          >
            {link.label}
          </a>
        ))}
      </nav>
    </footer>
  )
}

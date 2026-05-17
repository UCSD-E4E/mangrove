export function JoinCTA() {
  return (
    <section
      data-testid="join-cta"
      style={{
        padding: '100px 24px',
        backgroundColor: 'var(--accent)',
        textAlign: 'center',
      }}
    >
      <div style={{ maxWidth: '640px', margin: '0 auto' }}>
        <p
          className="overline"
          style={{ marginBottom: '1rem', color: 'rgba(255,255,255,0.65)' }}
        >
          Open science
        </p>

        <h2 style={{ marginBottom: '1.5rem', color: '#fff' }}>
          Collaborate with{' '}
          <em className="serif-italic">E4E</em>
        </h2>

        <p
          style={{
            color: 'rgba(255,255,255,0.8)',
            lineHeight: 1.8,
            marginBottom: '2.5rem',
            fontSize: '1rem',
          }}
        >
          We partner with conservation organizations, remote sensing labs, and
          satellite data providers. If you have data, expertise, or ideas —
          let's work together.
        </p>

        <a
          href="mailto:e4e@ucsd.edu"
          data-testid="cta-button"
          style={{
            display: 'inline-block',
            padding: '14px 36px',
            backgroundColor: '#fff',
            color: 'var(--accent)',
            borderRadius: '999px',
            fontWeight: 600,
            fontSize: '0.9rem',
            textDecoration: 'none',
          }}
        >
          Get in touch →
        </a>
      </div>
    </section>
  )
}

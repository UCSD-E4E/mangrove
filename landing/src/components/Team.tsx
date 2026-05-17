const MEMBERS = [
  { id: 'cho', name: 'Jason Cho', role: 'ML Research Lead', affiliation: 'UC San Diego' },
  { id: 'chen', name: 'Vivian Chen', role: 'Remote Sensing', affiliation: 'UC San Diego' },
  { id: 'park', name: 'David Park', role: 'Data Engineering', affiliation: 'UC San Diego' },
]

export function Team() {
  return (
    <section
      data-testid="team"
      style={{
        padding: '80px 24px',
        backgroundColor: 'var(--bg-raised)',
      }}
    >
      <div style={{ maxWidth: '960px', margin: '0 auto' }}>
        <p className="overline" style={{ marginBottom: '1rem' }}>The researchers</p>

        <h2 style={{ marginBottom: '3rem' }}>
          Meet the <em className="serif-italic">team</em>
        </h2>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '24px',
          }}
        >
          {MEMBERS.map(member => (
            <div
              key={member.id}
              data-testid="team-card"
              style={{
                background: 'var(--bg-panel)',
                border: '1px solid var(--border)',
                borderRadius: '8px',
                padding: '28px 24px',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
                gap: '6px',
              }}
            >
              <div
                data-testid="team-avatar"
                style={{
                  width: '56px',
                  height: '56px',
                  borderRadius: '50%',
                  backgroundColor: 'var(--accent)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#fff',
                  fontFamily: "'DM Mono', monospace",
                  fontSize: '0.9rem',
                  marginBottom: '10px',
                }}
              >
                {member.name.split(' ').map(w => w[0]).join('')}
              </div>

              <p style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text-primary)' }}>
                {member.name}
              </p>

              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                {member.role}
              </p>

              <p
                style={{
                  fontSize: '0.72rem',
                  color: 'var(--text-muted)',
                  fontFamily: "'DM Mono', monospace",
                }}
              >
                {member.affiliation}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}

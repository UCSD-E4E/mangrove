import type { CSSProperties } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useNavHide } from '../hooks/useNavHide'

const NAV_LINKS = [
  { label: 'Visualizer', href: '/visualizer' },
  { label: 'Blog', href: '/blog' },
  { label: 'Team', href: '/team' },
]

const linkStyle: CSSProperties = {
  color: 'var(--text-muted)',
  textDecoration: 'none',
  fontSize: '0.75rem',
  letterSpacing: '0.04em',
  fontFamily: "'DM Sans', sans-serif",
  transition: 'color 0.2s ease',
  cursor: 'pointer',
  background: 'none',
  border: 'none',
  padding: 0,
}

export function Nav() {
  const { hidden, scrolled } = useNavHide()
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <header
      data-hidden={String(hidden)}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        height: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 48px',
        background: 'rgba(247, 245, 240, 0.92)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: `1px solid ${scrolled ? 'var(--border)' : 'transparent'}`,
        transform: hidden ? 'translateY(-100%)' : 'translateY(0)',
        transition: 'transform 0.3s ease, border-color 0.2s ease',
      }}
    >
      {/* Wordmark */}
      <button
        onClick={() => navigate('/')}
        style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
      >
        <span style={{
          fontFamily: "'Instrument Serif', serif",
          color: 'var(--text-primary)',
          fontSize: '1.05rem',
        }}>
          Mangrove Monitor
        </span>
        <span style={{
          fontFamily: "'DM Mono', monospace",
          fontSize: '0.72rem',
          color: 'var(--text-muted)',
        }}>
          · E4E Lab, UC San Diego
        </span>
      </button>

      {/* Center links */}
      <nav aria-label="Site navigation" style={{ display: 'flex', gap: '3rem' }}>
        {NAV_LINKS.map(({ label, href }) => (
          <button
            key={label}
            onClick={() => navigate(href)}
            style={{
              ...linkStyle,
              color: location.pathname === href ? 'var(--accent)' : 'var(--text-muted)',
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* CTA */}
      <button
        onClick={() => navigate('/collaborate')}
        style={{
          background: 'none',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
          padding: '0.4rem 1.1rem',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '0.75rem',
          letterSpacing: '0.04em',
          fontFamily: "'DM Sans', sans-serif",
          transition: 'border-color 0.2s ease, color 0.2s ease',
        }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--accent)'
          ;(e.currentTarget as HTMLButtonElement).style.color = 'var(--accent)'
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)'
          ;(e.currentTarget as HTMLButtonElement).style.color = 'var(--text-secondary)'
        }}
      >
        Collaborate →
      </button>
    </header>
  )
}

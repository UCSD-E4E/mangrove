import { useState } from 'react'
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
  const [menuOpen, setMenuOpen] = useState(false)

  const go = (href: string) => {
    setMenuOpen(false)
    navigate(href)
  }

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
        padding: '0 clamp(20px, 4vw, 48px)',
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
        onClick={() => go('/')}
        style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem', background: 'none', border: 'none', cursor: 'pointer', padding: 0, flexShrink: 0 }}
      >
        <span style={{
          fontFamily: "'Instrument Serif', serif",
          color: 'var(--text-primary)',
          fontSize: '1.05rem',
        }}>
          Mangrove Monitoring
        </span>
        <span className="nav-subtitle" style={{
          fontFamily: "'DM Mono', monospace",
          fontSize: '0.72rem',
          color: 'var(--text-muted)',
          display: menuOpen ? 'none' : undefined,
        }}>
          · E4E Lab, UC San Diego
        </span>
      </button>

      {/* Center links — hidden on mobile via CSS */}
      <nav aria-label="Site navigation" className="nav-center" style={{ gap: '3rem' }}>
        {NAV_LINKS.map(({ label, href }) => (
          <button
            key={label}
            onClick={() => go(href)}
            style={{
              ...linkStyle,
              color: location.pathname === href ? 'var(--accent)' : 'var(--text-muted)',
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      {/* CTA — hidden on mobile via CSS */}
      <button
        className="nav-cta-btn"
        onClick={() => go('/collaborate')}
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

      {/* Hamburger — shown on mobile via CSS */}
      <button
        className="nav-hamburger"
        onClick={() => setMenuOpen(v => !v)}
        aria-label={menuOpen ? 'Close menu' : 'Open menu'}
        aria-expanded={menuOpen}
      >
        <span style={{ width: 22, height: 1.5, background: menuOpen ? 'transparent' : 'var(--text-primary)', display: 'block', transition: '0.2s' }} />
        <span style={{ width: 22, height: 1.5, background: 'var(--text-primary)', display: 'block', transition: '0.2s',
          transform: menuOpen ? 'rotate(45deg) translateY(-3px)' : 'none' }} />
        <span style={{ width: menuOpen ? 22 : 14, height: 1.5, background: 'var(--text-primary)', display: 'block', transition: '0.2s',
          transform: menuOpen ? 'rotate(-45deg) translateY(3px)' : 'none' }} />
      </button>

      {/* Mobile drawer */}
      {menuOpen && (
        <div className="nav-drawer">
          {NAV_LINKS.map(({ label, href }) => (
            <button
              key={label}
              onClick={() => go(href)}
              style={{
                ...linkStyle,
                fontSize: '1rem',
                textAlign: 'left',
                color: location.pathname === href ? 'var(--accent)' : 'var(--text-secondary)',
              }}
            >
              {label}
            </button>
          ))}
          <button
            onClick={() => go('/collaborate')}
            style={{ ...linkStyle, fontSize: '1rem', textAlign: 'left', color: 'var(--accent)' }}
          >
            Collaborate →
          </button>
        </div>
      )}
    </header>
  )
}

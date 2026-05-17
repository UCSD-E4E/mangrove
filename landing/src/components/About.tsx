import { useRef, useState, useEffect } from 'react'
import { useCountUp } from '../hooks/useCountUp'

const STATS = [
  {
    id: 'coverage',
    target: 147000,
    format: (v: number) => `${Math.round(v / 1000)}k km²`,
    label: 'mangrove area monitored',
  },
  {
    id: 'scale',
    target: 4,
    format: (v: number) => `${Math.round(v)}×`,
    label: 'higher resolution than prior methods',
  },
  {
    id: 'resolution',
    target: 0.35,
    format: (v: number) => `${v.toFixed(2)}m`,
    label: 'ground sampling distance',
  },
  {
    id: 'regions',
    target: 6,
    format: (v: number) => `${Math.round(v)} regions`,
    label: 'sites covered globally',
  },
]

function StatCell({
  target,
  format,
  label,
  isInView,
  borderRight,
  borderBottom,
}: {
  target: number
  format: (v: number) => string
  label: string
  isInView: boolean
  borderRight?: boolean
  borderBottom?: boolean
}) {
  const value = useCountUp({ target, duration: 2000, isInView })
  return (
    <div
      data-testid="stat-cell"
      style={{
        padding: '32px 28px',
        borderRight: borderRight ? '1px solid var(--border)' : undefined,
        borderBottom: borderBottom ? '1px solid var(--border)' : undefined,
      }}
    >
      <p
        style={{
          fontFamily: "'Instrument Serif', serif",
          fontWeight: 400,
          fontSize: 'clamp(1.8rem, 2.8vw, 2.8rem)',
          color: 'var(--text-primary)',
          marginBottom: '0.35rem',
          lineHeight: 1,
        }}
      >
        {format(value)}
      </p>
      <p className="caption">{label}</p>
    </div>
  )
}

export function About() {
  const sectionRef = useRef<HTMLElement>(null)
  const [isInView, setIsInView] = useState(false)

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setIsInView(true) },
      { threshold: 0.1 },
    )
    if (sectionRef.current) observer.observe(sectionRef.current)
    return () => observer.disconnect()
  }, [])

  return (
    <section
      ref={sectionRef}
      data-testid="about"
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        backgroundColor: 'var(--bg-panel)',
        opacity: isInView ? 1 : 0,
        transform: isInView ? 'translateY(0)' : 'translateY(32px)',
        transition: 'opacity 0.8s ease, transform 0.8s ease',
      }}
    >
      {/* Left column */}
      <div style={{ padding: '56px 48px' }}>
        <p className="overline" style={{ marginBottom: '1rem' }}>The Science</p>
        <h2 style={{ marginBottom: '1.5rem' }}>
          We map them to{' '}
          <em className="serif-italic">track them</em>
        </h2>
        <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8, marginBottom: '1rem', maxWidth: '420px' }}>
          Mangroves sequester carbon at rates up to five times higher than tropical forests, yet
          traditional mapping relies on coarse satellite imagery that misses individual canopies.
        </p>
        <p style={{ color: 'var(--text-secondary)', lineHeight: 1.8, maxWidth: '420px' }}>
          Our model fuses drone surveys with Sentinel-2 embeddings to produce sub-meter land cover
          predictions — enabling conservation teams to detect change at the scale that matters.
        </p>
      </div>

      {/* Right column — 2×2 stat grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          borderLeft: '1px solid var(--border)',
        }}
      >
        {STATS.map((stat, i) => (
          <StatCell
            key={stat.id}
            target={stat.target}
            format={stat.format}
            label={stat.label}
            isInView={isInView}
            borderRight={i % 2 === 0}
            borderBottom={i < 2}
          />
        ))}
      </div>
    </section>
  )
}

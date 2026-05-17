import { useEffect, useRef } from 'react'

export function IntroHero() {
  const lineRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = lineRef.current
    if (!el) return
    el.style.transform = 'scaleX(0)'
    el.style.transformOrigin = 'left'
    const t = setTimeout(() => {
      el.style.transition = 'transform 1.2s cubic-bezier(0.22,1,0.36,1) 0.6s'
      el.style.transform = 'scaleX(1)'
    }, 50)
    return () => clearTimeout(t)
  }, [])

  return (
    <section
      style={{
        minHeight: '100vh',
        background: '#f7f4ee',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: 'clamp(80px, 12vh, 120px) clamp(24px, 8vw, 120px) clamp(60px, 8vh, 100px)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Soft organic blob — top right */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          top: '-10%',
          right: '-5%',
          width: 'clamp(360px, 45vw, 680px)',
          height: 'clamp(360px, 45vw, 680px)',
          borderRadius: '60% 40% 55% 45% / 45% 55% 40% 60%',
          background:
            'radial-gradient(ellipse at 40% 40%, rgba(100,175,120,0.13) 0%, rgba(61,107,74,0.06) 50%, transparent 75%)',
          pointerEvents: 'none',
        }}
      />

      {/* Faint second blob — bottom left */}
      <div
        aria-hidden
        style={{
          position: 'absolute',
          bottom: '-8%',
          left: '-4%',
          width: 'clamp(260px, 32vw, 480px)',
          height: 'clamp(260px, 32vw, 480px)',
          borderRadius: '45% 55% 60% 40% / 55% 40% 60% 45%',
          background:
            'radial-gradient(ellipse at 60% 60%, rgba(100,175,120,0.09) 0%, transparent 70%)',
          pointerEvents: 'none',
        }}
      />

      <div style={{ maxWidth: '1100px', margin: '0 auto', width: '100%', position: 'relative', zIndex: 1 }}>

        {/* Overline */}
        <p
          style={{
            fontFamily: "'DM Mono', monospace",
            fontSize: '0.68rem',
            letterSpacing: '0.2em',
            textTransform: 'uppercase',
            color: 'rgba(61,107,74,0.7)',
            marginBottom: '2.5rem',
            animation: 'pageFadeIn 0.8s ease 0.1s both',
          }}
        >
          Engineers for Exploration · UC San Diego
        </p>

        {/* Headline */}
        <h1
          style={{
            fontFamily: "'Instrument Serif', serif",
            fontWeight: 400,
            fontSize: 'clamp(3rem, 7vw, 6.5rem)',
            color: '#1a1a18',
            lineHeight: 1.04,
            maxWidth: '820px',
            animation: 'pageFadeIn 0.9s ease 0.2s both',
          }}
        >
          Watching over
          <br />
          the world's{' '}
          <em style={{ fontStyle: 'italic', color: '#3d6b4a' }}>mangroves</em>
          <br />
          from space.
        </h1>

        {/* Rule */}
        <div
          ref={lineRef}
          style={{
            height: '1px',
            width: 'clamp(80px, 12vw, 180px)',
            background: 'rgba(61,107,74,0.35)',
            margin: 'clamp(28px, 4vh, 48px) 0',
          }}
        />

        {/* Subtext + stat row */}
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'flex-end',
            gap: 'clamp(32px, 6vw, 80px)',
            animation: 'pageFadeIn 1s ease 0.4s both',
          }}
        >
          <p
            style={{
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 'clamp(0.95rem, 1.4vw, 1.1rem)',
              color: '#666660',
              lineHeight: 1.85,
              maxWidth: '420px',
            }}
          >
            We build machine learning pipelines that turn raw satellite imagery
            into continuous, high-resolution maps of every mangrove coastline on
            Earth.
          </p>

          {/* Inline stats */}
          <div style={{ display: 'flex', gap: 'clamp(28px, 5vw, 60px)', flexWrap: 'wrap' }}>
            {[
              { value: '147M', label: 'hectares mapped' },
              { value: '94%', label: 'model accuracy' },
              { value: '5-day', label: 'update cycle' },
            ].map(s => (
              <div key={s.value}>
                <div
                  style={{
                    fontFamily: "'Instrument Serif', serif",
                    fontSize: 'clamp(1.6rem, 3vw, 2.4rem)',
                    color: '#1a1a18',
                    lineHeight: 1,
                  }}
                >
                  {s.value}
                </div>
                <div
                  style={{
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.65rem',
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase',
                    color: 'rgba(61,107,74,0.65)',
                    marginTop: '6px',
                  }}
                >
                  {s.label}
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* Scroll cue — bottom-left of section, clear of text column */}
      <div
        style={{
          position: 'absolute',
          bottom: 'clamp(28px, 4vh, 44px)',
          left: 'clamp(16px, 2.5vw, 36px)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: '10px',
          animation: 'pageFadeIn 1s ease 1s both',
          zIndex: 2,
        }}
      >
        <span
          style={{
            fontFamily: "'DM Mono', monospace",
            fontSize: '0.6rem',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: 'rgba(26,26,24,0.3)',
            writingMode: 'vertical-rl',
            transform: 'rotate(180deg)',
          }}
        >
          Scroll
        </span>
        <div
          style={{
            width: '1px',
            height: '48px',
            background: 'linear-gradient(to bottom, rgba(61,107,74,0.4), transparent)',
            animation: 'floatUp 2s ease-in-out infinite',
          }}
        />
      </div>
    </section>
  )
}

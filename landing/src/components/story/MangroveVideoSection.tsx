import { useRef, useEffect, useState, useCallback } from 'react'

interface Props {
  /** Optional: path to a drone/aerial mangrove video (15–30s recommended).
   *  When omitted, a CSS-animated background is used as a placeholder. */
  videoSrc?: string
}

const PHASES = [
  {
    range: [0, 0.2] as [number, number],
    lines: ['Mangroves'],
    sub: null,
  },
  {
    range: [0.2, 0.4] as [number, number],
    lines: ['Ancient forests', 'at the edge of the sea'],
    sub: null,
  },
  {
    range: [0.4, 0.6] as [number, number],
    lines: ['They breathe', 'through their roots'],
    sub: 'Pneumatophores pierce the tidal mud to reach air',
  },
  {
    range: [0.6, 0.8] as [number, number],
    lines: ['Carbon stored', '4× more than rainforests'],
    sub: 'The densest blue-carbon ecosystem on Earth',
  },
  {
    range: [0.8, 1.0] as [number, number],
    lines: ['Home to 1,500+', 'species'],
    sub: 'And shelter for 800 million people along coastlines worldwide',
  },
]

export function MangroveVideoSection({ videoSrc }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [progress, setProgress] = useState(0)
  const [phaseKey, setPhaseKey] = useState(0)
  const lastPhaseIdx = useRef(-1)
  const targetProgress = useRef(0)
  const rafId = useRef(0)

  const handleScroll = useCallback(() => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const scrollable = containerRef.current.offsetHeight - window.innerHeight
    const p = Math.max(0, Math.min(1, -rect.top / scrollable))
    targetProgress.current = p
    setProgress(p)
    const idx = PHASES.findIndex(ph => p >= ph.range[0] && p < ph.range[1])
    if (idx !== lastPhaseIdx.current) {
      lastPhaseIdx.current = idx
      setPhaseKey(k => k + 1)
    }
  }, [])

  // Smooth video scrub via rAF lerp — decoupled from scroll events
  useEffect(() => {
    const tick = () => {
      const video = videoRef.current
      if (video?.duration) {
        const target = targetProgress.current * video.duration
        const current = video.currentTime
        const diff = target - current
        // Fast lerp: close enough → snap, otherwise smooth
        video.currentTime = Math.abs(diff) < 0.01 ? target : current + diff * 0.1
      }
      rafId.current = requestAnimationFrame(tick)
    }
    rafId.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafId.current)
  }, [])

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  const activePhase =
    PHASES.find(ph => progress >= ph.range[0] && progress < ph.range[1]) ??
    PHASES[PHASES.length - 1]

  const activeIdx = PHASES.indexOf(activePhase)

  return (
    <div ref={containerRef} style={{ height: '500vh', position: 'relative' }}>
      <div
        style={{
          position: 'sticky',
          top: 0,
          height: '100vh',
          overflow: 'hidden',
          background: '#040b05',
        }}
      >
        {/* Background: video or animated CSS fallback */}
        {videoSrc ? (
          <video
            ref={videoRef}
            src={videoSrc}
            muted
            playsInline
            preload="auto"
            onLoadedMetadata={() => handleScroll()}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              opacity: 0.75,
            }}
          />
        ) : (
          <MangroveBackground progress={progress} />
        )}

        {/* Vignette */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            background:
              'radial-gradient(ellipse 80% 80% at 50% 50%, transparent 30%, rgba(4,11,5,0.7) 100%)',
            pointerEvents: 'none',
          }}
        />

        {/* Section label */}
        <div
          style={{
            position: 'absolute',
            top: '32px',
            left: 'clamp(20px, 4vw, 48px)',
            fontFamily: "'DM Mono', monospace",
            fontSize: '0.68rem',
            letterSpacing: '0.18em',
            textTransform: 'uppercase',
            color: 'rgba(200,220,200,0.45)',
          }}
        >
          What are mangroves
        </div>

        {/* Text overlay */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '0 clamp(24px, 8vw, 120px)',
            textAlign: 'center',
          }}
        >
          <div
            key={phaseKey}
            style={{ animation: 'storyFadeIn 0.55s cubic-bezier(0.22,1,0.36,1) forwards' }}
          >
            {activePhase.lines.map((line, i) => (
              <div
                key={i}
                style={{
                  fontFamily: "'Instrument Serif', serif",
                  fontSize: 'clamp(2.4rem, 7.5vw, 7.5rem)',
                  color: i === 0 ? '#e8e0d0' : 'rgba(232,224,208,0.82)',
                  lineHeight: 1.04,
                  fontWeight: 400,
                  fontStyle: i === 1 && activePhase.lines.length > 1 ? 'italic' : 'normal',
                }}
              >
                {line}
              </div>
            ))}
            {activePhase.sub && (
              <p
                style={{
                  marginTop: '2rem',
                  fontFamily: "'DM Sans', sans-serif",
                  fontSize: 'clamp(0.8rem, 1.5vw, 1rem)',
                  color: 'rgba(232,224,208,0.5)',
                  letterSpacing: '0.04em',
                  maxWidth: '560px',
                  margin: '1.5rem auto 0',
                  lineHeight: 1.7,
                }}
              >
                {activePhase.sub}
              </p>
            )}
          </div>
        </div>

        {/* Phase indicator dots */}
        <div
          style={{
            position: 'absolute',
            bottom: '36px',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
          }}
        >
          {PHASES.map((_, i) => (
            <div
              key={i}
              style={{
                width: i === activeIdx ? '28px' : '5px',
                height: '3px',
                borderRadius: '2px',
                background:
                  i <= activeIdx
                    ? 'rgba(200,220,200,0.75)'
                    : 'rgba(200,220,200,0.18)',
                transition: 'width 0.35s ease, background 0.35s ease',
              }}
            />
          ))}
        </div>

        {/* Scroll progress line */}
        <div
          style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            height: '2px',
            width: `${progress * 100}%`,
            background:
              'linear-gradient(to right, rgba(61,107,74,0.4), rgba(100,200,120,0.6))',
            transition: 'width 0.1s linear',
          }}
        />
      </div>
    </div>
  )
}

function MangroveBackground({ progress }: { progress: number }) {
  const y = 50 + progress * 12
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: `
          radial-gradient(ellipse 70% 50% at 50% ${y}%, #0d2e14 0%, transparent 65%),
          radial-gradient(ellipse 100% 60% at 20% 80%, #091a0b 0%, transparent 55%),
          radial-gradient(ellipse 60% 60% at 80% 20%, #071509 0%, transparent 50%),
          linear-gradient(180deg, #040b05 0%, #060e07 100%)
        `,
      }}
    >
      <div className="mangrove-rays" />
      {/* Root silhouettes */}
      <svg
        viewBox="0 0 1440 900"
        preserveAspectRatio="xMidYMid slice"
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          opacity: 0.18 + progress * 0.08,
        }}
      >
        {[120, 280, 440, 620, 800, 960, 1100, 1280].map((x, i) => (
          <g key={i} transform={`translate(${x}, 900)`}>
            <path
              d={`M0,0 C${-20 - i * 3},${-100 - i * 20} ${10 + i * 5},${-200 - i * 15} ${-5 + i * 2},${-350 - i * 30}`}
              stroke="rgba(40,120,50,0.6)"
              strokeWidth={2 + (i % 3)}
              fill="none"
            />
            <path
              d={`M${-15 + i},0 C${10 - i * 2},${-80} ${-20},${-160 - i * 10} ${5},${-300}`}
              stroke="rgba(30,100,40,0.45)"
              strokeWidth={1.5}
              fill="none"
            />
          </g>
        ))}
        {/* Water surface shimmer */}
        <ellipse
          cx={720}
          cy={700}
          rx={900}
          ry={80}
          fill="rgba(20,80,40,0.12)"
        />
      </svg>
    </div>
  )
}

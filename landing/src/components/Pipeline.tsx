import { useCallback, useEffect, useRef, useState } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

const STEPS = [
  {
    id: 'sentinel',
    step: '01',
    label: 'Sentinel-2 Input',
    description: '10 m multispectral imagery',
  },
  {
    id: 'encoder',
    step: '02',
    label: 'Contrastive Encoder',
    description: 'DINO-based feature extraction',
  },
  {
    id: 'segmentation',
    step: '03',
    label: 'Segmentation Head',
    description: 'SegFormer pixel classification',
  },
  {
    id: 'superres',
    step: '04',
    label: 'Super-Resolution',
    description: '28× SR-UNet upsampling',
  },
]

export function Pipeline() {
  const sectionRef = useRef<HTMLElement>(null)
  const [activeSteps, setActiveSteps] = useState<Set<string>>(new Set())

  const activateStep = useCallback((stepId: string) => {
    setActiveSteps(prev => new Set([...prev, stepId]))
  }, [])

  useEffect(() => {
    const ctx = gsap.context(() => {
      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top 70%',
        onEnter: () => {
          STEPS.forEach((step, i) => {
            gsap.delayedCall(i * 0.3, () => activateStep(step.id))
          })
        },
      })
    }, sectionRef)
    return () => ctx.revert()
  }, [activateStep])

  return (
    <section
      ref={sectionRef}
      data-testid="pipeline"
      style={{
        padding: '80px 24px',
        backgroundColor: 'var(--bg-page)',
      }}
    >
      <div style={{ maxWidth: '960px', margin: '0 auto' }}>
        <p className="overline" style={{ marginBottom: '1rem' }}>How it works</p>

        <h2 style={{ marginBottom: '3rem' }}>
          The <em className="serif-italic">pipeline</em>
        </h2>

        <div style={{ display: 'flex', alignItems: 'flex-start' }}>
          {STEPS.flatMap((step, i) => {
            const isActive = activeSteps.has(step.id)
            const items = []

            items.push(
              <div
                key={step.id}
                data-testid="pipeline-node"
                data-step-id={step.id}
                data-active={isActive ? 'true' : 'false'}
                style={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center',
                  opacity: isActive ? 1 : 0.35,
                  transition: 'opacity 0.4s ease',
                }}
              >
                <div
                  style={{
                    width: '52px',
                    height: '52px',
                    borderRadius: '50%',
                    border: `2px solid ${isActive ? 'var(--accent)' : 'var(--border)'}`,
                    backgroundColor: isActive ? 'var(--accent)' : 'transparent',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    marginBottom: '1rem',
                    transition: 'border-color 0.4s ease, background-color 0.4s ease',
                    color: isActive ? '#fff' : 'var(--text-muted)',
                    fontFamily: "'DM Mono', monospace",
                    fontSize: '0.7rem',
                  }}
                >
                  {step.step}
                </div>

                <p
                  style={{
                    fontWeight: 600,
                    fontSize: '0.9rem',
                    marginBottom: '0.4rem',
                    color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)',
                    transition: 'color 0.4s ease',
                  }}
                >
                  {step.label}
                </p>

                <p
                  style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-muted)',
                    lineHeight: 1.5,
                    maxWidth: '140px',
                  }}
                >
                  {step.description}
                </p>
              </div>
            )

            if (i < STEPS.length - 1) {
              items.push(
                <div
                  key={`conn-${i}`}
                  data-testid="pipeline-connector"
                  style={{
                    flexShrink: 0,
                    width: '40px',
                    height: '2px',
                    marginTop: '26px',
                    backgroundColor: isActive ? 'var(--accent)' : 'var(--border)',
                    transition: 'background-color 0.4s ease',
                  }}
                />
              )
            }

            return items
          })}
        </div>
      </div>
    </section>
  )
}

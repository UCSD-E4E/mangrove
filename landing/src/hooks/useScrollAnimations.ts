import { useEffect } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export function useScrollAnimations() {
  useEffect(() => {
    const ctx = gsap.context(() => {
      gsap.from('[data-testid="split-compare"]', {
        opacity: 0,
        y: 48,
        duration: 0.8,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: '[data-testid="split-compare"]',
          start: 'top 80%',
        },
      })

      gsap.from('[data-testid="team-card"]', {
        opacity: 0,
        y: 32,
        duration: 0.6,
        stagger: 0.12,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: '[data-testid="team"]',
          start: 'top 75%',
        },
      })

      gsap.from('[data-testid="join-cta"]', {
        opacity: 0,
        y: 48,
        duration: 0.8,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: '[data-testid="join-cta"]',
          start: 'top 80%',
        },
      })
    })

    return () => ctx.revert()
  }, [])
}

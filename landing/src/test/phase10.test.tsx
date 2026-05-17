import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'

import { Team } from '../components/Team'
import { JoinCTA } from '../components/JoinCTA'
import { Footer } from '../components/Footer'

describe('Phase 10 — Team', () => {
  it('renders without throwing', () => {
    expect(() => render(<Team />)).not.toThrow()
  })

  it('section has data-testid="team"', () => {
    render(<Team />)
    expect(screen.getByTestId('team')).toBeInTheDocument()
  })

  it('renders exactly 3 team cards', () => {
    render(<Team />)
    expect(screen.getAllByTestId('team-card')).toHaveLength(3)
  })

  it('"The researchers" overline is present', () => {
    render(<Team />)
    expect(screen.getByText('The researchers')).toBeInTheDocument()
  })

  it('each card has a non-empty avatar', () => {
    render(<Team />)
    const avatars = screen.getAllByTestId('team-avatar')
    expect(avatars).toHaveLength(3)
    avatars.forEach(av => expect(av.textContent?.trim().length).toBeGreaterThan(0))
  })

  it('all three affiliation labels say "UC San Diego"', () => {
    render(<Team />)
    const cards = screen.getAllByTestId('team-card')
    cards.forEach(card => expect(card.textContent).toContain('UC San Diego'))
  })
})

describe('Phase 10 — JoinCTA', () => {
  it('renders without throwing', () => {
    expect(() => render(<JoinCTA />)).not.toThrow()
  })

  it('section has data-testid="join-cta"', () => {
    render(<JoinCTA />)
    expect(screen.getByTestId('join-cta')).toBeInTheDocument()
  })

  it('CTA button is present', () => {
    render(<JoinCTA />)
    expect(screen.getByTestId('cta-button')).toBeInTheDocument()
  })

  it('CTA button text is "Get in touch →"', () => {
    render(<JoinCTA />)
    expect(screen.getByTestId('cta-button').textContent).toBe('Get in touch →')
  })

  it('"Open science" overline is present', () => {
    render(<JoinCTA />)
    expect(screen.getByText('Open science')).toBeInTheDocument()
  })

  it('CTA button is an anchor tag with an href', () => {
    render(<JoinCTA />)
    const btn = screen.getByTestId('cta-button')
    expect(btn.tagName).toBe('A')
    expect(btn.getAttribute('href')).toBeTruthy()
  })
})

describe('Phase 10 — Footer', () => {
  it('renders without throwing', () => {
    expect(() => render(<Footer />)).not.toThrow()
  })

  it('footer has data-testid="footer"', () => {
    render(<Footer />)
    expect(screen.getByTestId('footer')).toBeInTheDocument()
  })

  it('copyright element is present', () => {
    render(<Footer />)
    expect(screen.getByTestId('copyright')).toBeInTheDocument()
  })

  it('copyright mentions "Engineers for Exploration"', () => {
    render(<Footer />)
    expect(screen.getByTestId('copyright').textContent).toContain('Engineers for Exploration')
  })

  it('footer has 3 nav links', () => {
    render(<Footer />)
    const links = screen.getByTestId('footer').querySelectorAll('nav a')
    expect(links).toHaveLength(3)
  })

  it('"GitHub" link is present in footer nav', () => {
    render(<Footer />)
    expect(screen.getByText('GitHub')).toBeInTheDocument()
  })
})

import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'

import { Team } from '../components/Team'
import { JoinCTA } from '../components/JoinCTA'
import { Footer } from '../components/Footer'

describe('Phase 10 — Team', () => {
  it('renders without throwing', () => {
    expect(() => render(<MemoryRouter><Team /></MemoryRouter>)).not.toThrow()
  })

  it('section has data-testid="team"', () => {
    render(<MemoryRouter><Team /></MemoryRouter>)
    expect(screen.getByTestId('team')).toBeInTheDocument()
  })

  it('renders exactly 3 team cards', () => {
    render(<MemoryRouter><Team /></MemoryRouter>)
    expect(screen.getAllByTestId('team-card')).toHaveLength(3)
  })

  it('"The researchers" overline is present', () => {
    render(<MemoryRouter><Team /></MemoryRouter>)
    expect(screen.getByText('The researchers')).toBeInTheDocument()
  })

  it('each card has an avatar with an img', () => {
    render(<MemoryRouter><Team /></MemoryRouter>)
    const avatars = screen.getAllByTestId('team-avatar')
    expect(avatars).toHaveLength(3)
    avatars.forEach(av => expect(av.querySelector('img')).not.toBeNull())
  })

  it('each card contains the member name', () => {
    render(<MemoryRouter><Team /></MemoryRouter>)
    const cards = screen.getAllByTestId('team-card')
    cards.forEach(card => expect(card.textContent?.trim().length).toBeGreaterThan(0))
  })
})

describe('Phase 10 — JoinCTA', () => {
  it('renders without throwing', () => {
    expect(() => render(<MemoryRouter><JoinCTA /></MemoryRouter>)).not.toThrow()
  })

  it('section has data-testid="join-cta"', () => {
    render(<MemoryRouter><JoinCTA /></MemoryRouter>)
    expect(screen.getByTestId('join-cta')).toBeInTheDocument()
  })

  it('CTA button is present', () => {
    render(<MemoryRouter><JoinCTA /></MemoryRouter>)
    expect(screen.getByTestId('cta-button')).toBeInTheDocument()
  })

  it('CTA button text is "Get in touch →"', () => {
    render(<MemoryRouter><JoinCTA /></MemoryRouter>)
    expect(screen.getByTestId('cta-button').textContent).toBe('Get in touch →')
  })

  it('"Open science" overline is present', () => {
    render(<MemoryRouter><JoinCTA /></MemoryRouter>)
    expect(screen.getByText('Open science')).toBeInTheDocument()
  })

  it('CTA button navigates to collaborate', () => {
    render(<MemoryRouter><JoinCTA /></MemoryRouter>)
    const btn = screen.getByTestId('cta-button')
    expect(btn.tagName).toBe('BUTTON')
  })
})

describe('Phase 10 — Footer', () => {
  it('renders without throwing', () => {
    expect(() => render(<MemoryRouter><Footer /></MemoryRouter>)).not.toThrow()
  })

  it('footer has data-testid="footer"', () => {
    render(<MemoryRouter><Footer /></MemoryRouter>)
    expect(screen.getByTestId('footer')).toBeInTheDocument()
  })

  it('copyright element is present', () => {
    render(<MemoryRouter><Footer /></MemoryRouter>)
    expect(screen.getByTestId('copyright')).toBeInTheDocument()
  })

  it('copyright mentions "Engineers for Exploration"', () => {
    render(<MemoryRouter><Footer /></MemoryRouter>)
    expect(screen.getByTestId('copyright').textContent).toContain('Engineers for Exploration')
  })

  it('footer has 5 nav items (4 internal buttons + 1 external link)', () => {
    render(<MemoryRouter><Footer /></MemoryRouter>)
    const footer = screen.getByTestId('footer')
    const buttons = footer.querySelectorAll('nav button')
    const anchors = footer.querySelectorAll('nav a')
    expect(buttons.length + anchors.length).toBe(5)
  })

  it('"GitHub" link is present in footer nav', () => {
    render(<MemoryRouter><Footer /></MemoryRouter>)
    expect(screen.getByText('GitHub')).toBeInTheDocument()
  })
})

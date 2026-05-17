import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { Nav } from '../components/Nav'

// Decouple Nav rendering tests from scroll behavior
vi.mock('../hooks/useNavHide', () => ({
  useNavHide: vi.fn(() => ({ hidden: false, scrolled: false })),
}))

import { useNavHide } from '../hooks/useNavHide'

beforeEach(() => {
  vi.mocked(useNavHide).mockReturnValue({ hidden: false, scrolled: false })
})

// ── Rendering ─────────────────────────────────────────────────────────────

describe('Phase 2 — Nav rendering', () => {
  it('renders the wordmark "Mangrove Monitor"', () => {
    render(<Nav />)
    expect(screen.getByText('Mangrove Monitor')).toBeInTheDocument()
  })

  it('renders "· E4E Lab, UC San Diego"', () => {
    render(<Nav />)
    expect(screen.getByText('· E4E Lab, UC San Diego')).toBeInTheDocument()
  })

  it('renders exactly 3 center links', () => {
    render(<Nav />)
    const centerNav = screen.getByRole('navigation', { name: 'Site navigation' })
    expect(within(centerNav).getAllByRole('link')).toHaveLength(3)
  })

  it('center links are Research, Regions, Team in order', () => {
    render(<Nav />)
    const centerNav = screen.getByRole('navigation', { name: 'Site navigation' })
    const links = within(centerNav).getAllByRole('link')
    expect(links[0]).toHaveTextContent('Research')
    expect(links[1]).toHaveTextContent('Regions')
    expect(links[2]).toHaveTextContent('Team')
  })

  it('renders "Collaborate →" button', () => {
    render(<Nav />)
    expect(screen.getByRole('button', { name: /Collaborate/i })).toBeInTheDocument()
  })
})

// ── Structure ──────────────────────────────────────────────────────────────

describe('Phase 2 — Nav structure', () => {
  it('header has position: fixed', () => {
    render(<Nav />)
    expect(screen.getByRole('banner')).toHaveStyle({ position: 'fixed' })
  })

  it('header has z-index: 100', () => {
    render(<Nav />)
    expect(screen.getByRole('banner')).toHaveStyle({ zIndex: '100' })
  })
})

// ── Transform ─────────────────────────────────────────────────────────────

describe('Phase 2 — Nav transform', () => {
  it('sets data-hidden="false" when hook returns hidden=false', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: false, scrolled: false })
    render(<Nav />)
    expect(screen.getByRole('banner')).toHaveAttribute('data-hidden', 'false')
  })

  it('sets data-hidden="true" when hook returns hidden=true', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: true, scrolled: false })
    render(<Nav />)
    expect(screen.getByRole('banner')).toHaveAttribute('data-hidden', 'true')
  })

  it('applies translateY(-100%) when hidden=true', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: true, scrolled: false })
    render(<Nav />)
    expect(screen.getByRole('banner')).toHaveStyle({ transform: 'translateY(-100%)' })
  })

  it('applies translateY(0) when hidden=false', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: false, scrolled: false })
    render(<Nav />)
    expect(screen.getByRole('banner')).toHaveStyle({ transform: 'translateY(0)' })
  })
})

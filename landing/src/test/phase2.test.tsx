import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { Nav } from '../components/Nav'

// Decouple Nav rendering tests from scroll behavior
vi.mock('../hooks/useNavHide', () => ({
  useNavHide: vi.fn(() => ({ hidden: false, scrolled: false })),
}))

import { useNavHide } from '../hooks/useNavHide'

const renderNav = () => render(<MemoryRouter><Nav /></MemoryRouter>)

beforeEach(() => {
  vi.mocked(useNavHide).mockReturnValue({ hidden: false, scrolled: false })
})

// ── Rendering ─────────────────────────────────────────────────────────────

describe('Phase 2 — Nav rendering', () => {
  it('renders the wordmark "Mangrove Monitor"', () => {
    renderNav()
    expect(screen.getByText('Mangrove Monitor')).toBeInTheDocument()
  })

  it('renders "· E4E Lab, UC San Diego"', () => {
    renderNav()
    expect(screen.getByText('· E4E Lab, UC San Diego')).toBeInTheDocument()
  })

  it('renders exactly 3 center nav buttons', () => {
    renderNav()
    const centerNav = screen.getByRole('navigation', { name: 'Site navigation' })
    expect(within(centerNav).getAllByRole('button')).toHaveLength(3)
  })

  it('center nav buttons are Visualizer, Blog, Team in order', () => {
    renderNav()
    const centerNav = screen.getByRole('navigation', { name: 'Site navigation' })
    const btns = within(centerNav).getAllByRole('button')
    expect(btns[0]).toHaveTextContent('Visualizer')
    expect(btns[1]).toHaveTextContent('Blog')
    expect(btns[2]).toHaveTextContent('Team')
  })

  it('renders "Collaborate →" button', () => {
    renderNav()
    expect(screen.getByRole('button', { name: /Collaborate/i })).toBeInTheDocument()
  })
})

// ── Structure ──────────────────────────────────────────────────────────────

describe('Phase 2 — Nav structure', () => {
  it('header has position: fixed', () => {
    renderNav()
    expect(screen.getByRole('banner')).toHaveStyle({ position: 'fixed' })
  })

  it('header has z-index: 100', () => {
    renderNav()
    expect(screen.getByRole('banner')).toHaveStyle({ zIndex: '100' })
  })
})

// ── Transform ─────────────────────────────────────────────────────────────

describe('Phase 2 — Nav transform', () => {
  it('sets data-hidden="false" when hook returns hidden=false', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: false, scrolled: false })
    renderNav()
    expect(screen.getByRole('banner')).toHaveAttribute('data-hidden', 'false')
  })

  it('sets data-hidden="true" when hook returns hidden=true', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: true, scrolled: false })
    renderNav()
    expect(screen.getByRole('banner')).toHaveAttribute('data-hidden', 'true')
  })

  it('applies translateY(-100%) when hidden=true', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: true, scrolled: false })
    renderNav()
    expect(screen.getByRole('banner')).toHaveStyle({ transform: 'translateY(-100%)' })
  })

  it('applies translateY(0) when hidden=false', () => {
    vi.mocked(useNavHide).mockReturnValue({ hidden: false, scrolled: false })
    renderNav()
    expect(screen.getByRole('banner')).toHaveStyle({ transform: 'translateY(0)' })
  })
})

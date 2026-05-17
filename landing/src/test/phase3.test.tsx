import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('../components/GlobeViewer', () => ({
  GlobeViewer: vi.fn(() => null),
}))

import { Hero } from '../components/Hero'

describe('Phase 3 — Hero section', () => {
  it('renders the hero section', () => {
    render(<Hero />)
    expect(screen.getByTestId('hero')).toBeInTheDocument()
  })

  it('hero has height: 100vh', () => {
    render(<Hero />)
    // Use raw inline style — happy-dom resolves viewport units in getComputedStyle
    expect(screen.getByTestId('hero').style.height).toBe('100vh')
  })

  it('hero has backgroundColor: #f0ede6', () => {
    render(<Hero />)
    expect(screen.getByTestId('hero')).toHaveStyle({ backgroundColor: '#f0ede6' })
  })

  it('overline text is "Engineers for Exploration · UC San Diego"', () => {
    render(<Hero />)
    expect(screen.getByText('Engineers for Exploration · UC San Diego')).toBeInTheDocument()
  })

  it('h1 contains "A living map of Earth\'s"', () => {
    render(<Hero />)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toContain("A living map of Earth's")
  })

  it('<em> inside h1 contains "mangroves"', () => {
    render(<Hero />)
    const h1 = screen.getByRole('heading', { level: 1 })
    const em = h1.querySelector('em')
    expect(em).toBeInTheDocument()
    expect(em).toHaveTextContent('mangroves')
  })

  it('<em> has class serif-italic', () => {
    render(<Hero />)
    const em = screen.getByRole('heading', { level: 1 }).querySelector('em')
    expect(em).toHaveClass('serif-italic')
  })

  it('subhead paragraph is present', () => {
    render(<Hero />)
    expect(screen.getByText(/Drag the globe/i)).toBeInTheDocument()
  })

  it('globe root div exists', () => {
    render(<Hero />)
    expect(document.getElementById('globe-root')).toBeInTheDocument()
  })

  it('gradient overlay exists', () => {
    render(<Hero />)
    const overlay = screen.getByTestId('gradient-overlay')
    expect(overlay).toBeInTheDocument()
    // happy-dom silently drops gradient values from CSSOM; verify structural props only.
    // Gradient value is verified in E2E (real browser).
    expect(overlay.style.position).toBe('absolute')
    expect(overlay.style.height).toBe('65%')
  })

  it('scroll cue is present', () => {
    render(<Hero />)
    expect(screen.getByTestId('scroll-cue')).toBeInTheDocument()
    expect(screen.getByText('scroll')).toBeInTheDocument()
  })
})

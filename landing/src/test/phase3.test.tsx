import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('../components/GlobeViewer', () => ({
  GlobeViewer: vi.fn(() => null),
}))

vi.mock('../data/regions', () => ({
  regions: [
    { id: 'florida', name: 'Florida', status: 'trained', tilesUrl: '', pmtilesUrl: '', center: [0,0], zoom: 5 },
    { id: 'yucatan', name: 'Yucatan', status: 'planned', tilesUrl: '', pmtilesUrl: '', center: [0,0], zoom: 5 },
  ],
}))

import { Hero } from '../components/Hero'

describe('Phase 3 — Hero section', () => {
  it('renders the hero section', () => {
    render(<Hero />)
    expect(screen.getByTestId('hero')).toBeInTheDocument()
  })

  it('hero has height: 100vh', () => {
    render(<Hero />)
    expect(screen.getByTestId('hero').style.height).toBe('100vh')
  })

  it('hero has dark background', () => {
    render(<Hero />)
    expect(screen.getByTestId('hero')).toHaveStyle({ backgroundColor: '#0a1a0c' })
  })

  it('overlay label "Global visualizer" is present', () => {
    render(<Hero />)
    expect(screen.getByText('Global visualizer')).toBeInTheDocument()
  })

  it('h1 contains "Select a region"', () => {
    render(<Hero />)
    expect(screen.getByRole('heading', { level: 1 }).textContent).toContain('Select a region')
  })

  it('globe root div exists', () => {
    render(<Hero />)
    expect(document.getElementById('globe-root')).toBeInTheDocument()
  })

  it('region pills container is present', () => {
    render(<Hero />)
    expect(screen.getByTestId('region-pills')).toBeInTheDocument()
  })

  it('renders a pill for each mocked region', () => {
    render(<Hero />)
    expect(screen.getByText('Florida')).toBeInTheDocument()
    expect(screen.getByText('Yucatan')).toBeInTheDocument()
  })

  it('clicking a trained region pill calls onRegionClick', () => {
    const onRegionClick = vi.fn()
    render(<Hero onRegionClick={onRegionClick} />)
    screen.getByText('Florida').click()
    expect(onRegionClick).toHaveBeenCalledWith('florida')
  })
})

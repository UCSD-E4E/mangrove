import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

vi.mock('../components/GlobeViewer', () => ({ GlobeViewer: vi.fn(() => null) }))
vi.mock('gsap', () => ({ default: { registerPlugin: vi.fn(), context: vi.fn(() => ({ revert: vi.fn() })) } }))
vi.mock('gsap/ScrollTrigger', () => ({ ScrollTrigger: { create: vi.fn(() => ({ kill: vi.fn() })) } }))

import { RegionHUD } from '../components/RegionHUD'
import { TerrainLegend } from '../components/TerrainLegend'
import { RegionStrip } from '../components/RegionStrip'
import { TerrainControls } from '../components/TerrainControls'
import { regions } from '../data/regions'

const floridaRegion     = regions.find((r) => r.id === 'florida')!
const madagascarRegion  = regions.find((r) => r.id === 'madagascar_mozambique')!

// ─── RegionHUD ────────────────────────────────────────────────────────────────

describe('RegionHUD', () => {
  it('renders region displayName', () => {
    render(<RegionHUD region={floridaRegion} />)
    expect(screen.getByText(floridaRegion.displayName)).toBeInTheDocument()
  })

  it('renders lat/lon coordinates', () => {
    render(<RegionHUD region={floridaRegion} />)
    const coords = screen.getByTestId('region-coords')
    expect(coords.textContent).toContain('25.20°N')
    expect(coords.textContent).toContain('80.90°W')
  })

  it('updates when active region prop changes', () => {
    const { rerender } = render(<RegionHUD region={floridaRegion} />)
    expect(screen.getByText(floridaRegion.displayName)).toBeInTheDocument()
    rerender(<RegionHUD region={madagascarRegion} />)
    expect(screen.getByText(madagascarRegion.displayName)).toBeInTheDocument()
  })

  it('formats lat to 2 decimal places with N/S suffix', () => {
    render(<RegionHUD region={madagascarRegion} />)       // lat = -19.5
    expect(screen.getByTestId('region-coords').textContent).toContain('19.50°S')
  })

  it('formats lng to 2 decimal places with E/W suffix', () => {
    render(<RegionHUD region={madagascarRegion} />)       // lng = 44.5
    expect(screen.getByTestId('region-coords').textContent).toContain('44.50°E')
  })
})

// ─── TerrainLegend ────────────────────────────────────────────────────────────

describe('TerrainLegend', () => {
  it('renders exactly 5 rows', () => {
    render(<TerrainLegend />)
    expect(screen.getAllByTestId('legend-row')).toHaveLength(5)
  })

  it('row labels include Mangrove, Tree Cover, Water, Wetland, Built-up', () => {
    render(<TerrainLegend />)
    expect(screen.getByText('Mangrove')).toBeInTheDocument()
    expect(screen.getByText('Tree Cover')).toBeInTheDocument()
    expect(screen.getByText('Water')).toBeInTheDocument()
    expect(screen.getByText('Wetland')).toBeInTheDocument()
    expect(screen.getByText('Built-up')).toBeInTheDocument()
  })

  it('Mangrove dot color is #1e5032', () => {
    render(<TerrainLegend />)
    const mangroveRow = screen.getByText('Mangrove').closest('[data-testid="legend-row"]') as HTMLElement
    const dot = within(mangroveRow).getByTestId('legend-dot')
    expect(dot).toHaveStyle({ backgroundColor: '#1e5032' })
  })

  it('Water dot color is #4a90b8', () => {
    render(<TerrainLegend />)
    const waterRow = screen.getByText('Water').closest('[data-testid="legend-row"]') as HTMLElement
    const dot = within(waterRow).getByTestId('legend-dot')
    expect(dot).toHaveStyle({ backgroundColor: '#4a90b8' })
  })
})

// ─── RegionStrip ──────────────────────────────────────────────────────────────

describe('RegionStrip', () => {
  it('renders exactly 6 pills', () => {
    render(<RegionStrip activeRegionId="florida" onSelectRegion={vi.fn()} />)
    expect(screen.getAllByTestId('region-pill')).toHaveLength(6)
  })

  it('active pill has data-active="true"', () => {
    render(<RegionStrip activeRegionId="florida" onSelectRegion={vi.fn()} />)
    const pills = screen.getAllByTestId('region-pill')
    const active = pills.filter((p) => p.getAttribute('data-active') === 'true')
    expect(active).toHaveLength(1)
    expect(active[0]).toHaveTextContent('Florida')
  })

  it('trained-region pills have a green status dot (#3d6b4a)', () => {
    render(<RegionStrip activeRegionId="florida" onSelectRegion={vi.fn()} />)
    const pills = screen.getAllByTestId('region-pill')
    const trainedRegions = regions.filter((r) => r.status === 'trained')
    const dot = within(pills[0]).getByTestId('status-dot')
    expect(dot).toHaveStyle({ backgroundColor: '#3d6b4a' })
    expect(trainedRegions).toHaveLength(pills.length)
  })

  it('clicking a pill calls onSelectRegion with the correct id', async () => {
    const onSelect = vi.fn()
    render(<RegionStrip activeRegionId="florida" onSelectRegion={onSelect} />)
    const pills = screen.getAllByTestId('region-pill')
    await userEvent.click(pills[1]) // second region = brazil
    expect(onSelect).toHaveBeenCalledWith(regions[1].id)
  })
})

// ─── TerrainControls ──────────────────────────────────────────────────────────

describe('TerrainControls', () => {
  let onZoomIn: ReturnType<typeof vi.fn>
  let onZoomOut: ReturnType<typeof vi.fn>
  let onReset: ReturnType<typeof vi.fn>

  beforeEach(() => {
    onZoomIn = vi.fn()
    onZoomOut = vi.fn()
    onReset = vi.fn()
  })

  it('zoom+ button has aria-label="Zoom in"', () => {
    render(<TerrainControls onZoomIn={onZoomIn} onZoomOut={onZoomOut} onReset={onReset} />)
    expect(screen.getByRole('button', { name: 'Zoom in' })).toBeInTheDocument()
  })

  it('zoom- button has aria-label="Zoom out"', () => {
    render(<TerrainControls onZoomIn={onZoomIn} onZoomOut={onZoomOut} onReset={onReset} />)
    expect(screen.getByRole('button', { name: 'Zoom out' })).toBeInTheDocument()
  })

  it('reset button has aria-label="Reset view"', () => {
    render(<TerrainControls onZoomIn={onZoomIn} onZoomOut={onZoomOut} onReset={onReset} />)
    expect(screen.getByRole('button', { name: 'Reset view' })).toBeInTheDocument()
  })

  it('clicking zoom+ calls onZoomIn', async () => {
    render(<TerrainControls onZoomIn={onZoomIn} onZoomOut={onZoomOut} onReset={onReset} />)
    await userEvent.click(screen.getByRole('button', { name: 'Zoom in' }))
    expect(onZoomIn).toHaveBeenCalledOnce()
  })

  it('clicking zoom- calls onZoomOut', async () => {
    render(<TerrainControls onZoomIn={onZoomIn} onZoomOut={onZoomOut} onReset={onReset} />)
    await userEvent.click(screen.getByRole('button', { name: 'Zoom out' }))
    expect(onZoomOut).toHaveBeenCalledOnce()
  })

  it('clicking reset calls onReset', async () => {
    render(<TerrainControls onZoomIn={onZoomIn} onZoomOut={onZoomOut} onReset={onReset} />)
    await userEvent.click(screen.getByRole('button', { name: 'Reset view' }))
    expect(onReset).toHaveBeenCalledOnce()
  })
})

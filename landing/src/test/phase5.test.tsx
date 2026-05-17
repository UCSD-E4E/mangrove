import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, act } from '@testing-library/react'
import { createRef } from 'react'

vi.mock('@deck.gl/react', () => ({
  default: vi.fn(() => null),
}))
vi.mock('@deck.gl/core', () => ({
  MapView: vi.fn(() => ({})),
  FlyToInterpolator: vi.fn(() => ({})),
}))
vi.mock('@deck.gl/layers', () => ({
  ScatterplotLayer: vi.fn((props: Record<string, unknown>) => ({ _type: 'ScatterplotLayer', ...props })),
  BitmapLayer: vi.fn((props: Record<string, unknown>) => ({ _type: 'BitmapLayer', ...props })),
  TextLayer: vi.fn((props: Record<string, unknown>) => ({ _type: 'TextLayer', ...props })),
}))
vi.mock('@deck.gl/geo-layers', () => ({
  TerrainLayer: vi.fn((props: Record<string, unknown>) => ({ _type: 'TerrainLayer', ...props })),
  TileLayer: vi.fn((props: Record<string, unknown>) => ({ _type: 'TileLayer', ...props })),
}))
vi.mock('maplibre-gl', () => ({ default: {}, Map: vi.fn() }))

import DeckGL from '@deck.gl/react'
import { ScatterplotLayer } from '@deck.gl/layers'
import { GlobeViewer } from '../components/GlobeViewer'
import type { GlobeViewerHandle } from '../components/GlobeViewer'
import { regions } from '../data/regions'

const MockDeckGL = vi.mocked(DeckGL as unknown as (...args: unknown[]) => null)
const MockScatterplotLayer = vi.mocked(ScatterplotLayer as unknown as (props: unknown) => unknown)

function getLastDeckProps() {
  const calls = MockDeckGL.mock.calls
  return calls[calls.length - 1]?.[0] as Record<string, unknown> | undefined
}

function getLastScatterLayer() {
  const layers = (getLastDeckProps()?.layers as unknown[]) ?? []
  // Find the primary region dot layer specifically (not the glow layer)
  return layers.find(
    (l: unknown) => {
      const layer = l as Record<string, unknown>
      return layer?._type === 'ScatterplotLayer' && layer?.id === 'regions'
    },
  ) as Record<string, unknown> | undefined
}

describe('Phase 5 — GlobeViewer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Suppress the RAF rotation loop so async act() can settle between tests.
    // The loop registers callbacks but they're never fired unless a test overrides this.
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('mounts without throwing', () => {
    expect(() => render(<GlobeViewer />)).not.toThrow()
  })

  it('flyToRegion ref method is defined and callable', () => {
    const ref = createRef<GlobeViewerHandle>()
    render(<GlobeViewer ref={ref} />)
    expect(typeof ref.current?.flyToRegion).toBe('function')
  })

  it('flyToRegion("florida") updates viewState to Florida coords', async () => {
    const ref = createRef<GlobeViewerHandle>()
    render(<GlobeViewer ref={ref} />)
    await act(async () => { ref.current!.flyToRegion('florida') })
    const vs = getLastDeckProps()?.viewState as Record<string, number> | undefined
    expect(vs?.longitude).toBeCloseTo(-80.9, 1)
    expect(vs?.latitude).toBeCloseTo(25.2, 1)
    expect(vs?.pitch).toBe(52)
  })

  it('flyToRegion with unknown id warns and does not throw', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const ref = createRef<GlobeViewerHandle>()
    render(<GlobeViewer ref={ref} />)
    await act(async () => { ref.current!.flyToRegion('nonexistent') })
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('nonexistent'))
  })

  it('resetView() resets to globe view (pitch 0, low zoom)', async () => {
    const ref = createRef<GlobeViewerHandle>()
    render(<GlobeViewer ref={ref} />)
    await act(async () => { ref.current!.flyToRegion('everglades') })
    await act(async () => { ref.current!.resetView() })
    const vs = getLastDeckProps()?.viewState as Record<string, number> | undefined
    expect(vs?.pitch).toBe(0)
    expect(vs?.zoom).toBeLessThan(3)
  })

  it('auto-rotation increments longitude by ~0.025 per RAF call', async () => {
    // Override the suppressed RAF mock to capture and fire callbacks
    const rafCallbacks: FrameRequestCallback[] = []
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallbacks.push(cb)
      return rafCallbacks.length
    })

    render(<GlobeViewer />)
    const before = (getLastDeckProps()?.viewState as Record<string, number> | undefined)?.longitude ?? 0

    // Fire one callback; the rotation sets state then schedules the next one
    await act(async () => { rafCallbacks[0]?.(performance.now()) })

    const after = (getLastDeckProps()?.viewState as Record<string, number> | undefined)?.longitude ?? 0
    expect(after - before).toBeCloseTo(0.025, 3)
  })

  it('renders 6 ScatterplotLayer data points (one per region)', () => {
    render(<GlobeViewer />)
    const layer = getLastScatterLayer()
    expect((layer?.data as unknown[])?.length).toBe(6)
  })

  it('ScatterplotLayer is constructed with regions data', () => {
    render(<GlobeViewer />)
    expect(MockScatterplotLayer).toHaveBeenCalled()
    const callArgs = MockScatterplotLayer.mock.calls[0]?.[0] as Record<string, unknown>
    expect(callArgs?.data).toBe(regions)
  })

  it('trained regions getFillColor returns #3d6b4a ([61,107,74])', () => {
    render(<GlobeViewer />)
    const layer = getLastScatterLayer()
    const getColor = layer?.getFillColor as (d: (typeof regions)[0]) => [number, number, number]
    const trained = regions.find((r) => r.status === 'trained')!
    expect(getColor(trained)).toEqual([61, 107, 74])
  })

  it('testing regions getFillColor returns #aaa ([170,170,170])', () => {
    render(<GlobeViewer />)
    const layer = getLastScatterLayer()
    const getColor = layer?.getFillColor as (d: (typeof regions)[0]) => [number, number, number]
    const mockTesting = { ...regions[0], status: 'testing' as const }
    expect(getColor(mockTesting)).toEqual([170, 170, 170])
  })
})

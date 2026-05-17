import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'
import { renderHook } from '@testing-library/react'

// Mock GlobeViewer so About tests don't pull in deck.gl
vi.mock('../components/GlobeViewer', () => ({ GlobeViewer: vi.fn(() => null) }))

// Mock useCountUp to return target — tests component structure, not animation
vi.mock('../hooks/useCountUp', () => ({
  useCountUp: vi.fn(({ target }: { target: number }) => target),
}))

import { About } from '../components/About'
import { useCountUp } from '../hooks/useCountUp'

// Stub IntersectionObserver (not available in happy-dom)
const mockObserve = vi.fn()
const mockDisconnect = vi.fn()

describe('Phase 6 — About section', () => {
  beforeEach(() => {
    vi.stubGlobal('IntersectionObserver', vi.fn(() => ({
      observe: mockObserve,
      disconnect: mockDisconnect,
      unobserve: vi.fn(),
    })))
    mockObserve.mockClear()
    mockDisconnect.mockClear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders overline text', () => {
    render(<About />)
    expect(screen.getByText('The Science')).toBeInTheDocument()
  })

  it('H2 contains <em> with "track them"', () => {
    render(<About />)
    const h2 = screen.getByRole('heading', { level: 2 })
    const em = h2.querySelector('em')
    expect(em).toBeInTheDocument()
    expect(em).toHaveTextContent('track them')
  })

  it('renders exactly 4 stat cells', () => {
    render(<About />)
    expect(screen.getAllByTestId('stat-cell')).toHaveLength(4)
  })

  it('stat cells display formatted target values', () => {
    render(<About />)
    expect(screen.getByText(/147k km²/)).toBeInTheDocument()
    expect(screen.getByText(/4×/)).toBeInTheDocument()
    expect(screen.getByText(/0\.35m/)).toBeInTheDocument()
    expect(screen.getByText(/6 regions/)).toBeInTheDocument()
  })
})

// ─── useCountUp hook tests ─────────────────────────────────────────────────

describe('useCountUp', () => {
  // Use the real hook (not the module-level mock above which only affects About)
  // vi.mock is module-scoped; import the real implementation by un-mocking
  // We test the hook in a separate describe and import it directly from the module
  // Note: vi.mock above mocks the *consumer*; the hook's own module is still real here
  // because we're testing via renderHook which imports the hook directly.
  // Actually vi.mock affects ALL imports of that path in this file.
  // So we need a separate test file for the real hook.
  it('placeholder — real hook tests are in phase6-hook.test.ts', () => {
    // useCountUp is mocked to return target in this file
    const { result } = renderHook(() =>
      useCountUp({ target: 50, duration: 1000, isInView: false }),
    )
    // When mocked, it returns target regardless of isInView
    expect(result.current).toBe(50)
  })
})

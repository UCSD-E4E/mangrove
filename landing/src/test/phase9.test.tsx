import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'

import { Pipeline } from '../components/Pipeline'

vi.mock('gsap', () => ({
  default: {
    registerPlugin: vi.fn(),
    context: vi.fn(() => ({ revert: vi.fn() })),
    delayedCall: vi.fn(),
  },
}))

vi.mock('gsap/ScrollTrigger', () => ({
  ScrollTrigger: {
    create: vi.fn(),
    getAll: vi.fn(() => []),
    refresh: vi.fn(),
  },
}))

describe('Phase 9 — Pipeline', () => {
  beforeEach(() => {
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders without throwing', () => {
    expect(() => render(<Pipeline />)).not.toThrow()
  })

  it('section has data-testid="pipeline"', () => {
    render(<Pipeline />)
    expect(screen.getByTestId('pipeline')).toBeInTheDocument()
  })

  it('renders exactly 4 pipeline nodes', () => {
    render(<Pipeline />)
    expect(screen.getAllByTestId('pipeline-node')).toHaveLength(4)
  })

  it('renders exactly 3 pipeline connectors', () => {
    render(<Pipeline />)
    expect(screen.getAllByTestId('pipeline-connector')).toHaveLength(3)
  })

  it('"Sentinel-2 Input" label is present', () => {
    render(<Pipeline />)
    expect(screen.getByText('Sentinel-2 Input')).toBeInTheDocument()
  })

  it('"Contrastive Encoder" label is present', () => {
    render(<Pipeline />)
    expect(screen.getByText('Contrastive Encoder')).toBeInTheDocument()
  })

  it('"Segmentation Head" label is present', () => {
    render(<Pipeline />)
    expect(screen.getByText('Segmentation Head')).toBeInTheDocument()
  })

  it('"Super-Resolution" label is present', () => {
    render(<Pipeline />)
    expect(screen.getByText('Super-Resolution')).toBeInTheDocument()
  })

  it('all nodes start inactive (data-active="false")', () => {
    render(<Pipeline />)
    const nodes = screen.getAllByTestId('pipeline-node')
    for (const node of nodes) {
      expect(node.getAttribute('data-active')).toBe('false')
    }
  })

  it('"How it works" overline is present', () => {
    render(<Pipeline />)
    expect(screen.getByText('How it works')).toBeInTheDocument()
  })

  it('nodes have correct data-step-id attributes in order', () => {
    render(<Pipeline />)
    const nodes = screen.getAllByTestId('pipeline-node')
    const ids = nodes.map(n => n.getAttribute('data-step-id'))
    expect(ids).toEqual(['sentinel', 'encoder', 'segmentation', 'superres'])
  })
})

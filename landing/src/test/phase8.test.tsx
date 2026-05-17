import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'

import { SplitCompare } from '../components/SplitCompare'

// Helper: mock the viewer's bounding rect so drag math works
function mockViewerRect(viewer: HTMLElement, width = 1000) {
  vi.spyOn(viewer, 'getBoundingClientRect').mockReturnValue({
    left: 0, top: 0, right: width, bottom: 500,
    width, height: 500, x: 0, y: 0,
    toJSON: () => ({}),
  } as DOMRect)
}

function simulateDrag(handle: HTMLElement, _viewer: HTMLElement, clientX: number) {
  // startDrag attaches document-level listeners; fire move/up on document
  fireEvent.pointerDown(handle)
  fireEvent.pointerMove(document, { clientX })
  fireEvent.pointerUp(document)
}

describe('Phase 8 — SplitCompare', () => {
  beforeEach(() => {
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders without throwing', () => {
    expect(() => render(<SplitCompare />)).not.toThrow()
  })

  it('initial divider position is 42%', () => {
    render(<SplitCompare />)
    expect(screen.getByTestId('divider').style.left).toBe('42%')
  })

  it('both images are in the DOM', () => {
    render(<SplitCompare />)
    expect(screen.getByTestId('image-left')).toBeInTheDocument()
    expect(screen.getByTestId('image-right')).toBeInTheDocument()
  })

  it('"10m input" label is present', () => {
    render(<SplitCompare />)
    expect(screen.getByText('10m input')).toBeInTheDocument()
  })

  it('"0.35m SR output" label is present', () => {
    render(<SplitCompare />)
    expect(screen.getByText('0.35m SR output')).toBeInTheDocument()
  })

  it('divider handle has pointer event handlers (fires without error)', () => {
    render(<SplitCompare />)
    expect(() => fireEvent.pointerDown(screen.getByTestId('divider-handle'))).not.toThrow()
  })

  it('dragging handle to 10% sets divider to 10%', () => {
    render(<SplitCompare />)
    const viewer = screen.getByTestId('viewer')
    mockViewerRect(viewer, 1000)
    simulateDrag(screen.getByTestId('divider-handle'), viewer, 100) // 100/1000 = 10%
    expect(screen.getByTestId('divider').style.left).toBe('10%')
  })

  it('dragging handle to 90% sets divider to 90%', () => {
    render(<SplitCompare />)
    const viewer = screen.getByTestId('viewer')
    mockViewerRect(viewer, 1000)
    simulateDrag(screen.getByTestId('divider-handle'), viewer, 900) // 900/1000 = 90%
    expect(screen.getByTestId('divider').style.left).toBe('90%')
  })

  it('dragging to 0% clamps to 5%', () => {
    render(<SplitCompare />)
    const viewer = screen.getByTestId('viewer')
    mockViewerRect(viewer, 1000)
    simulateDrag(screen.getByTestId('divider-handle'), viewer, 0)
    expect(screen.getByTestId('divider').style.left).toBe('5%')
  })

  it('dragging to 100% clamps to 95%', () => {
    render(<SplitCompare />)
    const viewer = screen.getByTestId('viewer')
    mockViewerRect(viewer, 1000)
    simulateDrag(screen.getByTestId('divider-handle'), viewer, 1000)
    expect(screen.getByTestId('divider').style.left).toBe('95%')
  })

  it('metadata row has exactly 3 cells', () => {
    render(<SplitCompare />)
    expect(screen.getAllByTestId('metadata-cell')).toHaveLength(3)
  })
})

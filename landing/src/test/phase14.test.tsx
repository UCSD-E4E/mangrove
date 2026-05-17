import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'

import { ErrorBoundary } from '../components/ErrorBoundary'
import { SplitCompare } from '../components/SplitCompare'

// ─── ErrorBoundary ────────────────────────────────────────────────────────────

function Bomb(): never {
  throw new Error('test explosion')
}

describe('Phase 14 — ErrorBoundary', () => {
  beforeEach(() => {
    // Suppress React's internal error logging for expected throws
    vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders children when there is no error', () => {
    render(
      <ErrorBoundary fallback={<p>fallback</p>}>
        <p>content</p>
      </ErrorBoundary>,
    )
    expect(screen.getByText('content')).toBeInTheDocument()
  })

  it('does not render the fallback when there is no error', () => {
    render(
      <ErrorBoundary fallback={<p>fallback</p>}>
        <p>content</p>
      </ErrorBoundary>,
    )
    expect(screen.queryByText('fallback')).not.toBeInTheDocument()
  })

  it('shows fallback when a child throws', () => {
    render(
      <ErrorBoundary fallback={<p>Error occurred</p>}>
        <Bomb />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Error occurred')).toBeInTheDocument()
  })

  it('does not render children after an error', () => {
    render(
      <ErrorBoundary fallback={<p>Error occurred</p>}>
        <Bomb />
      </ErrorBoundary>,
    )
    expect(screen.queryByText('content')).not.toBeInTheDocument()
  })

  it('getDerivedStateFromError returns hasError: true', () => {
    const state = ErrorBoundary.getDerivedStateFromError(new Error('test'))
    expect(state).toEqual({ hasError: true })
  })
})

// ─── SplitCompare loading="lazy" ─────────────────────────────────────────────

describe('Phase 14 — SplitCompare img loading', () => {
  beforeEach(() => {
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('image-left has loading="lazy"', () => {
    render(<SplitCompare leftSrc="http://ex.com/l.jpg" rightSrc="http://ex.com/r.jpg" />)
    expect(screen.getByTestId('image-left').getAttribute('loading')).toBe('lazy')
  })

  it('image-right has loading="lazy"', () => {
    render(<SplitCompare leftSrc="http://ex.com/l.jpg" rightSrc="http://ex.com/r.jpg" />)
    expect(screen.getByTestId('image-right').getAttribute('loading')).toBe('lazy')
  })
})

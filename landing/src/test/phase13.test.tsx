import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'

import { SplitCompare } from '../components/SplitCompare'

// ─── SplitCompare image prop wiring ──────────────────────────────────────────

describe('Phase 13 — SplitCompare image props', () => {
  beforeEach(() => {
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(0)
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders img elements when leftSrc and rightSrc are provided', () => {
    render(<SplitCompare leftSrc="http://ex.com/left.jpg" rightSrc="http://ex.com/right.jpg" />)
    expect(screen.getByTestId('image-left').tagName).toBe('IMG')
    expect(screen.getByTestId('image-right').tagName).toBe('IMG')
  })

  it('falls back to div when no src props are passed', () => {
    render(<SplitCompare />)
    expect(screen.getByTestId('image-left').tagName).toBe('DIV')
    expect(screen.getByTestId('image-right').tagName).toBe('DIV')
  })

  it('img-left has the correct src attribute', () => {
    render(<SplitCompare leftSrc="http://ex.com/left.jpg" rightSrc="http://ex.com/right.jpg" />)
    expect(screen.getByTestId('image-left').getAttribute('src')).toBe('http://ex.com/left.jpg')
  })

  it('img-right has the correct src attribute', () => {
    render(<SplitCompare leftSrc="http://ex.com/left.jpg" rightSrc="http://ex.com/right.jpg" />)
    expect(screen.getByTestId('image-right').getAttribute('src')).toBe('http://ex.com/right.jpg')
  })

  it('drag still works when images are provided', () => {
    render(<SplitCompare leftSrc="http://ex.com/left.jpg" rightSrc="http://ex.com/right.jpg" />)
    expect(screen.getByTestId('divider').style.left).toBe('42%')
  })
})

// ─── classConfig ─────────────────────────────────────────────────────────────

describe('Phase 13 — classConfig', () => {
  it('CLASS_NAMES covers all 6 class indices', async () => {
    const { CLASS_NAMES } = await import('../lib/classConfig')
    expect(Object.keys(CLASS_NAMES)).toHaveLength(6)
    expect(CLASS_NAMES[5]).toBe('Mangrove')
  })

  it('CLASS_COLORS covers all 6 class indices and returns RGBA arrays', async () => {
    const { CLASS_COLORS } = await import('../lib/classConfig')
    for (let i = 0; i <= 5; i++) {
      expect(CLASS_COLORS[i]).toHaveLength(4)
    }
  })

  it('LEGEND_ITEMS contains Mangrove', async () => {
    const { LEGEND_ITEMS } = await import('../lib/classConfig')
    expect(LEGEND_ITEMS.some((item) => item.label === 'Mangrove')).toBe(true)
  })
})

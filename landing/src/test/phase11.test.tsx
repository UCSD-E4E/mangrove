import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook } from '@testing-library/react'

// vi.hoisted() runs before static imports, so these refs are safe in the factory.
const { mockFrom, mockRevert, mockContext } = vi.hoisted(() => {
  const mockRevert = vi.fn()
  const mockFrom = vi.fn()
  const mockContext = vi.fn((fn: unknown) => {
    if (typeof fn === 'function') (fn as () => void)()
    return { revert: mockRevert }
  })
  return { mockFrom, mockRevert, mockContext }
})

vi.mock('gsap', () => ({
  default: {
    registerPlugin: vi.fn(),
    context: mockContext,
    from: mockFrom,
  },
}))

vi.mock('gsap/ScrollTrigger', () => ({
  ScrollTrigger: { create: vi.fn(), getAll: vi.fn(() => []) },
}))

import { useScrollAnimations } from '../hooks/useScrollAnimations'

describe('Phase 11 — useScrollAnimations', () => {
  beforeEach(() => {
    mockFrom.mockClear()
    mockContext.mockClear()
    mockRevert.mockClear()
  })

  it('runs without throwing', () => {
    expect(() => renderHook(() => useScrollAnimations())).not.toThrow()
  })

  it('calls gsap.context once on mount', () => {
    renderHook(() => useScrollAnimations())
    expect(mockContext).toHaveBeenCalledTimes(1)
  })

  it('registers exactly 3 gsap.from animations', () => {
    renderHook(() => useScrollAnimations())
    expect(mockFrom).toHaveBeenCalledTimes(3)
  })

  it('animates [data-testid="split-compare"]', () => {
    renderHook(() => useScrollAnimations())
    const targets = mockFrom.mock.calls.map((c: unknown[]) => c[0])
    expect(targets).toContain('[data-testid="split-compare"]')
  })

  it('animates [data-testid="team-card"]', () => {
    renderHook(() => useScrollAnimations())
    const targets = mockFrom.mock.calls.map((c: unknown[]) => c[0])
    expect(targets).toContain('[data-testid="team-card"]')
  })

  it('animates [data-testid="join-cta"]', () => {
    renderHook(() => useScrollAnimations())
    const targets = mockFrom.mock.calls.map((c: unknown[]) => c[0])
    expect(targets).toContain('[data-testid="join-cta"]')
  })

  it('team-card animation uses a numeric stagger', () => {
    renderHook(() => useScrollAnimations())
    const teamCall = mockFrom.mock.calls.find((c: unknown[]) => c[0] === '[data-testid="team-card"]')
    expect(typeof (teamCall?.[1] as Record<string, unknown>)?.stagger).toBe('number')
  })

  it('calls ctx.revert on unmount', () => {
    const { unmount } = renderHook(() => useScrollAnimations())
    unmount()
    expect(mockRevert).toHaveBeenCalled()
  })
})

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useCountUp } from '../hooks/useCountUp'

describe('useCountUp hook', () => {
  let rafCallbacks: FrameRequestCallback[]

  beforeEach(() => {
    rafCallbacks = []
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallbacks.push(cb)
      return rafCallbacks.length
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns 0 when isInView=false', () => {
    const { result } = renderHook(() =>
      useCountUp({ target: 100, duration: 1000, isInView: false }),
    )
    expect(result.current).toBe(0)
    expect(rafCallbacks).toHaveLength(0) // RAF never registered
  })

  it('returns target value after duration ms when isInView=true', async () => {
    const { result } = renderHook(() =>
      useCountUp({ target: 100, duration: 1000, isInView: true }),
    )
    // First RAF fires to set startTime (t=0), then fires at t=duration
    await act(async () => { rafCallbacks[0]?.(0) })
    await act(async () => { rafCallbacks[rafCallbacks.length - 1]?.(1000) })
    expect(result.current).toBe(100)
  })

  it('intermediate values are between 0 and target', async () => {
    const { result } = renderHook(() =>
      useCountUp({ target: 100, duration: 1000, isInView: true }),
    )
    await act(async () => { rafCallbacks[0]?.(0) })   // start: value = 0
    await act(async () => { rafCallbacks[rafCallbacks.length - 1]?.(500) }) // 50% progress
    expect(result.current).toBeGreaterThan(0)
    expect(result.current).toBeLessThan(100)
  })

  it('never returns a value > target', async () => {
    const { result } = renderHook(() =>
      useCountUp({ target: 50, duration: 500, isInView: true }),
    )
    await act(async () => { rafCallbacks[0]?.(0) })
    // Fire well past duration
    await act(async () => { rafCallbacks[rafCallbacks.length - 1]?.(9999) })
    expect(result.current).toBeLessThanOrEqual(50)
  })

  it('cleans up RAF on unmount', () => {
    const { unmount } = renderHook(() =>
      useCountUp({ target: 100, duration: 1000, isInView: true }),
    )
    const lastId = rafCallbacks.length // ID returned for last RAF
    unmount()
    expect(vi.mocked(window.cancelAnimationFrame)).toHaveBeenCalledWith(lastId)
  })
})

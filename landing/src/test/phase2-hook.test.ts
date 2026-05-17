import { describe, it, expect, beforeEach, vi } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import { useNavHide } from '../hooks/useNavHide'

function setScrollY(value: number) {
  Object.defineProperty(window, 'scrollY', {
    writable: true,
    configurable: true,
    value,
  })
}

beforeEach(() => {
  setScrollY(0)
})

describe('Phase 2 — useNavHide', () => {
  it('initial state is hidden=false', () => {
    const { result } = renderHook(() => useNavHide())
    expect(result.current.hidden).toBe(false)
  })

  it('initial state is scrolled=false', () => {
    const { result } = renderHook(() => useNavHide())
    expect(result.current.scrolled).toBe(false)
  })

  it('scrolling down 90px sets hidden=true', () => {
    const { result } = renderHook(() => useNavHide())

    act(() => {
      setScrollY(90)
      window.dispatchEvent(new Event('scroll'))
    })

    expect(result.current.hidden).toBe(true)
  })

  it('scrolling back up sets hidden=false', () => {
    const { result } = renderHook(() => useNavHide())

    act(() => {
      setScrollY(90)
      window.dispatchEvent(new Event('scroll'))
    })
    expect(result.current.hidden).toBe(true)

    act(() => {
      setScrollY(40)
      window.dispatchEvent(new Event('scroll'))
    })
    expect(result.current.hidden).toBe(false)
  })

  it('does not set hidden=true before 80px threshold', () => {
    const { result } = renderHook(() => useNavHide())

    act(() => {
      setScrollY(70)
      window.dispatchEvent(new Event('scroll'))
    })

    expect(result.current.hidden).toBe(false)
  })

  it('sets scrolled=true when scrollY > 0', () => {
    const { result } = renderHook(() => useNavHide())

    act(() => {
      setScrollY(10)
      window.dispatchEvent(new Event('scroll'))
    })

    expect(result.current.scrolled).toBe(true)
  })

  it('cleans up scroll listener on unmount', () => {
    const removeSpy = vi.spyOn(window, 'removeEventListener')
    const { unmount } = renderHook(() => useNavHide())
    unmount()
    expect(removeSpy).toHaveBeenCalledWith('scroll', expect.any(Function))
    removeSpy.mockRestore()
  })
})

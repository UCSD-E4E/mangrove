import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import App from '../App'

function renderApp() {
  return render(<MemoryRouter><App /></MemoryRouter>)
}

describe('Phase 0 — scaffold', () => {
  it('vitest test harness runs', () => {
    expect(true).toBe(true)
  })

  it('App renders a main element', () => {
    renderApp()
    expect(screen.getByRole('main')).toBeInTheDocument()
  })

  it('App renders an h1', () => {
    renderApp()
    expect(screen.getAllByRole('heading', { level: 1 }).length).toBeGreaterThan(0)
  })
})

import { describe, it, expect } from 'vitest'
import { readFileSync } from 'fs'
import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'
import tokens from '../styles/tokens'

const __dirname = dirname(fileURLToPath(import.meta.url))
const css = readFileSync(resolve(__dirname, '../styles/globals.css'), 'utf-8')

// ── Helpers ────────────────────────────────────────────────────────────────

function extractClass(css: string, className: string): string {
  const pattern = new RegExp(`\\.${className}\\s*\\{([^}]+)\\}`, 's')
  const match = css.match(pattern)
  return match ? match[1] : ''
}

// ── Token values ───────────────────────────────────────────────────────────

describe('Phase 1 — tokens.ts values', () => {
  it('has correct --bg-page', () => expect(tokens['--bg-page']).toBe('#f7f5f0'))
  it('has correct --bg-panel', () => expect(tokens['--bg-panel']).toBe('#ffffff'))
  it('has correct --bg-raised', () => expect(tokens['--bg-raised']).toBe('#f2efe8'))
  it('has correct --text-primary', () => expect(tokens['--text-primary']).toBe('#1a1a18'))
  it('has correct --text-secondary', () => expect(tokens['--text-secondary']).toBe('#666660'))
  it('has correct --text-muted', () => expect(tokens['--text-muted']).toBe('#aaa'))
  it('has correct --accent', () => expect(tokens['--accent']).toBe('#3d6b4a'))
  it('has correct --border', () => expect(tokens['--border']).toBe('rgba(0,0,0,0.06)'))
  it('has correct --class-mangrove', () => expect(tokens['--class-mangrove']).toBe('#3d6b4a'))
  it('has correct --class-water', () => expect(tokens['--class-water']).toBe('#4a90b8'))
  it('has correct --class-built-up', () => expect(tokens['--class-built-up']).toBe('#8a6a4a'))
  it('has correct --class-other-veg', () => expect(tokens['--class-other-veg']).toBe('#7a9a6a'))
  it('has correct --class-bare', () => expect(tokens['--class-bare']).toBe('#c8c0b0'))
})

// ── CSS variables match tokens.ts ──────────────────────────────────────────

describe('Phase 1 — CSS variables match tokens.ts', () => {
  it('every token value appears in globals.css :root', () => {
    for (const [key, value] of Object.entries(tokens)) {
      expect(css, `Missing: ${key}: ${value}`).toContain(`${key}: ${value}`)
    }
  })
})

// ── Google Fonts ───────────────────────────────────────────────────────────

describe('Phase 1 — Google Fonts import', () => {
  it('imports from fonts.googleapis.com', () => {
    expect(css).toMatch(/@import url\(.*fonts\.googleapis\.com/)
  })

  it('includes Instrument Serif', () => {
    expect(css).toContain('Instrument+Serif')
  })

  it('includes DM Sans', () => {
    expect(css).toContain('DM+Sans')
  })

  it('includes DM Mono', () => {
    expect(css).toContain('DM+Mono')
  })
})

// ── .overline ─────────────────────────────────────────────────────────────

describe('Phase 1 — .overline class', () => {
  const block = extractClass(css, 'overline')

  it('has font-size: 0.68rem', () => {
    expect(block).toContain('font-size: 0.68rem')
  })

  it('has letter-spacing: 0.18em', () => {
    expect(block).toContain('letter-spacing: 0.18em')
  })

  it('has text-transform: uppercase', () => {
    expect(block).toContain('text-transform: uppercase')
  })

  it('uses DM Sans font', () => {
    expect(block).toContain("'DM Sans'")
  })
})

// ── .caption ──────────────────────────────────────────────────────────────

describe('Phase 1 — .caption class', () => {
  const block = extractClass(css, 'caption')

  it('has font-size: 0.72rem', () => {
    expect(block).toContain('font-size: 0.72rem')
  })

  it('uses DM Mono font', () => {
    expect(block).toContain("'DM Mono'")
  })
})

// ── .serif-italic ─────────────────────────────────────────────────────────

describe('Phase 1 — .serif-italic class', () => {
  const block = extractClass(css, 'serif-italic')

  it('uses Instrument Serif font', () => {
    expect(block).toContain("'Instrument Serif'")
  })

  it('has font-style: italic', () => {
    expect(block).toContain('font-style: italic')
  })

  it('has color: var(--accent)', () => {
    expect(block).toContain('color: var(--accent)')
  })
})

// ── Heading weights ────────────────────────────────────────────────────────

describe('Phase 1 — heading weights', () => {
  it('h1 uses font-weight: 400', () => {
    const h1Block = css.match(/h1\s*\{([^}]+)\}/s)?.[1] ?? ''
    expect(h1Block).toContain('font-weight: 400')
  })

  it('h2 uses font-weight: 400', () => {
    const h2Block = css.match(/h2\s*\{([^}]+)\}/s)?.[1] ?? ''
    expect(h2Block).toContain('font-weight: 400')
  })

  it('no font-weight 600 anywhere in the stylesheet', () => {
    expect(css).not.toContain('font-weight: 600')
  })

  it('no font-weight 700 anywhere in the stylesheet', () => {
    expect(css).not.toContain('font-weight: 700')
  })
})

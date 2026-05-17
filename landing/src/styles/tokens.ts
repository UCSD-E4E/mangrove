export const tokens = {
  '--bg-page':        '#f7f5f0',
  '--bg-panel':       '#ffffff',
  '--bg-raised':      '#f2efe8',
  '--text-primary':   '#1a1a18',
  '--text-secondary': '#666660',
  '--text-muted':     '#aaa',
  '--accent':         '#3d6b4a',
  '--border':         'rgba(0,0,0,0.06)',

  '--class-mangrove':  '#3d6b4a',
  '--class-water':     '#4a90b8',
  '--class-built-up':  '#8a6a4a',
  '--class-other-veg': '#7a9a6a',
  '--class-bare':      '#c8c0b0',
} as const

export type TokenKey = keyof typeof tokens
export type TokenValue = (typeof tokens)[TokenKey]

export default tokens

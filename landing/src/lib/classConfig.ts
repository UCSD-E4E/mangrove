// Class index definitions matching the GEE pipeline ESA_TO_CLASS remapping:
// {10:1, 20:0, 30:0, 40:0, 50:2, 60:0, 70:0, 80:3, 90:4, 95:5, 100:0}

export const CLASS_NAMES: Record<number, string> = {
  0: 'Background',
  1: 'Tree Cover',
  2: 'Built-up',
  3: 'Water',
  4: 'Wetland',
  5: 'Mangrove',
}

// RGBA tuples for deck.gl getFillColor
export const CLASS_COLORS: Record<number, [number, number, number, number]> = {
  0: [242, 239, 232, 120],
  1: [ 61, 107,  74, 200],
  2: [138, 106,  74, 200],
  3: [ 74, 144, 184, 200],
  4: [120, 160, 120, 200],
  5: [ 30,  80,  50, 230],
}

export const LEGEND_ITEMS = [
  { label: 'Mangrove',   color: '#1e5032' },
  { label: 'Tree Cover', color: '#3d6b4a' },
  { label: 'Water',      color: '#4a90b8' },
  { label: 'Wetland',    color: '#789c78' },
  { label: 'Built-up',   color: '#8a6a4a' },
]

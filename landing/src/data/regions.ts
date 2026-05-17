// In production, set VITE_TILES_BASE_URL to the R2 public bucket URL.
// Locally, leave it unset — tiles are served from landing/public/tiles/.
const BASE = import.meta.env.VITE_TILES_BASE_URL ?? ''

export interface Region {
  id: string
  name: string
  displayName: string
  coords: { lng: number; lat: number }
  zoom: number
  pitch: number
  /** Raster PNG tiles (z0–12) — fast overview. Empty string = none. */
  tilesUrl: string
  /** PMTiles vector source for z13+ detail. Empty string = none. */
  pmtilesUrl: string
  status: 'trained' | 'testing'
}

export const regions: Region[] = [
  {
    id: 'florida',
    name: 'Florida',
    displayName: 'Florida, Cuba & Bahamas',
    coords: { lng: -80.9, lat: 25.2 },
    zoom: 9,
    pitch: 45,
    tilesUrl: `${BASE}/tiles/florida/{z}/{x}/{y}.png`,
    pmtilesUrl: `${BASE}/tiles/florida.pmtiles`,
    status: 'trained',
  },
  {
    id: 'brazil',
    name: 'Brazil',
    displayName: 'Brazil',
    coords: { lng: -48.5, lat: -1.4 },
    zoom: 9,
    pitch: 45,
    tilesUrl: '',
    pmtilesUrl: '',
    status: 'trained',
  },
  {
    id: 'indonesia',
    name: 'Indonesia',
    displayName: 'Indonesia',
    coords: { lng: 104.8, lat: -2.7 },
    zoom: 9,
    pitch: 45,
    tilesUrl: '',
    pmtilesUrl: '',
    status: 'trained',
  },
  {
    id: 'madagascar_mozambique',
    name: 'Madagascar & Mozambique',
    displayName: 'Madagascar + Mozambique',
    coords: { lng: 44.5, lat: -19.5 },
    zoom: 9,
    pitch: 45,
    tilesUrl: '',
    pmtilesUrl: '',
    status: 'trained',
  },
  {
    id: 'north_australia',
    name: 'N. Australia',
    displayName: 'North Australia',
    coords: { lng: 136.5, lat: -12.5 },
    zoom: 9,
    pitch: 45,
    tilesUrl: '',
    pmtilesUrl: '',
    status: 'trained',
  },
  {
    id: 'east_india_bangladesh',
    name: 'Sundarbans',
    displayName: 'Sundarbans, Bangladesh & India',
    coords: { lng: 89.5, lat: 21.9 },
    zoom: 9,
    pitch: 45,
    tilesUrl: '',
    pmtilesUrl: '',
    status: 'trained',
  },
]

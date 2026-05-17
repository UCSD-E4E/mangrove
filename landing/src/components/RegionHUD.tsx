import type { Region } from '../data/regions'

function formatLat(lat: number): string {
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? 'N' : 'S'}`
}
function formatLng(lng: number): string {
  return `${Math.abs(lng).toFixed(2)}°${lng >= 0 ? 'E' : 'W'}`
}

interface Props {
  region: Region | null
}

export function RegionHUD({ region }: Props) {
  if (!region) return null
  return (
    <div
      data-testid="region-hud"
      style={{
        position: 'absolute',
        top: '1.5rem',
        left: '1.5rem',
        background: 'var(--bg-panel)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '12px 16px',
        zIndex: 10,
        minWidth: '200px',
      }}
    >
      <p className="overline" style={{ marginBottom: '0.4rem' }}>
        {region.displayName}
      </p>
      <p
        data-testid="region-coords"
        className="caption"
        style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}
      >
        {formatLat(region.coords.lat)} · {formatLng(region.coords.lng)}
      </p>
    </div>
  )
}

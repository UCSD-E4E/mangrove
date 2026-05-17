import { LEGEND_ITEMS } from '../lib/classConfig'

export function TerrainLegend() {
  return (
    <div
      data-testid="terrain-legend"
      style={{
        position: 'absolute',
        top: '1.5rem',
        right: '1.5rem',
        background: 'var(--bg-panel)',
        border: '1px solid var(--border)',
        borderRadius: '6px',
        padding: '12px 16px',
        zIndex: 10,
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        minWidth: '140px',
      }}
    >
      {LEGEND_ITEMS.map((item) => (
        <div
          key={item.label}
          data-testid="legend-row"
          style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
        >
          <div
            data-testid="legend-dot"
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              backgroundColor: item.color,
              flexShrink: 0,
            }}
          />
          <span className="caption" style={{ color: 'var(--text-secondary)' }}>
            {item.label}
          </span>
        </div>
      ))}
    </div>
  )
}

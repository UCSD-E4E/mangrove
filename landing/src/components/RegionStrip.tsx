import { regions } from '../data/regions'

interface Props {
  activeRegionId: string
  onSelectRegion: (id: string) => void
}

export function RegionStrip({ activeRegionId, onSelectRegion }: Props) {
  return (
    <div
      data-testid="region-strip"
      style={{
        position: 'absolute',
        bottom: '1.5rem',
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        gap: '6px',
        zIndex: 10,
        background: 'rgba(247,245,240,0.85)',
        backdropFilter: 'blur(12px)',
        borderRadius: '999px',
        padding: '6px 8px',
        border: '1px solid var(--border)',
      }}
    >
      {regions.map((region) => {
        const isActive = region.id === activeRegionId
        return (
          <button
            key={region.id}
            data-testid="region-pill"
            data-active={isActive ? 'true' : 'false'}
            onClick={() => onSelectRegion(region.id)}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '5px',
              padding: '4px 12px',
              borderRadius: '999px',
              border: 'none',
              cursor: 'pointer',
              backgroundColor: isActive ? 'var(--accent)' : 'transparent',
              color: isActive ? '#fff' : 'var(--text-secondary)',
              fontFamily: "'DM Sans', sans-serif",
              fontSize: '0.72rem',
              fontWeight: 400,
              letterSpacing: '0.03em',
              transition: 'background-color 0.2s, color 0.2s',
            }}
          >
            <span
              data-testid="status-dot"
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: region.status === 'trained' ? '#3d6b4a' : '#aaa',
                flexShrink: 0,
              }}
            />
            {region.name}
          </button>
        )
      })}
    </div>
  )
}

interface Props {
  onZoomIn: () => void
  onZoomOut: () => void
  onReset: () => void
}

const btnStyle: React.CSSProperties = {
  width: '36px',
  height: '36px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  border: '1px solid var(--border)',
  borderRadius: '6px',
  background: 'var(--bg-panel)',
  cursor: 'pointer',
  fontFamily: "'DM Sans', sans-serif",
  fontSize: '1rem',
  color: 'var(--text-secondary)',
}

export function TerrainControls({ onZoomIn, onZoomOut, onReset }: Props) {
  return (
    <div
      data-testid="terrain-controls"
      style={{
        position: 'absolute',
        bottom: '1.5rem',
        right: '1.5rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        zIndex: 10,
      }}
    >
      <button aria-label="Zoom in" onClick={onZoomIn} style={btnStyle}>+</button>
      <button aria-label="Zoom out" onClick={onZoomOut} style={btnStyle}>−</button>
      <button aria-label="Reset view" onClick={onReset} style={{ ...btnStyle, fontSize: '0.85rem' }}>↺</button>
    </div>
  )
}

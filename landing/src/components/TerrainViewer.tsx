import { useEffect, useRef, useState } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { MapViewer } from './MapViewer'
import type { MapViewerHandle } from './MapViewer'
import { RegionHUD } from './RegionHUD'
import { TerrainLegend } from './TerrainLegend'
import { RegionStrip } from './RegionStrip'
import { TerrainControls } from './TerrainControls'
import { regions } from '../data/regions'

gsap.registerPlugin(ScrollTrigger)

export function TerrainViewer() {
  const sectionRef = useRef<HTMLElement>(null)
  const mapRef = useRef<MapViewerHandle>(null)
  const [activeRegionId, setActiveRegionId] = useState('florida')
  const [mapMounted, setMapMounted] = useState(false)

  const activeRegion = regions.find((r) => r.id === activeRegionId) ?? null

  // Mount MapViewer only when the section is approaching the viewport so
  // WebGL shader compilation doesn't block the page's initial scroll.
  useEffect(() => {
    if (!sectionRef.current) return
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setMapMounted(true) },
      { rootMargin: '400px' },
    )
    observer.observe(sectionRef.current)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!sectionRef.current) return
    const ctx = gsap.context(() => {
      ScrollTrigger.create({
        trigger: sectionRef.current,
        pin: true,
        start: 'top top',
        end: '+=150%',
        onEnter: () => mapRef.current?.flyToRegion('florida'),
      })
    }, sectionRef)
    return () => ctx.revert()
  }, [])

  const handleSelectRegion = (id: string) => {
    setActiveRegionId(id)
    mapRef.current?.flyToRegion(id)
  }

  const handleReset = () => {
    setActiveRegionId('florida')
    mapRef.current?.resetView()
  }

  return (
    <section
      ref={sectionRef}
      data-testid="terrain-viewer"
      style={{ position: 'relative', height: '100vh', overflow: 'hidden' }}
    >
      <div style={{ position: 'absolute', inset: 0, background: '#1a1f2e' }}>
        {mapMounted && (
          <MapViewer ref={mapRef} tilesUrl={activeRegion?.tilesUrl ?? ''} pmtilesUrl={activeRegion?.pmtilesUrl ?? ''} />
        )}
      </div>

      <div
        style={{
          position: 'absolute',
          top: '24px',
          left: '24px',
          zIndex: 10,
          color: '#fff',
          textShadow: '0 1px 4px rgba(0,0,0,0.5)',
        }}
      >
        <p className="overline" style={{ marginBottom: '4px', color: 'rgba(255,255,255,0.7)' }}>
          Global coverage
        </p>
        <h2 style={{ margin: 0 }}>
          Explore <em className="serif-italic">six regions</em>
        </h2>
      </div>

      <RegionHUD region={activeRegion} />
      <TerrainLegend />
      <RegionStrip activeRegionId={activeRegionId} onSelectRegion={handleSelectRegion} />
      <TerrainControls
        onZoomIn={() => mapRef.current?.zoomIn()}
        onZoomOut={() => mapRef.current?.zoomOut()}
        onReset={handleReset}
      />
    </section>
  )
}

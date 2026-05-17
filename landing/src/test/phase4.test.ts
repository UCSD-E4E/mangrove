import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { regions } from '../data/regions'
import type { Region } from '../data/regions'

describe('Phase 4 — Regions data layer', () => {
  it('exports exactly 6 regions', () => {
    expect(regions).toHaveLength(6)
  })

  it('each region has all required fields', () => {
    const required: (keyof Region)[] = [
      'id', 'name', 'displayName', 'coords', 'zoom', 'pitch', 'tilesUrl', 'pmtilesUrl', 'status',
    ]
    for (const region of regions) {
      for (const field of required) {
        expect(region[field], `${region.id}.${field}`).toBeDefined()
      }
    }
  })

  it('each region has coords with lng and lat', () => {
    for (const region of regions) {
      expect(typeof region.coords.lng).toBe('number')
      expect(typeof region.coords.lat).toBe('number')
    }
  })

  it('all lng values are in [-180, 180]', () => {
    for (const region of regions) {
      expect(region.coords.lng).toBeGreaterThanOrEqual(-180)
      expect(region.coords.lng).toBeLessThanOrEqual(180)
    }
  })

  it('all lat values are in [-90, 90]', () => {
    for (const region of regions) {
      expect(region.coords.lat).toBeGreaterThanOrEqual(-90)
      expect(region.coords.lat).toBeLessThanOrEqual(90)
    }
  })

  it('zoom is a positive number for all regions', () => {
    for (const region of regions) {
      expect(region.zoom).toBeGreaterThan(0)
    }
  })

  it('pitch is between 0 and 85 for all regions', () => {
    for (const region of regions) {
      expect(region.pitch).toBeGreaterThanOrEqual(0)
      expect(region.pitch).toBeLessThanOrEqual(85)
    }
  })

  it('all regions have status === "trained"', () => {
    for (const region of regions) {
      expect(region.status).toBe('trained')
    }
  })

  it('tilesUrl is a string on every region', () => {
    for (const region of regions) {
      expect(typeof region.tilesUrl).toBe('string')
    }
  })
})

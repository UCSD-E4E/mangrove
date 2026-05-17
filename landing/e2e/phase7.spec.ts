import { test, expect } from '@playwright/test'

test('Terrain section is present in the page', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="terrain-viewer"]')).toBeAttached()
})

test('Terrain section contains a canvas once scrolled into view', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="terrain-viewer"]')?.scrollIntoView(),
  )
  const terrain = page.locator('[data-testid="terrain-viewer"]')
  await expect(terrain.locator('canvas')).toBeAttached({ timeout: 8000 })
})

test('TerrainLegend renders 5 rows', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="terrain-viewer"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(300)
  await expect(page.locator('[data-testid="legend-row"]')).toHaveCount(5)
})

test('Clicking a region pill updates RegionHUD to that region name', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="terrain-viewer"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(300)
  // Click the Brazil pill
  const pills = page.locator('[data-testid="region-pill"]')
  await pills.nth(1).click() // index 1 = Brazil
  await expect(page.locator('[data-testid="region-hud"]')).toContainText('Brazil')
})

test('RegionStrip renders 6 pills', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="region-pill"]').first()).toBeAttached()
  await expect(page.locator('[data-testid="region-pill"]')).toHaveCount(6)
})

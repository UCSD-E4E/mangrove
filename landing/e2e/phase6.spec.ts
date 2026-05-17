import { test, expect } from '@playwright/test'

test('About section is below the fold on page load', async ({ page }) => {
  await page.goto('/')
  const viewport = page.viewportSize()!
  const box = await page.locator('[data-testid="about"]').boundingBox()
  expect(box!.y).toBeGreaterThanOrEqual(viewport.height)
})

test('About section becomes visible when scrolled into view', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="about"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(1200) // allow fade-in transition (0.8s)
  const opacity = await page
    .locator('[data-testid="about"]')
    .evaluate((el) => window.getComputedStyle(el).opacity)
  expect(parseFloat(opacity)).toBeGreaterThan(0.9)
})

test('Stat values animate to correct final values after scroll', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="about"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(2500) // count-up duration is 2000ms + buffer
  const about = page.locator('[data-testid="about"]')
  await expect(about.getByText(/147k km²/)).toBeVisible()
  await expect(about.getByText(/4×/)).toBeVisible()
  await expect(about.getByText(/0\.35m/)).toBeVisible()
  await expect(about.getByText(/6 regions/)).toBeVisible()
})

test('About section is invisible (opacity 0) before scrolling', async ({ page }) => {
  await page.goto('/')
  // Do NOT scroll — check initial opacity
  const opacity = await page
    .locator('[data-testid="about"]')
    .evaluate((el) => window.getComputedStyle(el).opacity)
  expect(parseFloat(opacity)).toBeLessThan(0.1)
})

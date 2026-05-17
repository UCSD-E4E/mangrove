import { test, expect } from '@playwright/test'

test('Canvas element is present in hero on page load', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="hero"] canvas')).toBeAttached({ timeout: 5000 })
})

test('Canvas has non-zero dimensions', async ({ page }) => {
  await page.goto('/')
  const canvas = page.locator('[data-testid="hero"] canvas').first()
  await expect(canvas).toBeAttached({ timeout: 5000 })
  const box = await canvas.boundingBox()
  expect(box?.width).toBeGreaterThan(0)
  expect(box?.height).toBeGreaterThan(0)
})

test('No unhandled JS errors during globe load and auto-rotation', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', (err) => errors.push(err.message))
  await page.goto('/')
  // Allow a few frames of auto-rotation
  await page.waitForTimeout(400)
  // Filter known harmless browser warnings
  const critical = errors.filter(
    (e) => !e.includes('ResizeObserver') && !e.includes('Non-Error promise rejection')
  )
  expect(critical).toHaveLength(0)
})

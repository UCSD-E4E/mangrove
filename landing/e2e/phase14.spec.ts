import { test, expect } from '@playwright/test'

test('page title is correct', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveTitle('Mangrove Monitor — E4E Lab, UC San Diego')
})

test('meta description is present and non-empty', async ({ page }) => {
  await page.goto('/')
  const content = await page.locator('meta[name="description"]').getAttribute('content')
  expect(content).toBeTruthy()
  expect(content!.length).toBeGreaterThan(10)
})

test('og:title meta tag is present', async ({ page }) => {
  await page.goto('/')
  const content = await page.locator('meta[property="og:title"]').getAttribute('content')
  expect(content).toContain('Mangrove Monitor')
})

test('every H2 on the page has at least one em child', async ({ page }) => {
  await page.goto('/')
  const h2sWithoutEm = await page.evaluate(() =>
    Array.from(document.querySelectorAll('h2'))
      .filter(h => h.querySelector('em') === null)
      .map(h => h.textContent?.trim() ?? ''),
  )
  expect(h2sWithoutEm).toHaveLength(0)
})

test('TerrainControls icon buttons have aria-label attributes', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('button[aria-label="Zoom in"]')).toBeAttached()
  await expect(page.locator('button[aria-label="Zoom out"]')).toBeAttached()
  await expect(page.locator('button[aria-label="Reset view"]')).toBeAttached()
})

test('no JS errors on page load and full scroll', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', err => errors.push(err.message))
  await page.goto('/')
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight))
  await page.waitForTimeout(500)
  expect(errors).toHaveLength(0)
})

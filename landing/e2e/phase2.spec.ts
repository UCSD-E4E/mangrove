import { test, expect } from '@playwright/test'

test('Nav is visible at page load', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('header')).toBeVisible()
  await expect(page.locator('header')).toHaveAttribute('data-hidden', 'false')
})

test('Nav contains wordmark and three links', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('header').getByText('Mangrove Monitor')).toBeVisible()
  await expect(page.locator('nav[aria-label="Site navigation"] a')).toHaveCount(3)
})

test('Nav hides after scrolling down past 80px', async ({ page }) => {
  await page.goto('/')
  await page.mouse.wheel(0, 300)
  await page.waitForTimeout(500)
  await expect(page.locator('header')).toHaveAttribute('data-hidden', 'true')
})

test('Nav reappears after scrolling back up', async ({ page }) => {
  await page.goto('/')
  await page.mouse.wheel(0, 300)
  await page.waitForTimeout(500)
  // Programmatic scroll-to-top is more reliable than mouse.wheel on heavy pages
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.waitForTimeout(500)
  await expect(page.locator('header')).toHaveAttribute('data-hidden', 'false')
})

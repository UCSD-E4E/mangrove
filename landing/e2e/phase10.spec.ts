import { test, expect } from '@playwright/test'

test('Team section renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="team"]')).toBeAttached()
})

test('Team section has exactly 3 cards', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="team-card"]')).toHaveCount(3)
})

test('JoinCTA section renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="join-cta"]')).toBeAttached()
})

test('CTA button is present with correct text', async ({ page }) => {
  await page.goto('/')
  const btn = page.locator('[data-testid="cta-button"]')
  await expect(btn).toBeAttached()
  await expect(btn).toContainText('Get in touch')
})

test('Footer renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="footer"]')).toBeAttached()
})

test('Footer copyright mentions Engineers for Exploration', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="copyright"]')).toContainText('Engineers for Exploration')
})

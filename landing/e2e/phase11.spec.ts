import { test, expect } from '@playwright/test'

test('page loads without JS errors', async ({ page }) => {
  const errors: string[] = []
  page.on('pageerror', err => errors.push(err.message))
  await page.goto('/')
  await page.waitForLoadState('networkidle')
  expect(errors).toHaveLength(0)
})

test('SplitCompare is visible after scrolling into view', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="split-compare"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(1200) // 0.8 s animation + buffer
  await expect(page.locator('[data-testid="split-compare"]')).toBeVisible()
})

test('Team cards are visible after scrolling to team section', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="team"]')?.scrollIntoView(),
  )
  // stagger: 3 cards × 0.12 s apart + 0.6 s duration → ~0.84 s total; allow generous buffer
  await page.waitForTimeout(1500)
  const cards = page.locator('[data-testid="team-card"]')
  await expect(cards.nth(0)).toBeVisible()
  await expect(cards.nth(1)).toBeVisible()
  await expect(cards.nth(2)).toBeVisible()
})

test('JoinCTA is visible after scrolling into view', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="join-cta"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(1200)
  await expect(page.locator('[data-testid="join-cta"]')).toBeVisible()
})

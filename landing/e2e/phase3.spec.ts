import { test, expect } from '@playwright/test'

test('Hero section is visible', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="hero"]')).toBeVisible()
})

test('Hero fills the full viewport height', async ({ page }) => {
  await page.goto('/')
  const viewportHeight = page.viewportSize()?.height ?? 768
  const heroBox = await page.locator('[data-testid="hero"]').boundingBox()
  expect(heroBox?.height).toBe(viewportHeight)
})

test('Hero overline text is correct', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="hero"]').getByText('Engineers for Exploration · UC San Diego')).toBeVisible()
})

test('Hero h1 contains mangroves in italic', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('h1')).toContainText("A living map of Earth's")
  await expect(page.locator('h1 em')).toContainText('mangroves')
})

test('Hero subhead is visible', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="hero"]').getByText(/Drag the globe/i)).toBeVisible()
})

test('Hero gradient overlay exists', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="gradient-overlay"]')).toBeAttached()
})

test('Scroll cue is visible', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="scroll-cue"]')).toBeVisible()
})

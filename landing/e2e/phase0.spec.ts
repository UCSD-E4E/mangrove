import { test, expect } from '@playwright/test'

test('page loads and body is visible', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('body')).toBeVisible()
})

test('page title is correct', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveTitle('Mangrove Monitor — E4E Lab, UC San Diego')
})

test('App renders an h1', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('h1')).toBeVisible()
})

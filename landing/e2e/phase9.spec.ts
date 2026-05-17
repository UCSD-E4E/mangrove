import { test, expect } from '@playwright/test'

async function scrollToPipeline(page: import('@playwright/test').Page) {
  await page.evaluate(() =>
    document.querySelector('[data-testid="pipeline"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(400)
}

test('Pipeline section renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="pipeline"]')).toBeAttached()
})

test('4 pipeline nodes are in the DOM', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="pipeline-node"]')).toHaveCount(4)
})

test('3 pipeline connectors are in the DOM', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="pipeline-connector"]')).toHaveCount(3)
})

test('All step labels are present', async ({ page }) => {
  await page.goto('/')
  const pipeline = page.locator('[data-testid="pipeline"]')
  await expect(pipeline.getByText('Sentinel-2 Input')).toBeAttached()
  await expect(pipeline.getByText('Contrastive Encoder')).toBeAttached()
  await expect(pipeline.getByText('Segmentation Head')).toBeAttached()
  await expect(pipeline.getByText('Super-Resolution', { exact: true })).toBeAttached()
})

test('Nodes become active after scrolling into view', async ({ page }) => {
  await page.goto('/')
  await scrollToPipeline(page)
  // stagger: 4 nodes × 0.3 s = 1.2 s total; allow generous buffer
  await page.waitForTimeout(2000)
  const nodes = page.locator('[data-testid="pipeline-node"]')
  await expect(nodes.nth(0)).toHaveAttribute('data-active', 'true')
  await expect(nodes.nth(1)).toHaveAttribute('data-active', 'true')
  await expect(nodes.nth(2)).toHaveAttribute('data-active', 'true')
  await expect(nodes.nth(3)).toHaveAttribute('data-active', 'true')
})

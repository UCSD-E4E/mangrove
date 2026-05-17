import { test, expect } from '@playwright/test'

// Without VITE_SPLIT_LEFT_URL / VITE_SPLIT_RIGHT_URL set, SplitCompare must
// fall back gracefully to the gradient div placeholders.

test('image-left falls back to a div element when VITE_SPLIT_LEFT_URL is not set', async ({ page }) => {
  await page.goto('/')
  const tag = await page
    .locator('[data-testid="image-left"]')
    .evaluate(el => el.tagName.toLowerCase())
  expect(tag).toBe('div')
})

test('image-right falls back to a div element when VITE_SPLIT_RIGHT_URL is not set', async ({ page }) => {
  await page.goto('/')
  const tag = await page
    .locator('[data-testid="image-right"]')
    .evaluate(el => el.tagName.toLowerCase())
  expect(tag).toBe('div')
})

test('SplitCompare drag still works after image-prop refactor', async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() =>
    document.querySelector('[data-testid="split-compare"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(400)

  const center = await page.locator('[data-testid="divider-handle"]').evaluate(el => {
    const r = el.getBoundingClientRect()
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 }
  })

  const before = await page
    .locator('[data-testid="divider"]')
    .evaluate(el => parseFloat((el as HTMLElement).style.left))

  await page.evaluate(
    ({ sx, ex, y }) => {
      const handle = document.querySelector('[data-testid="divider-handle"]')!
      handle.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: sx, clientY: y }))
      document.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: ex, clientY: y }))
      document.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }))
    },
    { sx: center.x, ex: center.x - 150, y: center.y },
  )
  await page.waitForTimeout(100)

  const after = await page
    .locator('[data-testid="divider"]')
    .evaluate(el => parseFloat((el as HTMLElement).style.left))

  expect(after).toBeLessThan(before)
})

import { test, expect } from '@playwright/test'

async function scrollToSplitCompare(page: import('@playwright/test').Page) {
  await page.evaluate(() =>
    document.querySelector('[data-testid="split-compare"]')?.scrollIntoView(),
  )
  await page.waitForTimeout(400)
}

/** Dispatch pointer events directly in the browser to trigger the React handler. */
async function dragHandle(
  page: import('@playwright/test').Page,
  deltaX: number,
) {
  const center = await page.locator('[data-testid="divider-handle"]').evaluate((el) => {
    const r = el.getBoundingClientRect()
    return { x: r.left + r.width / 2, y: r.top + r.height / 2 }
  })

  await page.evaluate(
    ({ sx, ex, y }) => {
      const handle = document.querySelector('[data-testid="divider-handle"]')!
      handle.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: sx, clientY: y }))
      document.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: ex, clientY: y }))
      document.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }))
    },
    { sx: center.x, ex: center.x + deltaX, y: center.y },
  )

  await page.waitForTimeout(100) // allow React to commit the state update
}

async function readDividerPct(page: import('@playwright/test').Page) {
  return page.locator('[data-testid="divider"]').evaluate(
    (el) => parseFloat((el as HTMLElement).style.left),
  )
}

test('SplitCompare section renders', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="split-compare"]')).toBeAttached()
})

test('Both images are in the DOM', async ({ page }) => {
  await page.goto('/')
  await expect(page.locator('[data-testid="image-left"]')).toBeAttached()
  await expect(page.locator('[data-testid="image-right"]')).toBeAttached()
})

test('User can drag divider left — position decreases', async ({ page }) => {
  await page.goto('/')
  await scrollToSplitCompare(page)
  const before = await readDividerPct(page)
  await dragHandle(page, -200)
  const after = await readDividerPct(page)
  expect(after).toBeLessThan(before)
})

test('User can drag divider right — position increases', async ({ page }) => {
  await page.goto('/')
  await scrollToSplitCompare(page)
  const before = await readDividerPct(page)
  await dragHandle(page, +200)
  const after = await readDividerPct(page)
  expect(after).toBeGreaterThan(before)
})

test('Drag handle has ew-resize cursor (pointer events wired up)', async ({ page }) => {
  await page.goto('/')
  await scrollToSplitCompare(page)
  const cursor = await page
    .locator('[data-testid="divider-handle"]')
    .evaluate((el) => window.getComputedStyle(el).cursor)
  expect(cursor).toBe('ew-resize')
})

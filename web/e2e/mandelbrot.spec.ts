import { test, expect } from "@playwright/test";

test.describe("Mandelbrot default view", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    // Wait for all tiles in the first generation to finish rendering.
    await page.waitForSelector("canvas[data-rendered]", { timeout: 20_000 });
    // Wait one animation frame so the GPU blit commits to the canvas backing buffer.
    await page.evaluate(() => new Promise<void>(r => requestAnimationFrame(() => r())));
  });

  test("renders a recognizable Mandelbrot set", async ({ page }) => {
    const canvas = page.locator("#canvas");
    // Mask the stats overlay — it contains a live gen counter that changes
    // between render passes and would cause spurious golden failures.
    await expect(canvas).toHaveScreenshot("mandelbrot-default.png", {
      mask: [page.locator("#stats-overlay")],
    });
  });

  test("canvas[data-rendered] is set after all tiles complete", async ({ page }) => {
    // Verifies the session's completion signal — used by the beforeEach wait.
    const gen = await page.locator("#canvas").getAttribute("data-rendered");
    expect(Number(gen)).toBeGreaterThanOrEqual(1);
  });
});

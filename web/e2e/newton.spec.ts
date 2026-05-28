import { test, expect } from "@playwright/test";

async function waitForRender(page: import("@playwright/test").Page) {
  await page.waitForSelector("canvas[data-rendered]", { timeout: 20_000 });
  await page.evaluate(() => new Promise<void>(r => requestAnimationFrame(() => r())));
}

// The stats overlay (#stats-overlay) contains a live generation counter that
// increments between render passes. Mask it so gen-count changes don't cause
// spurious golden failures — we want to test fractal geometry, not UI text.
const MASK = (page: import("@playwright/test").Page) => [page.locator("#stats-overlay")];

test.describe("Newton fractals", () => {
  test("renders z³−1 preset", async ({ page }) => {
    await page.goto("/#f=newton&preset=z3-1");
    await waitForRender(page);
    const canvas = page.locator("#canvas");
    await expect(canvas).toHaveScreenshot("newton-z3-1.png", { mask: MASK(page) });
  });

  test("renders z⁴−1 preset", async ({ page }) => {
    await page.goto("/#f=newton&preset=z4-1");
    await waitForRender(page);
    const canvas = page.locator("#canvas");
    await expect(canvas).toHaveScreenshot("newton-z4-1.png", { mask: MASK(page) });
  });

  test("z³−1 basin boundary region is stable", async ({ page }) => {
    await page.goto("/#f=newton&preset=z3-1");
    await waitForRender(page);
    // Clip the right-side boundary zone where all three basins interleave.
    // This region has fine fractal structure and is sensitive to coloring regressions.
    await expect(page).toHaveScreenshot("newton-boundary-clip.png", {
      clip: { x: 750, y: 280, width: 120, height: 120 },
    });
  });

  test("z³−1 with custom LUT palette from URL", async ({ page }) => {
    // A grayscale palette applied to Newton coloring — exercises the Newton LUT path.
    const GRAY_PALETTE = {
      stops: [
        { t: 0, r: 0.1, g: 0.1, b: 0.1, hueDir: "short" },
        { t: 0.5, r: 0.9, g: 0.9, b: 0.9, hueDir: "short" },
      ],
      cycleLen: 32,
    };
    const url = `/#f=newton&preset=z3-1&p=${encodeURIComponent(JSON.stringify(GRAY_PALETTE))}`;
    await page.goto(url);
    await waitForRender(page);
    await expect(page.locator("#canvas")).toHaveScreenshot(
      "newton-z3-1-gray-palette.png",
      { mask: MASK(page) },
    );
  });
});

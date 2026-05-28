import { test, expect } from "@playwright/test";

const MASK = (page: import("@playwright/test").Page) => [page.locator("#stats-overlay")];

async function waitForRender(page: import("@playwright/test").Page) {
  await page.waitForSelector("canvas[data-rendered]", { timeout: 30_000 });
  await page.evaluate(() => new Promise<void>(r => requestAnimationFrame(() => r())));
}

test.describe("Mandelbrot default view", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForRender(page);
  });

  test("renders a recognizable Mandelbrot set", async ({ page }) => {
    const canvas = page.locator("#canvas");
    // Mask the stats overlay — it contains a live gen counter that changes
    // between render passes and would cause spurious golden failures.
    await expect(canvas).toHaveScreenshot("mandelbrot-default.png", {
      mask: MASK(page),
    });
  });

  test("canvas[data-rendered] is set after all tiles complete", async ({ page }) => {
    // Verifies the session's completion signal — used by the beforeEach wait.
    const gen = await page.locator("#canvas").getAttribute("data-rendered");
    expect(Number(gen)).toBeGreaterThanOrEqual(1);
  });
});

test.describe("Mandelbrot deep zoom (perturbation path)", () => {
  // Elephant valley entrance — zoom_exp=17 engages the perturbation + BLA path.
  // This is a stable reference coordinate used in deep-zoom benchmark suites.
  const DEEP_URL =
    "#cx=0.3602404434376143&cy=-0.6413130610648031&z=17&f=mandelbrot";

  test("renders elephant valley at zoom_exp=17", async ({ page }) => {
    test.setTimeout(60_000);
    await page.goto(`/${DEEP_URL}`);
    await waitForRender(page);
    await expect(page.locator("#canvas")).toHaveScreenshot(
      "mandelbrot-deep-elephant.png",
      { mask: MASK(page) },
    );
  });
});

test.describe("Palette URL round-trip", () => {
  // Encode a minimal two-stop Fire-like palette directly in the URL.
  // The session should restore it on load and produce a visually distinct render.
  const FIRE_PALETTE = {
    stops: [
      { t: 0, r: 0, g: 0, b: 0, hueDir: "short" },
      { t: 0.5, r: 1, g: 0.2, b: 0, hueDir: "short" },
    ],
    cycleLen: 48,
  };
  const PALETTE_URL =
    `/#cx=-0.5&cy=0&z=0&f=mandelbrot&p=${encodeURIComponent(JSON.stringify(FIRE_PALETTE))}`;

  test("restores custom palette from URL hash", async ({ page }) => {
    await page.goto(PALETTE_URL);
    await waitForRender(page);
    await expect(page.locator("#canvas")).toHaveScreenshot(
      "mandelbrot-palette-fire-url.png",
      { mask: MASK(page) },
    );
  });
});

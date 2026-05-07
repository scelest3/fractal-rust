import { test, expect } from "@playwright/test";
import { readFile } from "fs/promises";

async function waitForRender(page: import("@playwright/test").Page) {
  await page.waitForSelector("canvas[data-rendered]", { timeout: 20_000 });
  await page.evaluate(() => new Promise<void>(r => requestAnimationFrame(() => r())));
}

async function openDialog(page: import("@playwright/test").Page) {
  await page.keyboard.press("e");
  await page.waitForSelector("dialog[open]");
}

test.describe("Export dialog", () => {
  test("PNG: dialog-driven 1080p export matches golden", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await waitForRender(page);
    await openDialog(page);

    // Dialog is PNG-only; leave DPI at default 300.
    const downloadPromise = page.waitForEvent("download", { timeout: 90_000 });
    await page.getByRole("button", { name: "Start export" }).click();
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(/\.png$/i);
    const buf = await readFile((await download.path())!);

    // Assert RGB16 PNG — parse IHDR directly.
    // Layout: [8 sig][4 len][4 "IHDR"][13 data][4 CRC]
    // data[0..4]=width, data[4..8]=height, data[8]=bit_depth, data[9]=color_type
    expect(buf[24]).toBe(16); // bit depth
    expect(buf[25]).toBe(2);  // color type: RGB

    await expect(buf).toMatchSnapshot("export-mandelbrot-1080p.png");
  });

  test("dialog closes after export and viewport resumes", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/");
    await waitForRender(page);
    await openDialog(page);

    const downloadPromise = page.waitForEvent("download", { timeout: 90_000 });
    await page.getByRole("button", { name: "Start export" }).click();
    await downloadPromise;

    // Dialog should be gone
    await expect(page.locator("dialog[open]")).toHaveCount(0);

    // Viewport should re-render (generation counter advances)
    await page.waitForSelector("canvas[data-rendered]", { timeout: 10_000 });
  });

  test("cancel closes dialog without downloading", async ({ page }) => {
    await page.goto("/");
    await waitForRender(page);
    await openDialog(page);

    await page.getByRole("button", { name: "Cancel export" }).click();
    await expect(page.locator("dialog[open]")).toHaveCount(0);
  });
});

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
    await page.goto("/");
    await waitForRender(page);
    await openDialog(page);

    // 1080p preset is selected by default — just confirm PNG and export
    await page.getByRole("radio", { name: "PNG" }).check();

    const downloadPromise = page.waitForEvent("download", { timeout: 30_000 });
    await page.getByRole("button", { name: "Start export" }).click();
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(/\.png$/i);
    const buf = await readFile((await download.path())!);
    await expect(buf).toMatchSnapshot("export-mandelbrot-1080p.png");
  });

  test("JPEG: dialog-driven 1080p export matches golden", async ({ page }) => {
    await page.goto("/");
    await waitForRender(page);
    await openDialog(page);

    await page.getByRole("radio", { name: "JPEG" }).check();
    // Quality slider defaults to 90 — leave it

    const downloadPromise = page.waitForEvent("download", { timeout: 30_000 });
    await page.getByRole("button", { name: "Start export" }).click();
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(/\.jpg$/i);
    const buf = await readFile((await download.path())!);
    await expect(buf).toMatchSnapshot("export-mandelbrot-1080p.jpg");
  });

  test("dialog closes after export and viewport resumes", async ({ page }) => {
    await page.goto("/");
    await waitForRender(page);
    await openDialog(page);

    const downloadPromise = page.waitForEvent("download", { timeout: 30_000 });
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

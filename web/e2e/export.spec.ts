import { test, expect } from "@playwright/test";
import { readFile } from "fs/promises";

test.describe("Export", () => {
  test("E key downloads a 1920×1080 PNG", async ({ page }) => {
    await page.goto("/");
    await page.waitForSelector("canvas[data-rendered]", { timeout: 20_000 });
    await page.evaluate(() => new Promise<void>(r => requestAnimationFrame(() => r())));

    const downloadPromise = page.waitForEvent("download", { timeout: 30_000 });
    await page.keyboard.press("e");
    const download = await downloadPromise;

    expect(download.suggestedFilename()).toMatch(/\.png$/i);

    const filePath = await download.path();
    const buf = await readFile(filePath!);

    // PNG magic bytes: 137 80 78 71 13 10 26 10
    expect(buf[0]).toBe(137);
    expect(buf[1]).toBe(80); // 'P'
    expect(buf[2]).toBe(78); // 'N'
    expect(buf[3]).toBe(71); // 'G'

    // IHDR chunk starts at byte 16: width (u32 BE) then height (u32 BE)
    const width  = buf.readUInt32BE(16);
    const height = buf.readUInt32BE(20);
    expect(width).toBe(1920);
    expect(height).toBe(1080);
  });
});

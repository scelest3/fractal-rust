import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  expect: {
    toHaveScreenshot: {
      // 2 LSB tolerance on 8-bit output per ARCHITECTURE.md testing guidelines.
      maxDiffPixelRatio: 0,
      threshold: 2 / 255,
    },
  },
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 15_000,
  },
});

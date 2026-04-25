import { defineConfig } from "vite";

const crossOriginHeaders = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

export default defineConfig({
  server: { headers: crossOriginHeaders },
  preview: { headers: crossOriginHeaders },
  build: {
    target: "es2022",
  },
});

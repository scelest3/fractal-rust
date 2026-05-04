import { pixelStep } from "./viewport.ts";
import type { GlPipeline } from "./gl-pipeline.ts";

/**
 * Height of each render strip in pixels.
 * Capped by the GPU's maximum renderbuffer size.
 */
export function computeStripHeight(
  exportHeight: number,
  maxRenderbufferSize: number,
): number {
  return Math.min(maxRenderbufferSize, exportHeight);
}

/**
 * Number of strips needed to cover the full export height.
 */
export function computeStripCount(
  exportHeight: number,
  stripHeight: number,
): number {
  return Math.ceil(exportHeight / stripHeight);
}

/**
 * Fractal units per pixel for an export render.
 * Fixes the vertical fractal extent to match the viewport; wider or taller
 * exports reveal more content on the sides or top/bottom respectively.
 */
export function computeExportStep(
  zoomExp: number,
  exportHeight: number,
): number {
  return pixelStep(zoomExp, exportHeight);
}

// ── Export pipeline types ─────────────────────────────────────────────────────

export interface WorkerLease {
  workers: Worker[];
  tileSab: SharedArrayBuffer;
  release(): void;
}

export interface ExportParams {
  width: number;
  height: number;
  format: "png" | "jpeg";
  view: { cx: string; cy: string; zoom_exp: number };
  fractalKind: "mandelbrot" | "newton";
  maxIter: number;
  useF64x2: boolean;
  cxRef: number;
  cyRef: number;
  newton?: { degree: number; coeffs: number[]; rootsRe: number[]; rootsIm: number[] };
}

// ── ExportSession ─────────────────────────────────────────────────────────────

const TILE = 256;
const TILE_SLOT_BYTES = TILE * TILE * 4 * 4; // 1 MiB per slot

function flipRows(pixels: Uint8Array, width: number, height: number): Uint8ClampedArray {
  const rowBytes = width * 4;
  const out = new Uint8ClampedArray(pixels.length);
  for (let y = 0; y < height; y++) {
    out.set(
      pixels.subarray((height - 1 - y) * rowBytes, (height - y) * rowBytes),
      y * rowBytes,
    );
  }
  return out;
}

type TileReadyMsg = { type: "tile_ready"; slotIndex: number; tileX: number; tileY: number; generation: number };

/**
 * Renders an export image using the leased worker pool and GL pipeline.
 * WorkerLease.release() is always called in a finally block.
 */
export class ExportSession {
  private cancelled = false;

  constructor(
    private readonly lease: WorkerLease,
    private readonly gl: GlPipeline,
    private readonly params: ExportParams,
    private readonly onProgress: (done: number, total: number) => void,
  ) {}

  cancel(): void {
    this.cancelled = true;
  }

  async run(): Promise<Blob | null> {
    try {
      return await this.doRun();
    } finally {
      this.lease.release();
    }
  }

  private async doRun(): Promise<Blob | null> {
    const { params, gl } = this;
    const { width, height } = params;

    const maxRbs = gl.getMaxRenderbufferSize();
    const stripHeight = computeStripHeight(height, maxRbs);
    const stripCount = computeStripCount(height, stripHeight);
    const step = computeExportStep(params.view.zoom_exp, height);

    const offscreen = new OffscreenCanvas(width, height);
    const ctx = offscreen.getContext("2d")!;

    gl.initExportFBOs(width, stripHeight);
    gl.setFractalKind(params.fractalKind === "newton" ? 1 : 0);
    if (params.fractalKind === "newton" && params.newton) {
      gl.setNewtonDegree(params.newton.degree);
    }

    const tilesX = Math.ceil(width / TILE);
    let totalTiles = 0;
    for (let s = 0; s < stripCount; s++) {
      const stripY = s * stripHeight;
      const actualH = Math.min(stripHeight, height - stripY);
      totalTiles += tilesX * Math.ceil(actualH / TILE);
    }
    let doneTiles = 0;

    for (let s = 0; s < stripCount; s++) {
      if (this.cancelled) return null;

      const stripY = s * stripHeight;
      const actualH = Math.min(stripHeight, height - stripY);
      const tilesY = Math.ceil(actualH / TILE);

      gl.clearExportFBO();
      await this.renderStrip(stripY, tilesX, tilesY, step);
      doneTiles += tilesX * tilesY;
      this.onProgress(doneTiles, totalTiles);

      if (this.cancelled) return null;
      gl.blitExportStrip();

      if (this.cancelled) return null;
      const pixels = await gl.readbackStrip();

      if (this.cancelled) return null;
      const flipped = flipRows(pixels, width, actualH);
      ctx.putImageData(new ImageData(flipped, width, actualH), 0, stripY);
    }

    return offscreen.convertToBlob({
      type: params.format === "jpeg" ? "image/jpeg" : "image/png",
    });
  }

  private renderStrip(
    stripY: number,
    tilesX: number,
    tilesY: number,
    step: number,
  ): Promise<void> {
    return new Promise((resolve) => {
      const { params, lease, gl } = this;
      const { width, height, useF64x2, cxRef, cyRef } = params;

      // Build pending tile list for this strip
      const pending: { tx: number; ty: number; deltaRe: number; deltaIm: number }[] = [];
      for (let ty = 0; ty < tilesY; ty++) {
        for (let tx = 0; tx < tilesX; tx++) {
          let deltaRe: number;
          let deltaIm: number;
          if (useF64x2) {
            deltaRe = (tx * TILE - width / 2) * step;
            deltaIm = (stripY + ty * TILE - height / 2) * step;
          } else {
            deltaRe = cxRef + (tx * TILE - width / 2) * step;
            deltaIm = cyRef + (stripY + ty * TILE - height / 2) * step;
          }
          pending.push({ tx, ty, deltaRe, deltaIm });
        }
      }

      const total = pending.length;
      let completed = 0;
      const busyWorkers = new Set<number>();

      const dispatch = () => {
        for (let i = 0; i < lease.workers.length; i++) {
          if (pending.length === 0) break;
          if (busyWorkers.has(i)) continue;
          const tile = pending.shift()!;
          busyWorkers.add(i);
          lease.workers[i].postMessage({
            type: "render_tile",
            deltaRe: tile.deltaRe,
            deltaIm: tile.deltaIm,
            step,
            maxIter: params.maxIter,
            slotIndex: i,
            tileX: tile.tx,
            tileY: tile.ty,
            generation: 1,
            useF64x2,
            cxRef,
            cyRef,
            fractalKind: params.fractalKind,
            newton: params.newton,
          });
        }
      };

      // Install export handler on each worker
      lease.workers.forEach((w, i) => {
        w.onmessage = (e: MessageEvent) => {
          const msg = e.data as TileReadyMsg;
          if (msg.type !== "tile_ready") return;
          busyWorkers.delete(i);
          gl.uploadExportTile(lease.tileSab, msg.slotIndex, msg.tileX, msg.tileY);
          completed++;
          if (completed === total) {
            resolve();
          } else {
            dispatch();
          }
        };
      });

      dispatch();
    });
  }
}

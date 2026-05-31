/**
 * BoxZoom — right-click drag (desktop) or long-press drag (touch) to zoom into
 * a selected region.
 *
 * The user draws a free-form rectangle; on release the shorter dimension is
 * expanded to match the canvas aspect ratio so the result fills the viewport.
 * A minimum box size of 8×8 px is enforced to prevent accidental zooms on
 * right-click without drag.
 */
import { pixelToFractal, type ViewState } from "./viewport.ts";

const MIN_BOX_PX = 8;
const LONG_PRESS_MS = 450;
const LONG_PRESS_CANCEL_PX = 10;

export class BoxZoom {
  private readonly box: HTMLDivElement;
  private startOffset = { x: 0, y: 0 };
  private startClient = { x: 0, y: 0 };
  private active = false;

  // Long-press touch state
  private longPressTimer: ReturnType<typeof setTimeout> | null = null;
  private longPressPointerId = -1;
  private longPressStart = { x: 0, y: 0 };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly getView: () => ViewState,
    private readonly onZoom: (view: ViewState) => void,
    private readonly getPixelScale: () => number = () => 1,
    private readonly onLongPressActivate?: () => void,
  ) {
    this.box = this.makeBox();

    canvas.addEventListener("contextmenu", (e) => e.preventDefault());
    canvas.addEventListener("pointerdown", (e) => this.onDown(e));
    canvas.addEventListener("pointermove", (e) => this.onMove(e));
    canvas.addEventListener("pointerup", (e) => this.onUp(e));
    canvas.addEventListener("pointercancel", () => this.cancel());
  }

  private makeBox(): HTMLDivElement {
    const el = document.createElement("div");
    Object.assign(el.style, {
      position: "fixed",
      display: "none",
      border: "2px dashed rgba(255,255,255,0.85)",
      background: "rgba(255,255,255,0.08)",
      boxSizing: "border-box",
      pointerEvents: "none",
      zIndex: "998",
    });
    document.body.appendChild(el);
    return el;
  }

  private onDown(e: PointerEvent): void {
    if (e.button === 2) {
      // Desktop right-click drag
      e.preventDefault();
      this.canvas.setPointerCapture(e.pointerId);
      const s = this.getPixelScale();
      this.startOffset = { x: e.offsetX * s, y: e.offsetY * s };
      this.startClient = { x: e.clientX, y: e.clientY };
      this.active = true;
    } else if (e.pointerType === "touch") {
      // Touch: start long-press timer; activate box zoom if held still long enough
      this.longPressPointerId = e.pointerId;
      this.longPressStart = { x: e.clientX, y: e.clientY };
      const startOffset = { x: e.offsetX * this.getPixelScale(), y: e.offsetY * this.getPixelScale() };
      const startClient = { x: e.clientX, y: e.clientY };
      this.longPressTimer = setTimeout(() => {
        this.longPressTimer = null;
        this.canvas.setPointerCapture(this.longPressPointerId);
        this.startOffset = startOffset;
        this.startClient = startClient;
        this.active = true;
        this.onLongPressActivate?.();
      }, LONG_PRESS_MS);
    }
  }

  private onMove(e: PointerEvent): void {
    // Cancel long-press if the finger strays beyond the threshold
    if (this.longPressTimer !== null && e.pointerId === this.longPressPointerId) {
      const dx = e.clientX - this.longPressStart.x;
      const dy = e.clientY - this.longPressStart.y;
      if (dx * dx + dy * dy > LONG_PRESS_CANCEL_PX * LONG_PRESS_CANCEL_PX) {
        clearTimeout(this.longPressTimer);
        this.longPressTimer = null;
      }
    }
    if (!this.active) return;
    const x = Math.min(this.startClient.x, e.clientX);
    const y = Math.min(this.startClient.y, e.clientY);
    const w = Math.abs(e.clientX - this.startClient.x);
    const h = Math.abs(e.clientY - this.startClient.y);
    Object.assign(this.box.style, {
      display: "block",
      left: `${x}px`,
      top: `${y}px`,
      width: `${w}px`,
      height: `${h}px`,
    });
  }

  private onUp(e: PointerEvent): void {
    // Clear timer if touch lifts before long-press activates
    if (this.longPressTimer !== null && e.pointerId === this.longPressPointerId) {
      clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }
    const committed = e.button === 2 ||
      (e.pointerType === "touch" && e.pointerId === this.longPressPointerId);
    if (!committed || !this.active) return;
    this.cancel();
    const s = this.getPixelScale();
    this.commit({ x: e.offsetX * s, y: e.offsetY * s });
  }

  private cancel(): void {
    if (this.longPressTimer !== null) {
      clearTimeout(this.longPressTimer);
      this.longPressTimer = null;
    }
    this.active = false;
    this.box.style.display = "none";
  }

  private commit(end: { x: number; y: number }): void {
    const { x: ex, y: ey } = end;
    const { x: sx, y: sy } = this.startOffset;

    if (Math.abs(ex - sx) < MIN_BOX_PX || Math.abs(ey - sy) < MIN_BOX_PX) return;

    const W = this.canvas.width;
    const H = this.canvas.height;
    const view = this.getView();

    const x1 = Math.min(sx, ex);
    const y1 = Math.min(sy, ey);
    const x2 = Math.max(sx, ex);
    const y2 = Math.max(sy, ey);

    const { fx: fx1, fy: fy1 } = pixelToFractal(x1, y1, view, W, H);
    const { fx: fx2, fy: fy2 } = pixelToFractal(x2, y2, view, W, H);

    const cx = (fx1 + fx2) / 2;
    const cy = (fy1 + fy2) / 2;

    // Expand the shorter fractal dimension to match the canvas aspect ratio.
    let fWidth = Math.abs(fx2 - fx1);
    let fHeight = Math.abs(fy2 - fy1);
    const aspect = W / H;
    if (fWidth / fHeight > aspect) {
      fHeight = fWidth / aspect;
    } else {
      fWidth = fHeight * aspect;
    }

    // zoom_exp = log10(default_height / new_height), where default_height = 4.
    const zoom_exp = Math.log10(4 / fHeight);

    this.onZoom({ cx: cx.toString(), cy: cy.toString(), zoom_exp });
  }
}

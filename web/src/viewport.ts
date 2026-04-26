/**
 * ViewState and ZoomPanFSM — pure coordinate and input module.
 * No WASM dependency. Used by session.ts to drive tile dispatch.
 */

// ── ViewState ─────────────────────────────────────────────────────────────────

/**
 * Canonical representation of the current fractal view.
 *
 * cx and cy are decimal strings to preserve full precision when serialized
 * into the URL hash or posted to the Orbit Worker. zoom_exp is log₁₀ of the
 * magnification factor: zoom_exp = 0 shows the full Mandelbrot set;
 * zoom_exp = 6 is 10⁶× magnification.
 */
export interface ViewState {
  /** Fractal center x — decimal string, e.g. "-0.743643887". */
  cx: string;
  /** Fractal center y — decimal string, e.g. "0.131825905". */
  cy: string;
  /** log₁₀(magnification). 0 = full view; larger = deeper zoom. */
  zoom_exp: number;
}

/** The default view showing the full Mandelbrot set. */
export const DEFAULT_VIEW: ViewState = {
  cx: "-0.5",
  cy: "0.0",
  zoom_exp: 0,
};

// ── Coordinate transforms ─────────────────────────────────────────────────────

/**
 * Fractal units visible across the canvas height at the given zoom level.
 * At zoom_exp = 0 the canvas height spans 4 fractal units (−2 to +2).
 */
export function fractalHeight(zoom_exp: number): number {
  return 4 * Math.pow(10, -zoom_exp);
}

/**
 * Fractal units per pixel for a given zoom level and canvas height.
 * This is the "pixel step" used to convert between pixel and fractal space.
 */
export function pixelStep(zoom_exp: number, canvasHeight: number): number {
  return fractalHeight(zoom_exp) / canvasHeight;
}

/**
 * Convert a canvas pixel (px, py) to fractal coordinates.
 * Canvas and fractal Y axes are parallel (both increase downward).
 */
export function pixelToFractal(
  px: number,
  py: number,
  view: ViewState,
  canvasWidth: number,
  canvasHeight: number,
): { fx: number; fy: number } {
  const step = pixelStep(view.zoom_exp, canvasHeight);
  return {
    fx: parseFloat(view.cx) + (px - canvasWidth / 2) * step,
    fy: parseFloat(view.cy) + (py - canvasHeight / 2) * step,
  };
}

/**
 * Convert fractal coordinates to canvas pixel (px, py).
 * Inverse of pixelToFractal.
 */
export function fractalToPixel(
  fx: number,
  fy: number,
  view: ViewState,
  canvasWidth: number,
  canvasHeight: number,
): { px: number; py: number } {
  const step = pixelStep(view.zoom_exp, canvasHeight);
  return {
    px: canvasWidth / 2 + (fx - parseFloat(view.cx)) / step,
    py: canvasHeight / 2 + (fy - parseFloat(view.cy)) / step,
  };
}

// ── ZoomPanFSM ────────────────────────────────────────────────────────────────

export type FsmState = "IDLE" | "PANNING" | "ZOOMING";

/**
 * Pan and zoom sensitivity. One typical mouse-wheel tick (deltaY ≈ 100)
 * produces a 0.1 change in zoom_exp — one tenth of a decade.
 */
const ZOOM_SPEED = 0.001;

/**
 * Milliseconds of inactivity after the last wheel event before `onZoomSettled`
 * fires. This is the trigger for reference orbit recomputation in Phase 2.
 */
const ZOOM_SETTLE_MS = 150;

export interface ZoomPanOptions {
  onViewChange?: (view: ViewState) => void;
  /** Fires once after wheel input has been quiet for ZOOM_SETTLE_MS. */
  onZoomSettled?: (view: ViewState) => void;
}

/**
 * Finite-state machine that translates pointer and wheel input into ViewState
 * changes.
 *
 * States:
 *   IDLE    — no active input
 *   PANNING — pointer down, dragging
 *   ZOOMING — wheel turning; transitions to IDLE after 300 ms of silence
 *
 * Call `attach(canvas)` to wire DOM events, or call the handle* methods
 * directly in tests.
 */
export class ZoomPanFSM {
  private state: FsmState = "IDLE";
  private view: ViewState;
  private canvasWidth = 0;
  private canvasHeight = 0;

  // Pan tracking
  private panStartPx = 0;
  private panStartPy = 0;

  // Zoom debounce
  private zoomDebounceId: ReturnType<typeof setTimeout> | null = null;

  private readonly notifyChange: (v: ViewState) => void;
  private readonly notifySettled: (v: ViewState) => void;

  constructor(initialView: ViewState, options: ZoomPanOptions = {}) {
    this.view = { ...initialView };
    this.notifyChange = options.onViewChange ?? (() => {});
    this.notifySettled = options.onZoomSettled ?? (() => {});
  }

  /** Current FSM state. */
  getState(): FsmState {
    return this.state;
  }

  /** Snapshot of the current view (immutable copy). */
  getView(): ViewState {
    return { ...this.view };
  }

  /** Set canvas dimensions (required before any handle* calls). */
  setCanvasSize(width: number, height: number): void {
    this.canvasWidth = width;
    this.canvasHeight = height;
  }

  /**
   * Attach DOM event listeners to a canvas element.
   * Returns a cleanup function that removes all listeners.
   */
  attach(canvas: HTMLCanvasElement): () => void {
    this.setCanvasSize(canvas.width, canvas.height);

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      this.handleWheel(e.deltaY, e.offsetX, e.offsetY);
    };
    const onPointerDown = (e: PointerEvent) => {
      canvas.setPointerCapture(e.pointerId);
      this.handlePointerDown(e.offsetX, e.offsetY);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (e.buttons === 0) return;
      this.handlePointerMove(e.offsetX, e.offsetY);
    };
    const onPointerUp = () => this.handlePointerUp();

    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    return () => {
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
    };
  }

  // ── Event handlers (public for testing) ─────────────────────────────────────

  handlePointerDown(px: number, py: number): void {
    this.state = "PANNING";
    this.panStartPx = px;
    this.panStartPy = py;
  }

  handlePointerMove(px: number, py: number): void {
    if (this.state !== "PANNING") return;

    const step = pixelStep(this.view.zoom_exp, this.canvasHeight);
    // Dragging right (px increases) means canvas moves right → fractal center moves left.
    const cx = parseFloat(this.view.cx) - (px - this.panStartPx) * step;
    const cy = parseFloat(this.view.cy) - (py - this.panStartPy) * step;

    this.panStartPx = px;
    this.panStartPy = py;

    this.view = { ...this.view, cx: cx.toString(), cy: cy.toString() };
    this.notifyChange(this.view);
  }

  handlePointerUp(): void {
    if (this.state === "PANNING") {
      this.state = "IDLE";
    }
  }

  /**
   * Handle a wheel event. deltaY > 0 (scroll down) zooms out; deltaY < 0 zooms in.
   * The fractal point under (px, py) stays fixed on screen after the zoom.
   */
  handleWheel(deltaY: number, px: number, py: number): void {
    // Fractal point under the cursor — must stay at (px, py) after zoom.
    const { fx, fy } = pixelToFractal(
      px,
      py,
      this.view,
      this.canvasWidth,
      this.canvasHeight,
    );

    // Scroll down (deltaY > 0) = zoom out = zoom_exp decreases.
    const zoom_exp = this.view.zoom_exp - deltaY * ZOOM_SPEED;

    // Recompute center so that (fx, fy) remains at pixel (px, py).
    const newStep = pixelStep(zoom_exp, this.canvasHeight);
    const cx = fx - (px - this.canvasWidth / 2) * newStep;
    const cy = fy - (py - this.canvasHeight / 2) * newStep;

    this.view = { cx: cx.toString(), cy: cy.toString(), zoom_exp };
    this.state = "ZOOMING";
    this.notifyChange(this.view);

    // Restart the 300 ms settle debounce.
    if (this.zoomDebounceId !== null) {
      clearTimeout(this.zoomDebounceId);
    }
    this.zoomDebounceId = setTimeout(() => {
      this.zoomDebounceId = null;
      this.state = "IDLE";
      this.notifySettled(this.view);
    }, ZOOM_SETTLE_MS);
  }
}

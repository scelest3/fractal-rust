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

/** Fractal-kind-specific parameters for URL serialization. */
export interface FractalParams {
  /** Non-integer exponent for multibrot (default 3.0) and julia (default 2.0). */
  exponent?: number;
  /** Julia fixed-c real part (default -0.7). */
  jre?: number;
  /** Julia fixed-c imaginary part (default 0.27). */
  jim?: number;
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

// ── URL serialization ─────────────────────────────────────────────────────────

/**
 * Serialize view state to a URL fragment key=value string (no leading `#`).
 *
 * Format: `cx=…&cy=…&z=…&f=…[&<fractal-specific params>]`
 *
 * zoom_exp is emitted as a decimal string (not a float) so that precision is
 * preserved through the v1 zoom range. Extra params (e.g. Newton preset) are
 * appended verbatim after `&f=…`.
 */
export function serializeViewState(
  view: ViewState,
  fractalKind: string,
  extraParams?: string,
  fractalParams?: FractalParams,
): string {
  const parts = [
    `cx=${view.cx}`,
    `cy=${view.cy}`,
    `z=${view.zoom_exp}`,
    `f=${fractalKind}`,
  ];
  if (fractalKind === "multibrot" && fractalParams?.exponent !== undefined) {
    parts.push(`exp=${fractalParams.exponent}`);
  }
  if (fractalKind === "julia") {
    if (fractalParams?.exponent !== undefined) parts.push(`exp=${fractalParams.exponent}`);
    if (fractalParams?.jre !== undefined) parts.push(`jre=${fractalParams.jre}`);
    if (fractalParams?.jim !== undefined) parts.push(`jim=${fractalParams.jim}`);
  }
  if (extraParams) parts.push(extraParams);
  return parts.join("&");
}

/**
 * Parse a URL fragment string (no leading `#`) for shared view state.
 *
 * Returns null if required fields (cx, cy, z, f) are absent or malformed.
 * The `extraParams` string (everything after `f=…`) is returned raw for
 * fractal-specific deserialization.
 */
export function deserializeViewState(fragment: string): {
  view: ViewState;
  fractalKind: string;
  extraParams: string;
  exponent?: number;
  juliaC?: { re: number; im: number };
} | null {
  const params = new URLSearchParams(fragment);
  const cx = params.get("cx");
  const cy = params.get("cy");
  const zStr = params.get("z");
  const f = params.get("f");
  if (!cx || !cy || zStr === null || !f) return null;
  const zoom_exp = parseFloat(zStr);
  if (isNaN(zoom_exp)) return null;

  const result: NonNullable<ReturnType<typeof deserializeViewState>> = {
    view: { cx, cy, zoom_exp },
    fractalKind: f,
    extraParams: fragment,
  };

  if (f === "multibrot") {
    const expStr = params.get("exp");
    result.exponent =
      expStr !== null && !isNaN(parseFloat(expStr)) ? parseFloat(expStr) : 3.0;
  }
  if (f === "julia") {
    const expStr = params.get("exp");
    const jreStr = params.get("jre");
    const jimStr = params.get("jim");
    result.exponent =
      expStr !== null && !isNaN(parseFloat(expStr)) ? parseFloat(expStr) : 2.0;
    result.juliaC = {
      re: jreStr !== null && !isNaN(parseFloat(jreStr)) ? parseFloat(jreStr) : -0.7,
      im: jimStr !== null && !isNaN(parseFloat(jimStr)) ? parseFloat(jimStr) : 0.27,
    };
  }

  return result;
}

// ── ZoomPanFSM ────────────────────────────────────────────────────────────────

export type FsmState = "IDLE" | "PANNING" | "ZOOMING" | "PINCHING";

/**
 * Pan and zoom sensitivity. One typical mouse-wheel tick (deltaY ≈ 100)
 * produces a 0.1 change in zoom_exp — one tenth of a decade.
 */
const ZOOM_SPEED = 0.001;

/** Pinch zoom sensitivity multiplier relative to wheel zoom. */
const PINCH_SENSITIVITY = 2;

/** Default precision cap: F64x2 renders cleanly to zoom_exp ≈ 17. */
const DEFAULT_MAX_ZOOM_EXP = 17;
const DEFAULT_MIN_ZOOM_EXP = -7;

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
  private pixelScale = 1; // CSS px → physical px multiplier for event coords

  // Zoom range
  private minZoomExp = DEFAULT_MIN_ZOOM_EXP;
  private maxZoomExp = DEFAULT_MAX_ZOOM_EXP;

  // Pointer tracking (pan + pinch)
  private pointers: Map<number, { x: number; y: number }> = new Map();
  private pinchLastDist = 0;

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

  /** Abort the current pan or pinch without changing the view. */
  cancel(): void {
    this.pointers.clear();
    this.state = "IDLE";
  }

  /** Snapshot of the current view (immutable copy). */
  getView(): ViewState {
    return { ...this.view };
  }

  /**
   * Overwrite the view from an external source (e.g. box-zoom or URL restore).
   * Cancels any active pan or zoom debounce and returns to IDLE.
   */
  setView(view: ViewState): void {
    this.view = { ...view, zoom_exp: this.clampZoom(view.zoom_exp) };
    this.state = "IDLE";
    if (this.zoomDebounceId !== null) {
      clearTimeout(this.zoomDebounceId);
      this.zoomDebounceId = null;
    }
    this.notifyChange(this.view);
  }

  /**
   * Constrain the zoom range. Call with fractal-specific limits — e.g. Newton
   * uses (-7, 14) since only f64 is implemented; Mandelbrot uses (-7, 17).
   */
  setZoomExpRange(min: number, max: number): void {
    this.minZoomExp = min;
    this.maxZoomExp = max;
    // Re-clamp the current view if it's now out of range.
    const clamped = this.clampZoom(this.view.zoom_exp);
    if (clamped !== this.view.zoom_exp) {
      this.view = { ...this.view, zoom_exp: clamped };
      this.notifyChange(this.view);
    }
  }

  private clampZoom(exp: number): number {
    return Math.max(this.minZoomExp, Math.min(this.maxZoomExp, exp));
  }

  /** Scale factor applied to CSS pointer-event coordinates (set to devicePixelRatio when HiDPI). */
  setPixelScale(scale: number): void {
    this.pixelScale = scale;
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
      this.handleWheel(e.deltaY, e.offsetX * this.pixelScale, e.offsetY * this.pixelScale);
    };
    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0) return; // ignore right-click (used by BoxZoom)
      canvas.setPointerCapture(e.pointerId);
      this.handlePointerDown(e.pointerId, e.offsetX * this.pixelScale, e.offsetY * this.pixelScale);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!this.pointers.has(e.pointerId)) return;
      if (this.state === "PANNING" && (e.buttons & 1) === 0) return;
      this.handlePointerMove(e.pointerId, e.offsetX * this.pixelScale, e.offsetY * this.pixelScale);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (e.button !== 0) return;
      this.handlePointerUp(e.pointerId);
    };

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

  handlePointerDown(pointerId: number, px: number, py: number): void {
    this.pointers.set(pointerId, { x: px, y: py });
    if (this.pointers.size === 1) {
      this.state = "PANNING";
    } else if (this.pointers.size === 2) {
      this.state = "PINCHING";
      this.pinchLastDist = this.activePinchDist();
    }
  }

  handlePointerMove(pointerId: number, px: number, py: number): void {
    if (this.state === "PANNING") {
      const prev = this.pointers.get(pointerId);
      if (!prev) return;
      // Dragging right (px increases) means canvas moves right → fractal center moves left.
      const step = pixelStep(this.view.zoom_exp, this.canvasHeight);
      const cx = parseFloat(this.view.cx) - (px - prev.x) * step;
      const cy = parseFloat(this.view.cy) - (py - prev.y) * step;
      this.view = { ...this.view, cx: cx.toString(), cy: cy.toString() };
      this.pointers.set(pointerId, { x: px, y: py });
      this.notifyChange(this.view);
    } else if (this.state === "PINCHING") {
      this.pointers.set(pointerId, { x: px, y: py });
      const dist = this.activePinchDist();
      if (dist > 0 && this.pinchLastDist > 0) {
        const mid = this.activePinchMidpoint();
        // Spreading fingers (dist > lastDist) → log10 > 0 → negative deltaY → zoom in.
        const deltaY = -PINCH_SENSITIVITY * Math.log10(dist / this.pinchLastDist) / ZOOM_SPEED;
        this.applyZoomAtPoint(deltaY, mid.x, mid.y);
        this.notifyChange(this.view);
      }
      this.pinchLastDist = dist;
    }
  }

  handlePointerUp(pointerId: number): void {
    this.pointers.delete(pointerId);
    if (this.state === "PINCHING") {
      this.state = this.pointers.size >= 1 ? "PANNING" : "IDLE";
    } else if (this.state === "PANNING" && this.pointers.size === 0) {
      this.state = "IDLE";
    }
  }

  /**
   * Handle a wheel event. deltaY > 0 (scroll down) zooms out; deltaY < 0 zooms in.
   * The fractal point under (px, py) stays fixed on screen after the zoom.
   */
  handleWheel(deltaY: number, px: number, py: number): void {
    this.applyZoomAtPoint(deltaY, px, py);
    this.state = "ZOOMING";
    this.notifyChange(this.view);

    // Restart the 150 ms settle debounce.
    if (this.zoomDebounceId !== null) {
      clearTimeout(this.zoomDebounceId);
    }
    this.zoomDebounceId = setTimeout(() => {
      this.zoomDebounceId = null;
      this.state = "IDLE";
      this.notifySettled(this.view);
    }, ZOOM_SETTLE_MS);
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  /** Zoom math shared by handleWheel and pinch — updates view, does not touch state or debounce. */
  private applyZoomAtPoint(deltaY: number, px: number, py: number): void {
    const { fx, fy } = pixelToFractal(px, py, this.view, this.canvasWidth, this.canvasHeight);
    // Scroll down (deltaY > 0) = zoom out = zoom_exp decreases.
    const zoom_exp = this.clampZoom(this.view.zoom_exp - deltaY * ZOOM_SPEED);
    // Recompute center so that (fx, fy) remains at pixel (px, py).
    const newStep = pixelStep(zoom_exp, this.canvasHeight);
    const cx = fx - (px - this.canvasWidth / 2) * newStep;
    const cy = fy - (py - this.canvasHeight / 2) * newStep;
    this.view = { cx: cx.toString(), cy: cy.toString(), zoom_exp };
  }

  private activePinchDist(): number {
    const pts = [...this.pointers.values()];
    if (pts.length < 2) return 0;
    const dx = pts[1].x - pts[0].x;
    const dy = pts[1].y - pts[0].y;
    return Math.sqrt(dx * dx + dy * dy);
  }

  private activePinchMidpoint(): { x: number; y: number } {
    const pts = [...this.pointers.values()];
    return { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2 };
  }
}

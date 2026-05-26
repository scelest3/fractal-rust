import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  type ViewState,
  DEFAULT_VIEW,
  fractalHeight,
  pixelStep,
  pixelToFractal,
  fractalToPixel,
  ZoomPanFSM,
  serializeViewState,
  deserializeViewState,
} from "../viewport.ts";

// ── fractalHeight ─────────────────────────────────────────────────────────────

describe("fractalHeight", () => {
  it("at zoom_exp=0, height is 4 fractal units", () => {
    expect(fractalHeight(0)).toBeCloseTo(4.0);
  });

  it("each decade of zoom halves the visible area by 10×", () => {
    expect(fractalHeight(1)).toBeCloseTo(fractalHeight(0) / 10);
    expect(fractalHeight(2)).toBeCloseTo(fractalHeight(0) / 100);
  });

  it("negative zoom_exp zooms out (shows more)", () => {
    expect(fractalHeight(-1)).toBeCloseTo(40.0);
  });
});

// ── pixelStep ─────────────────────────────────────────────────────────────────

describe("pixelStep", () => {
  it("at zoom_exp=0 and height=400, step is 0.01", () => {
    // 4 units / 400 px = 0.01 units/px
    expect(pixelStep(0, 400)).toBeCloseTo(0.01);
  });

  it("doubling canvas height halves the pixel step", () => {
    expect(pixelStep(0, 800)).toBeCloseTo(pixelStep(0, 400) / 2);
  });

  it("zoom_exp=1 gives a 10× smaller step", () => {
    expect(pixelStep(1, 400)).toBeCloseTo(pixelStep(0, 400) / 10);
  });
});

// ── pixelToFractal ────────────────────────────────────────────────────────────

describe("pixelToFractal", () => {
  const view: ViewState = { cx: "0.0", cy: "0.0", zoom_exp: 0 };
  const W = 800,
    H = 400;

  it("canvas center maps to the fractal center", () => {
    const { fx, fy } = pixelToFractal(W / 2, H / 2, view, W, H);
    expect(fx).toBeCloseTo(0);
    expect(fy).toBeCloseTo(0);
  });

  it("top-left pixel gives negative fx and fy", () => {
    const { fx, fy } = pixelToFractal(0, 0, view, W, H);
    expect(fx).toBeLessThan(0);
    expect(fy).toBeLessThan(0);
  });

  it("respects the view cx offset", () => {
    const shifted: ViewState = { cx: "1.0", cy: "0.0", zoom_exp: 0 };
    const { fx } = pixelToFractal(W / 2, H / 2, shifted, W, H);
    expect(fx).toBeCloseTo(1.0);
  });

  it("deeper zoom shrinks the pixel-to-fractal mapping", () => {
    const zoomed: ViewState = { cx: "0.0", cy: "0.0", zoom_exp: 1 };
    const { fx: fx0 } = pixelToFractal(W / 2 + 100, H / 2, view, W, H);
    const { fx: fx1 } = pixelToFractal(W / 2 + 100, H / 2, zoomed, W, H);
    expect(Math.abs(fx1)).toBeCloseTo(Math.abs(fx0) / 10);
  });
});

// ── fractalToPixel ────────────────────────────────────────────────────────────

describe("fractalToPixel", () => {
  const view: ViewState = { cx: "0.0", cy: "0.0", zoom_exp: 0 };
  const W = 800,
    H = 400;

  it("fractal center maps to canvas center", () => {
    const { px, py } = fractalToPixel(0, 0, view, W, H);
    expect(px).toBeCloseTo(W / 2);
    expect(py).toBeCloseTo(H / 2);
  });
});

// ── Round-trips ───────────────────────────────────────────────────────────────

describe("coordinate round-trips", () => {
  const view: ViewState = { cx: "-0.5", cy: "0.1", zoom_exp: 2 };
  const W = 1024,
    H = 768;

  it("pixel → fractal → pixel is identity", () => {
    const px0 = 300,
      py0 = 200;
    const { fx, fy } = pixelToFractal(px0, py0, view, W, H);
    const { px, py } = fractalToPixel(fx, fy, view, W, H);
    expect(px).toBeCloseTo(px0, 10);
    expect(py).toBeCloseTo(py0, 10);
  });

  it("fractal → pixel → fractal is identity", () => {
    const fx0 = -0.743,
      fy0 = 0.1318;
    const { px, py } = fractalToPixel(fx0, fy0, view, W, H);
    const { fx, fy } = pixelToFractal(px, py, view, W, H);
    expect(fx).toBeCloseTo(fx0, 10);
    expect(fy).toBeCloseTo(fy0, 10);
  });
});

// ── serializeViewState / deserializeViewState ─────────────────────────────────

describe("serializeViewState", () => {
  it("emits cx, cy, z, f keys", () => {
    const view: ViewState = { cx: "-0.743643887", cy: "0.131825905", zoom_exp: 12.5 };
    const s = serializeViewState(view, "mandelbrot");
    expect(s).toContain("cx=-0.743643887");
    expect(s).toContain("cy=0.131825905");
    expect(s).toContain("z=12.5");
    expect(s).toContain("f=mandelbrot");
  });

  it("appends extraParams after f=", () => {
    const view: ViewState = { cx: "0.0", cy: "0.0", zoom_exp: 0 };
    const s = serializeViewState(view, "newton", "preset=z3-1");
    expect(s).toContain("f=newton");
    expect(s).toContain("preset=z3-1");
  });

  it("omits extra when not provided", () => {
    const s = serializeViewState(DEFAULT_VIEW, "mandelbrot");
    expect(s).not.toContain("preset=");
    expect(s).not.toContain("coeffs=");
  });

  it("preserves cx/cy string precision exactly", () => {
    const cx = "-0.7436438870371587302";
    const view: ViewState = { cx, cy: "0.0", zoom_exp: 0 };
    const s = serializeViewState(view, "mandelbrot");
    expect(s).toContain(`cx=${cx}`);
  });
});

describe("deserializeViewState", () => {
  it("returns null for empty fragment", () => {
    expect(deserializeViewState("")).toBeNull();
  });

  it("returns null when required fields are missing", () => {
    expect(deserializeViewState("cx=0&cy=0")).toBeNull();
    expect(deserializeViewState("cx=0&cy=0&z=1")).toBeNull();
    expect(deserializeViewState("z=1&f=mandelbrot")).toBeNull();
  });

  it("returns null for non-numeric z", () => {
    expect(deserializeViewState("cx=0&cy=0&z=bad&f=mandelbrot")).toBeNull();
  });

  it("parses mandelbrot fragment correctly", () => {
    const result = deserializeViewState("cx=-0.5&cy=0.0&z=3&f=mandelbrot");
    expect(result).not.toBeNull();
    expect(result!.view.cx).toBe("-0.5");
    expect(result!.view.cy).toBe("0.0");
    expect(result!.view.zoom_exp).toBe(3);
    expect(result!.fractalKind).toBe("mandelbrot");
  });

  it("returns extraParams for fractal-specific deserialization", () => {
    const result = deserializeViewState("cx=0&cy=0&z=0&f=newton&preset=z3-1");
    expect(result!.extraParams).toContain("preset=z3-1");
  });
});

describe("serializeViewState / deserializeViewState round-trips", () => {
  it("round-trips mandelbrot view", () => {
    const view: ViewState = { cx: "-0.743643887", cy: "0.131825905", zoom_exp: 12.5 };
    const fragment = serializeViewState(view, "mandelbrot");
    const result = deserializeViewState(fragment)!;
    expect(result.view.cx).toBe(view.cx);
    expect(result.view.cy).toBe(view.cy);
    expect(result.view.zoom_exp).toBe(view.zoom_exp);
    expect(result.fractalKind).toBe("mandelbrot");
  });

  it("round-trips newton view with preset extra params", () => {
    const view: ViewState = { cx: "0.0", cy: "0.0", zoom_exp: 2 };
    const fragment = serializeViewState(view, "newton", "preset=z4-1");
    const result = deserializeViewState(fragment)!;
    expect(result.view.zoom_exp).toBe(2);
    expect(result.fractalKind).toBe("newton");
    expect(result.extraParams).toContain("preset=z4-1");
  });

  it("preserves high-precision cx/cy strings through round-trip", () => {
    const cx = "-1.9999999999999980";
    const cy = "0.000000000000001337";
    const view: ViewState = { cx, cy, zoom_exp: 0 };
    const fragment = serializeViewState(view, "mandelbrot");
    const result = deserializeViewState(fragment)!;
    expect(result.view.cx).toBe(cx);
    expect(result.view.cy).toBe(cy);
  });
});

// ── ZoomPanFSM ────────────────────────────────────────────────────────────────

describe("ZoomPanFSM", () => {
  const W = 800,
    H = 400;
  let fsm: ZoomPanFSM;

  beforeEach(() => {
    vi.useFakeTimers();
    fsm = new ZoomPanFSM({ cx: "0.0", cy: "0.0", zoom_exp: 0 });
    fsm.setCanvasSize(W, H);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // ── Initial state ───────────────────────────────────────────────────────────

  it("starts in IDLE", () => {
    expect(fsm.getState()).toBe("IDLE");
  });

  it("getView returns the initial view", () => {
    const v = fsm.getView();
    expect(parseFloat(v.cx)).toBeCloseTo(0);
    expect(v.zoom_exp).toBe(0);
  });

  // ── Pan state transitions ───────────────────────────────────────────────────

  it("IDLE → PANNING on pointerdown", () => {
    fsm.handlePointerDown(400, 200);
    expect(fsm.getState()).toBe("PANNING");
  });

  it("PANNING → IDLE on pointerup", () => {
    fsm.handlePointerDown(400, 200);
    fsm.handlePointerUp();
    expect(fsm.getState()).toBe("IDLE");
  });

  it("pointermove does nothing when not panning", () => {
    const onChange = vi.fn();
    fsm = new ZoomPanFSM(DEFAULT_VIEW, { onViewChange: onChange });
    fsm.setCanvasSize(W, H);
    fsm.handlePointerMove(100, 100);
    expect(onChange).not.toHaveBeenCalled();
  });

  // ── Pan direction ───────────────────────────────────────────────────────────

  it("dragging right decreases cx (canvas moves right, fractal center moves left)", () => {
    const onChange = vi.fn();
    fsm = new ZoomPanFSM({ cx: "0.0", cy: "0.0", zoom_exp: 0 }, { onViewChange: onChange });
    fsm.setCanvasSize(W, H);

    fsm.handlePointerDown(400, 200);
    fsm.handlePointerMove(600, 200); // drag 200 px right

    expect(onChange).toHaveBeenCalled();
    expect(parseFloat(fsm.getView().cx)).toBeLessThan(0);
  });

  it("dragging down decreases cy", () => {
    fsm.handlePointerDown(400, 200);
    fsm.handlePointerMove(400, 400); // drag 200 px down
    expect(parseFloat(fsm.getView().cy)).toBeLessThan(0);
  });

  it("pan updates start position incrementally (each move delta is relative)", () => {
    fsm.handlePointerDown(400, 200);
    fsm.handlePointerMove(300, 200); // move 100 left → cx +100*step
    const cx1 = parseFloat(fsm.getView().cx);

    fsm.handlePointerMove(200, 200); // move another 100 left → cx +100*step more
    const cx2 = parseFloat(fsm.getView().cx);

    const step = pixelStep(0, H);
    expect(cx2).toBeCloseTo(cx1 + 100 * step, 10);
  });

  // ── Zoom state transitions ──────────────────────────────────────────────────

  it("wheel → ZOOMING", () => {
    fsm.handleWheel(100, W / 2, H / 2);
    expect(fsm.getState()).toBe("ZOOMING");
  });

  it("scroll down (deltaY > 0) decreases zoom_exp (zooms out)", () => {
    fsm.handleWheel(100, W / 2, H / 2);
    expect(fsm.getView().zoom_exp).toBeLessThan(0);
  });

  it("scroll up (deltaY < 0) increases zoom_exp (zooms in)", () => {
    fsm.handleWheel(-100, W / 2, H / 2);
    expect(fsm.getView().zoom_exp).toBeGreaterThan(0);
  });

  // ── Zoom-settle debounce ────────────────────────────────────────────────────

  it("onZoomSettled fires after 150 ms of quiet", () => {
    const onSettled = vi.fn();
    fsm = new ZoomPanFSM({ cx: "0.0", cy: "0.0", zoom_exp: 0 }, { onZoomSettled: onSettled });
    fsm.setCanvasSize(W, H);

    fsm.handleWheel(-100, W / 2, H / 2);
    expect(onSettled).not.toHaveBeenCalled();

    vi.advanceTimersByTime(149);
    expect(onSettled).not.toHaveBeenCalled();

    vi.advanceTimersByTime(1); // 150 ms total
    expect(onSettled).toHaveBeenCalledOnce();
  });

  it("ZOOMING → IDLE after zoom settles", () => {
    fsm.handleWheel(-100, W / 2, H / 2);
    vi.advanceTimersByTime(150);
    expect(fsm.getState()).toBe("IDLE");
  });

  it("debounce resets on each wheel event — settles 150 ms after the last one", () => {
    const onSettled = vi.fn();
    fsm = new ZoomPanFSM({ cx: "0.0", cy: "0.0", zoom_exp: 0 }, { onZoomSettled: onSettled });
    fsm.setCanvasSize(W, H);

    fsm.handleWheel(-100, W / 2, H / 2);
    vi.advanceTimersByTime(100);

    fsm.handleWheel(-100, W / 2, H / 2); // restart debounce
    vi.advanceTimersByTime(100); // only 100 ms since last wheel
    expect(onSettled).not.toHaveBeenCalled();

    vi.advanceTimersByTime(50); // 150 ms since last wheel
    expect(onSettled).toHaveBeenCalledOnce();
  });

  // ── Zoom keeps cursor position fixed ─────────────────────────────────────────

  it("zoom at cursor keeps the fractal point under the cursor fixed", () => {
    // Cursor at (600, 200): fractal point is (2, 0) given step=0.01
    const cursorPx = 600,
      cursorPy = 200;

    const before = pixelToFractal(cursorPx, cursorPy, fsm.getView(), W, H);

    fsm.handleWheel(-100, cursorPx, cursorPy); // zoom in

    const after = pixelToFractal(cursorPx, cursorPy, fsm.getView(), W, H);

    expect(after.fx).toBeCloseTo(before.fx, 8);
    expect(after.fy).toBeCloseTo(before.fy, 8);
  });

  it("zoom at canvas center does not shift cx or cy", () => {
    fsm.handleWheel(-100, W / 2, H / 2); // zoom at center
    // When cursor is at canvas center, cx/cy should not change
    expect(parseFloat(fsm.getView().cx)).toBeCloseTo(0, 8);
    expect(parseFloat(fsm.getView().cy)).toBeCloseTo(0, 8);
  });

  // ── zoom_exp change per wheel tick ────────────────────────────────────────────

  it("zoom_exp changes by ~0.1 per 100-unit wheel tick", () => {
    fsm.handleWheel(-100, W / 2, H / 2);
    expect(fsm.getView().zoom_exp).toBeCloseTo(0.1, 5);
  });
});

//! Coloring algorithms: ports the GLSL blit shader to pure Rust.
//!
//! `color_pixel` is the authoritative implementation; the GLSL shader is a
//! validated approximation tested against it via Playwright golden tests.

const LUT_SIZE: usize = 4096;

pub struct ColorParams {
    pub cycle_len: f32,
    pub trap_radius: f32,
    pub trap_strength: f32,
    pub trap_color: [f32; 3],
    pub interior_color: [f32; 3],
    pub shading_color: [f32; 3],
    pub distance_strength: f32,
    pub distance_pow: f32,
    pub angle_strength: f32,
    pub period_strength: f32,
    pub period_cycle_len: f32,
    pub dist_weight: f32,
    pub newton_degree: u32,
    pub unresolved_color: [f32; 3],
}

/// Color one raw pixel, returning RGB16 (no alpha).
///
/// `raw`          — `[f32; 4]` from the tile buffer (fractal-kind-specific layout)
/// `lut`          — 4096-entry RGBA32F palette table
/// `params`       — all shader uniforms
/// `fractal_kind` — 0 = Mandelbrot, 1 = Newton
pub fn color_pixel(
    raw: [f32; 4],
    lut: &[[f32; 4]],
    params: &ColorParams,
    fractal_kind: u32,
) -> [u16; 3] {
    let [r, g, b] = shade(raw, lut, params, fractal_kind);
    [to_u16(r), to_u16(g), to_u16(b)]
}

fn shade(raw: [f32; 4], lut: &[[f32; 4]], p: &ColorParams, fractal_kind: u32) -> [f32; 3] {
    // Path 3 — Newton
    if fractal_kind == 1 {
        if raw[3] < 0.5 {
            return p.unresolved_color;
        }
        let hue = (raw[2] / p.newton_degree as f32) * 360.0;
        return hsl_to_rgb(hue, 0.85, 0.55);
    }

    // Path 2 — Interior Mandelbrot
    if raw[3] < 0.5 {
        let dist = clamp(raw[1], 0.0, 1.0).powf(p.distance_pow);
        let col = mix3(p.interior_color, p.shading_color, p.distance_strength * dist);
        let t_angle = glsl_fract((raw[2] + core::f32::consts::PI) / (2.0 * core::f32::consts::PI));
        let col = mix3(col, sample_lut(lut, t_angle), p.angle_strength);
        if raw[0] >= 0.5 && p.period_strength > 0.0 {
            let t_per = glsl_fract(raw[0] / p.period_cycle_len);
            return mix3(col, sample_lut(lut, t_per), p.period_strength);
        }
        return col;
    }

    // Path 1 — Escaped Mandelbrot
    let t = glsl_fract((raw[0] + p.dist_weight * raw[2]) / p.cycle_len);
    let escape_color = sample_lut(lut, t);
    let blend = p.trap_strength * smoothstep(p.trap_radius, 0.0, raw[1]);
    mix3(escape_color, p.trap_color, blend)
}

/// Linear interpolation into the 4096-entry LUT, matching GL_LINEAR texel sampling.
fn sample_lut(lut: &[[f32; 4]], t: f32) -> [f32; 3] {
    let pos = t * (LUT_SIZE as f32 - 1.0);
    let i0 = (pos as usize).min(LUT_SIZE - 1);
    let i1 = (i0 + 1).min(LUT_SIZE - 1);
    let frac = pos - pos.floor();
    [
        lut[i0][0] + frac * (lut[i1][0] - lut[i0][0]),
        lut[i0][1] + frac * (lut[i1][1] - lut[i0][1]),
        lut[i0][2] + frac * (lut[i1][2] - lut[i0][2]),
    ]
}

/// HSL → RGB matching the GLSL implementation in gl-pipeline.ts:57–68.
fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - (glsl_mod(h / 60.0, 2.0) - 1.0).abs());
    let m = l - c * 0.5;
    let (r, g, b) = if      h < 60.0  { (c, x, 0.0) }
                    else if h < 120.0 { (x, c, 0.0) }
                    else if h < 180.0 { (0.0, c, x) }
                    else if h < 240.0 { (0.0, x, c) }
                    else if h < 300.0 { (x, 0.0, c) }
                    else              { (c, 0.0, x) };
    [r + m, g + m, b + m]
}

fn to_u16(f: f32) -> u16 {
    (clamp(f, 0.0, 1.0) * 65535.0).round() as u16
}

/// GLSL fract: always returns value in [0, 1) regardless of sign.
fn glsl_fract(x: f32) -> f32 {
    x - x.floor()
}

/// GLSL mod: always same sign as y (matches GLSL, differs from Rust's % operator).
fn glsl_mod(x: f32, y: f32) -> f32 {
    x - y * (x / y).floor()
}

fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)]
}

/// GLSL smoothstep(edge0, edge1, x).
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_lut(r: f32, g: f32, b: f32) -> [[f32; 4]; 4096] {
        [[r, g, b, 1.0]; 4096]
    }

    fn default_params() -> ColorParams {
        ColorParams {
            cycle_len: 64.0,
            trap_radius: 0.5,
            trap_strength: 0.0,
            trap_color: [1.0, 1.0, 0.0],
            interior_color: [0.0, 0.0, 0.0],
            shading_color: [1.0, 1.0, 1.0],
            distance_strength: 0.0,
            distance_pow: 1.0,
            angle_strength: 0.0,
            period_strength: 0.0,
            period_cycle_len: 8.0,
            dist_weight: 0.0,
            newton_degree: 3,
            unresolved_color: [0.05, 0.05, 0.05],
        }
    }

    // ── Path 1: Escaped Mandelbrot ────────────────────────────────────────────

    #[test]
    fn escaped_no_trap_samples_lut() {
        // solid red LUT; escaped pixel with no trap → pure red
        let lut = flat_lut(1.0, 0.0, 0.0);
        let raw = [16.0, 0.9, 0.0, 1.0]; // smooth_t=16, orbit_min_r=0.9 (far from trap), escaped
        let result = color_pixel(raw, &lut, &default_params(), 0);
        assert_eq!(result, [65535, 0, 0]);
    }

    #[test]
    fn escaped_full_trap_blend() {
        // orbit_min_r=0 → smoothstep(0.5, 0.0, 0.0) = 1.0 → full trap color
        let lut = flat_lut(1.0, 0.0, 0.0);
        let params = ColorParams {
            trap_strength: 1.0,
            trap_color: [0.0, 1.0, 0.0],
            trap_radius: 0.5,
            ..default_params()
        };
        let raw = [0.0, 0.0, 0.0, 1.0]; // orbit_min_r=0 → inside trap
        let result = color_pixel(raw, &lut, &params, 0);
        assert_eq!(result, [0, 65535, 0]);
    }

    // ── Path 2: Interior Mandelbrot ───────────────────────────────────────────

    #[test]
    fn interior_default_is_black() {
        let lut = flat_lut(1.0, 0.0, 0.0);
        let raw = [0.0, 0.0, 0.0, 0.0]; // interior
        let result = color_pixel(raw, &lut, &default_params(), 0);
        assert_eq!(result, [0, 0, 0]);
    }

    #[test]
    fn interior_full_distance_shading() {
        // distance_strength=1, distance_pow=1, orbit_min_r=1.0 → full shading_color
        let lut = flat_lut(0.0, 0.0, 0.0);
        let params = ColorParams {
            distance_strength: 1.0,
            interior_color: [0.0, 0.0, 0.0],
            shading_color: [1.0, 1.0, 1.0],
            ..default_params()
        };
        let raw = [0.0, 1.0, 0.0, 0.0]; // orbit_min_r=1.0
        let result = color_pixel(raw, &lut, &params, 0);
        assert_eq!(result, [65535, 65535, 65535]);
    }

    // ── Path 3: Newton ────────────────────────────────────────────────────────

    #[test]
    fn newton_unresolved() {
        let lut = flat_lut(0.0, 0.0, 0.0);
        let raw = [0.0, 0.0, 0.0, 0.0]; // converged=0 → unresolved
        let result = color_pixel(raw, &lut, &default_params(), 1);
        // 0.05 * 65535 ≈ 3276.75 → rounds to 3277
        assert_eq!(result, [3277, 3277, 3277]);
    }

    #[test]
    fn newton_converged_root_zero() {
        let lut = flat_lut(0.0, 0.0, 0.0);
        let raw = [3.0, 0.0, 0.0, 1.0]; // root_index=0, newton_degree=3 → hue=0°
        let [r, g, b] = color_pixel(raw, &lut, &default_params(), 1);
        // hsl_to_rgb(0°, 0.85, 0.55): h=0 → sextant 0 → (c, x, 0)+m
        // c = (1-|2*0.55-1|)*0.85 = (1-0.1)*0.85 = 0.765
        // x = 0.765*(1-|mod(0/60,2)-1|) = 0.765*(1-1) = 0.0
        // m = 0.55 - 0.765/2 = 0.55 - 0.3825 = 0.1675
        // rgb = [0.765+0.1675, 0.0+0.1675, 0.0+0.1675] = [0.9325, 0.1675, 0.1675]
        assert_eq!(r, to_u16(0.9325));
        assert_eq!(g, to_u16(0.1675));
        assert_eq!(b, to_u16(0.1675));
    }

    // ── hsl_to_rgb ────────────────────────────────────────────────────────────

    #[test]
    fn hsl_to_rgb_green_sextant() {
        // hue=120° → pure green sextant (x, c, 0) + m
        let [r, g, b] = hsl_to_rgb(120.0, 1.0, 0.5);
        assert!((r - 0.0).abs() < 1e-5, "r={r}");
        assert!((g - 1.0).abs() < 1e-5, "g={g}");
        assert!((b - 0.0).abs() < 1e-5, "b={b}");
    }

    #[test]
    fn hsl_to_rgb_blue_sextant() {
        // hue=240° → pure blue
        let [r, g, b] = hsl_to_rgb(240.0, 1.0, 0.5);
        assert!((r - 0.0).abs() < 1e-5, "r={r}");
        assert!((g - 0.0).abs() < 1e-5, "g={g}");
        assert!((b - 1.0).abs() < 1e-5, "b={b}");
    }

    // ── to_u16 edge cases ─────────────────────────────────────────────────────

    #[test]
    fn to_u16_clamps_and_rounds() {
        assert_eq!(to_u16(0.0), 0);
        assert_eq!(to_u16(1.0), 65535);
        assert_eq!(to_u16(1.5), 65535); // clamped
        assert_eq!(to_u16(-0.5), 0);    // clamped
        assert_eq!(to_u16(0.5), 32768); // 0.5 * 65535 = 32767.5 → rounds to 32768
    }

    // ── glsl_fract ────────────────────────────────────────────────────────────

    #[test]
    fn glsl_fract_negative_is_positive() {
        // Rust's f32::fract(-0.3) = -0.3, glsl_fract(-0.3) = 0.7
        let f = glsl_fract(-0.3_f32);
        assert!(f >= 0.0 && f < 1.0, "glsl_fract(-0.3) = {f}");
        assert!((f - 0.7).abs() < 1e-6, "expected ~0.7, got {f}");
    }
}

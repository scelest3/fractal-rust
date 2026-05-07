//! PNG encoding for the fractal export pipeline.
//!
//! Calls into `crates/coloring` to apply LUT coloring to raw RGBA32F tile data,
//! then encodes the result as a 16-bit RGB PNG with full metadata.

use std::io::Write as _;

use coloring::{color_pixel, ColorParams};

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum EncodingError {
    Png(png::EncodingError),
    Io(std::io::Error),
}

impl From<png::EncodingError> for EncodingError {
    fn from(e: png::EncodingError) -> Self {
        EncodingError::Png(e)
    }
}

impl From<std::io::Error> for EncodingError {
    fn from(e: std::io::Error) -> Self {
        EncodingError::Io(e)
    }
}

// ── Public types ──────────────────────────────────────────────────────────────

pub struct PngMetadata<'a> {
    pub dpi: u32,
    pub cx: &'a str,
    pub cy: &'a str,
    pub zoom_exp: f64,
    pub fractal_kind: &'a str, // "mandelbrot" or "newton"
    pub max_iter: u32,
    pub newton_degree: Option<u32>,
    /// ISO 8601 UTC creation timestamp. `None` omits the tEXt chunk.
    /// On WASM pass `new Date().toISOString()` from the caller; on native
    /// use `SystemTime::now()` formatted by the caller.
    pub creation_time: Option<&'a str>,
}

// ── encode_png ────────────────────────────────────────────────────────────────

/// Encode a frame of raw RGBA32F fractal pixels as a 16-bit RGB PNG.
///
/// `pixels` is in GL readback order (bottom row first). The output PNG has rows
/// in top-down order (row 0 = top of image).
pub fn encode_png(
    pixels: &[f32],     // RGBA32F, width × height × 4, bottom-up from GL readback
    width: u32,
    height: u32,
    lut: &[f32],        // 4096 × 4 floats
    params: &ColorParams,
    fractal_kind: u32,  // 0 = Mandelbrot, 1 = Newton
    metadata: &PngMetadata,
) -> Result<Vec<u8>, EncodingError> {
    let entries = lut_entries(lut);
    let mut buf = Vec::new();

    // ── Encoder setup ─────────────────────────────────────────────────────────
    let mut encoder = png::Encoder::new(&mut buf, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);

    // pHYs: pixels per metre = DPI / 0.0254
    let ppm = (metadata.dpi as f64 / 0.0254).round() as u32;
    encoder.set_pixel_dims(Some(png::PixelDimensions {
        xppu: ppm,
        yppu: ppm,
        unit: png::Unit::Meter,
    }));

    // sRGB — perceptual rendering intent
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);

    // tEXt metadata
    encoder.add_text_chunk("Software".into(), "fractal-rust".into())?;
    if let Some(t) = metadata.creation_time {
        encoder.add_text_chunk("Creation Time".into(), t.into())?;
    }
    encoder.add_text_chunk("fractal:cx".into(), metadata.cx.into())?;
    encoder.add_text_chunk("fractal:cy".into(), metadata.cy.into())?;
    encoder.add_text_chunk("fractal:zoom_exp".into(), format!("{}", metadata.zoom_exp))?;
    encoder.add_text_chunk("fractal:kind".into(), metadata.fractal_kind.into())?;
    encoder.add_text_chunk("fractal:max_iter".into(), format!("{}", metadata.max_iter))?;
    if let Some(deg) = metadata.newton_degree {
        encoder.add_text_chunk("fractal:newton_degree".into(), format!("{deg}"))?;
    }

    // ── Pixel data ────────────────────────────────────────────────────────────
    let mut writer = encoder.write_header()?;

    // Stream one row at a time: GL bottom-up → PNG top-down.
    // Peak allocation is one row (width × 6 bytes) rather than the full image.
    let row_floats = (width * 4) as usize;
    let mut row_buf = vec![0u8; width as usize * 6];
    {
        let mut stream = writer.stream_writer()?;
        for row_idx in (0..height).rev() {
            let row_start = row_idx as usize * row_floats;
            for col in 0..width as usize {
                let pi = row_start + col * 4;
                let raw = [pixels[pi], pixels[pi + 1], pixels[pi + 2], pixels[pi + 3]];
                let [r, g, b] = color_pixel(raw, entries, params, fractal_kind);
                let base = col * 6;
                row_buf[base..base + 2].copy_from_slice(&r.to_be_bytes());
                row_buf[base + 2..base + 4].copy_from_slice(&g.to_be_bytes());
                row_buf[base + 4..base + 6].copy_from_slice(&b.to_be_bytes());
            }
            stream.write_all(&row_buf)?;
        }
        stream.flush()?;
    } // drops stream, flushing IEND back into writer
    drop(writer);
    Ok(buf)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Reinterpret a flat `&[f32]` LUT as `&[[f32; 4]]` entries without copying.
fn lut_entries(lut: &[f32]) -> &[[f32; 4]] {
    assert_eq!(lut.len() % 4, 0);
    // Safety: [f32; 4] has identical layout to 4 consecutive f32s.
    unsafe { std::slice::from_raw_parts(lut.as_ptr() as *const [f32; 4], lut.len() / 4) }
}


// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn white_lut() -> Vec<f32> {
        vec![1.0f32; 4096 * 4]
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

    fn meta(dpi: u32) -> PngMetadata<'static> {
        PngMetadata {
            dpi,
            cx: "-0.75",
            cy: "0.0",
            zoom_exp: 0.0,
            fractal_kind: "mandelbrot",
            max_iter: 256,
            newton_degree: None,
            creation_time: Some("2025-01-01T00:00:00Z"),
        }
    }

    /// Walk the PNG chunk stream and return the data bytes of the first chunk
    /// whose 4-byte type tag matches `tag`, or `None` if not found.
    fn find_chunk<'a>(png: &'a [u8], tag: &[u8; 4]) -> Option<&'a [u8]> {
        let mut i = 8usize; // skip 8-byte PNG signature
        while i + 8 <= png.len() {
            let len = u32::from_be_bytes(png[i..i + 4].try_into().unwrap()) as usize;
            if &png[i + 4..i + 8] == tag {
                return Some(&png[i + 8..i + 8 + len]);
            }
            i += 12 + len; // length(4) + tag(4) + data(len) + crc(4)
        }
        None
    }

    // ── pHYs ──────────────────────────────────────────────────────────────────

    #[test]
    fn phys_chunk_300_dpi() {
        // 300 / 0.0254 = 11811.02… → 11811
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32]; // 1×1 escaped pixel
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 0, &meta(300))
            .unwrap();
        let data = find_chunk(&out, b"pHYs").expect("pHYs chunk missing");
        assert_eq!(u32::from_be_bytes(data[0..4].try_into().unwrap()), 11811, "xppu");
        assert_eq!(u32::from_be_bytes(data[4..8].try_into().unwrap()), 11811, "yppu");
        assert_eq!(data[8], 1, "unit (1=metre)");
    }

    #[test]
    fn phys_chunk_72_dpi() {
        // 72 / 0.0254 = 2834.64… → 2835
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 0, &meta(72))
            .unwrap();
        let data = find_chunk(&out, b"pHYs").expect("pHYs chunk missing");
        assert_eq!(u32::from_be_bytes(data[0..4].try_into().unwrap()), 2835, "72 dpi xppu");
    }

    #[test]
    fn phys_chunk_150_dpi() {
        // 150 / 0.0254 = 5905.51… → 5906
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 0, &meta(150))
            .unwrap();
        let data = find_chunk(&out, b"pHYs").expect("pHYs chunk missing");
        assert_eq!(u32::from_be_bytes(data[0..4].try_into().unwrap()), 5906, "150 dpi xppu");
    }

    // ── tEXt ──────────────────────────────────────────────────────────────────

    #[test]
    fn text_metadata_keys_present() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 0, &meta(300))
            .unwrap();
        for key in &[
            b"fractal:cx" as &[u8],
            b"fractal:cy",
            b"fractal:zoom_exp",
            b"fractal:kind",
            b"fractal:max_iter",
            b"Software",
        ] {
            assert!(
                out.windows(key.len()).any(|w| w == *key),
                "missing tEXt key: {}",
                std::str::from_utf8(key).unwrap_or("?"),
            );
        }
    }

    #[test]
    fn text_newton_degree_present_when_some() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let m = PngMetadata { newton_degree: Some(3), ..meta(300) };
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 1, &m).unwrap();
        assert!(
            out.windows(b"fractal:newton_degree".len())
                .any(|w| w == b"fractal:newton_degree"),
            "newton_degree key missing",
        );
    }

    // ── IHDR ──────────────────────────────────────────────────────────────────

    #[test]
    fn ihdr_dimensions_and_depth() {
        // 4×2 image, all interior pixels
        let pixels = vec![0.0f32; 4 * 2 * 4];
        let out = encode_png(&pixels, 4, 2, &white_lut(), &default_params(), 0, &meta(300))
            .unwrap();
        // PNG layout: 8-byte sig, then IHDR chunk: 4-len + 4-tag + 13-data + 4-crc
        assert_eq!(u32::from_be_bytes(out[16..20].try_into().unwrap()), 4, "width");
        assert_eq!(u32::from_be_bytes(out[20..24].try_into().unwrap()), 2, "height");
        assert_eq!(out[24], 16, "bit depth");
        assert_eq!(out[25], 2, "color type (RGB)");
    }

    // ── Row orientation ───────────────────────────────────────────────────────

    #[test]
    fn row_orientation_flipped() {
        // 1×2 image in GL bottom-up order:
        //   row 0 (bottom of image): interior pixel → black
        //   row 1 (top of image):    escaped pixel  → white (white LUT, no trap)
        // PNG top-down: row 0 of output = top of image = white.
        let pixels: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0, // GL row 0 = bottom: interior → black
            0.0, 0.0, 0.0, 1.0, // GL row 1 = top:    escaped  → white
        ];
        let out =
            encode_png(&pixels, 1, 2, &white_lut(), &default_params(), 0, &meta(300)).unwrap();

        // Decode with the png crate and inspect pixel values.
        let decoder = png::Decoder::new(std::io::Cursor::new(&out));
        let mut reader = decoder.read_info().unwrap();
        let mut img = vec![0u8; reader.output_buffer_size().unwrap()];
        reader.next_frame(&mut img).unwrap();

        // RGB16: 6 bytes per pixel. Top pixel (PNG row 0) should be white.
        assert_eq!(&img[0..6], &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF], "top row = white");
        // Bottom pixel (PNG row 1) should be black.
        assert_eq!(&img[6..12], &[0x00, 0x00, 0x00, 0x00, 0x00, 0x00], "bottom row = black");
    }
}

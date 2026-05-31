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
    pub fractal_kind: &'a str, // "mandelbrot" | "newton" | "multibrot" | "julia"
    pub max_iter: u32,
    pub newton_degree: Option<u32>,
    /// ISO 8601 UTC creation timestamp. `None` omits the tEXt chunk.
    /// On WASM pass `new Date().toISOString()` from the caller; on native
    /// use `SystemTime::now()` formatted by the caller.
    pub creation_time: Option<&'a str>,
    /// Non-integer exponent for Multibrot (z^n) and Julia (z^n + c). `None` for Mandelbrot/Newton.
    pub exponent: Option<f64>,
    /// Real part of the Julia set's fixed parameter c. `None` for non-Julia kinds.
    pub julia_c_re: Option<f64>,
    /// Imaginary part of the Julia set's fixed parameter c. `None` for non-Julia kinds.
    pub julia_c_im: Option<f64>,
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
    if let Some(e) = metadata.exponent {
        encoder.add_text_chunk("fractal:exponent".into(), format!("{e}"))?;
    }
    if let Some(re) = metadata.julia_c_re {
        encoder.add_text_chunk("fractal:julia_c_re".into(), format!("{re}"))?;
    }
    if let Some(im) = metadata.julia_c_im {
        encoder.add_text_chunk("fractal:julia_c_im".into(), format!("{im}"))?;
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

// ── StreamingPngEncoder ───────────────────────────────────────────────────────

/// `Write` adaptor over a raw pointer to a `Vec<u8>`.
///
/// Used by `StreamingPngEncoder` to let `png::Writer` (and its owned
/// `StreamWriter`) write compressed PNG bytes into the encoder's output buffer
/// without requiring a borrow that would outlive the encoder.
///
/// # Safety
/// The caller must ensure the pointed-to `Vec<u8>` remains valid and exclusively
/// accessible for the lifetime of any live `VecPtrWriter`. This is upheld by
/// `StreamingPngEncoder`'s field layout and single-threaded WASM usage.
struct VecPtrWriter(*mut Vec<u8>);

impl std::io::Write for VecPtrWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        unsafe { (*self.0).write(buf) }
    }
    fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
}

/// Stateful PNG encoder for strip-by-strip export.
///
/// Typical usage:
/// ```ignore
/// let mut enc = StreamingPngEncoder::new(width, height, lut, params, kind, &meta)?;
/// for each strip (top-to-bottom order):
///     enc.append_strip(&strip_pixels, strip_height)?;
/// let bytes = enc.finish()?;
/// ```
///
/// `pixels` passed to `append_strip` must be in GL readback order (bottom row
/// first within the strip). Strips must be supplied in top-to-bottom image
/// order so that PNG rows are written sequentially.
///
/// # Design note
///
/// `stream` is a `png::StreamWriter<'static, VecPtrWriter>` produced by
/// `Writer::into_stream_writer()`. This consumes the `Writer` so the stream
/// owns the compressor and the `VecPtrWriter`; no borrow lifetime is involved.
/// `output` is a `Box<Vec<u8>>` (stable heap address). `VecPtrWriter` holds a
/// raw pointer to that Vec. Moving `StreamingPngEncoder` moves the `Box`
/// pointer but not the pointed-to Vec, so the raw pointer remains valid.
/// `stream` is declared before `output` to ensure stream drops first (flushing
/// IEND through `VecPtrWriter`), then `output` is freed.
pub struct StreamingPngEncoder {
    // Drop order: stream drops first (writes IEND to *output), then output is freed.
    stream: png::StreamWriter<'static, VecPtrWriter>,
    output: Box<Vec<u8>>,
    row_buf: Vec<u8>,
    row_floats: usize,
    lut: Vec<[f32; 4]>,
    params: ColorParams,
    fractal_kind: u32,
    width: u32,
}

impl StreamingPngEncoder {
    /// Begin a new streaming encode session and write the PNG header.
    pub fn new(
        width: u32,
        height: u32,
        lut: &[f32],
        params: ColorParams,
        fractal_kind: u32,
        metadata: &PngMetadata,
    ) -> Result<Self, EncodingError> {
        let lut_vec = lut_entries(lut).to_vec();
        let mut output = Box::new(Vec::<u8>::new());
        // SAFETY: `output` is heap-allocated (Box), so the Vec's address on the
        // heap is stable even when StreamingPngEncoder is moved. `VecPtrWriter`
        // holds a raw pointer to that allocation, not to the Box itself.
        let raw: *mut Vec<u8> = output.as_mut();

        let mut encoder = png::Encoder::new(VecPtrWriter(raw), width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Sixteen);
        let ppm = (metadata.dpi as f64 / 0.0254).round() as u32;
        encoder.set_pixel_dims(Some(png::PixelDimensions {
            xppu: ppm,
            yppu: ppm,
            unit: png::Unit::Meter,
        }));
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
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
        if let Some(e) = metadata.exponent {
            encoder.add_text_chunk("fractal:exponent".into(), format!("{e}"))?;
        }
        if let Some(re) = metadata.julia_c_re {
            encoder.add_text_chunk("fractal:julia_c_re".into(), format!("{re}"))?;
        }
        if let Some(im) = metadata.julia_c_im {
            encoder.add_text_chunk("fractal:julia_c_im".into(), format!("{im}"))?;
        }

        // into_stream_writer consumes the Writer and returns StreamWriter<'static, W>,
        // so no borrow lifetime escapes — stream can be stored alongside output.
        let stream = encoder.write_header()?.into_stream_writer()?;
        Ok(StreamingPngEncoder {
            stream,
            output,
            row_buf: vec![0u8; width as usize * 6],
            row_floats: (width * 4) as usize,
            lut: lut_vec,
            params,
            fractal_kind,
            width,
        })
    }

    /// Encode one rendered strip.
    ///
    /// `pixels` is `width × strip_height × 4` RGBA f32 in GL bottom-up order
    /// (row 0 = bottom of strip in image space). Strips must be supplied in
    /// top-to-bottom image order.
    pub fn append_strip(&mut self, pixels: &[f32], strip_height: u32) -> Result<(), EncodingError> {
        for row_idx in (0..strip_height as usize).rev() {
            let row_start = row_idx * self.row_floats;
            for col in 0..self.width as usize {
                let pi = row_start + col * 4;
                let raw = [pixels[pi], pixels[pi + 1], pixels[pi + 2], pixels[pi + 3]];
                let [r, g, b] = color_pixel(raw, &self.lut, &self.params, self.fractal_kind);
                let base = col * 6;
                self.row_buf[base..base + 2].copy_from_slice(&r.to_be_bytes());
                self.row_buf[base + 2..base + 4].copy_from_slice(&g.to_be_bytes());
                self.row_buf[base + 4..base + 6].copy_from_slice(&b.to_be_bytes());
            }
            self.stream.write_all(&self.row_buf)?;
        }
        Ok(())
    }

    /// Finalize the PNG (flush and write IEND) and return the encoded bytes.
    ///
    /// Returns `Err` if not all rows were written (`height` rows total expected).
    pub fn finish(self) -> Result<Vec<u8>, EncodingError> {
        let StreamingPngEncoder { stream, output, .. } = self;
        // finish() flushes the deflate compressor and writes IEND via VecPtrWriter.
        // output is still live at this point (destructor order is field declaration order,
        // but since we destructure, we control the drop order explicitly here).
        stream.finish()?;
        Ok(*output)
    }
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
            newton_color_mode: 0,
            newton_phase: 0.0,
            newton_smooth: false,
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
            exponent: None,
            julia_c_re: None,
            julia_c_im: None,
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

    #[test]
    fn text_exponent_present_for_multibrot() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let m = PngMetadata { fractal_kind: "multibrot", exponent: Some(3.0), ..meta(300) };
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 2, &m).unwrap();
        assert!(
            out.windows(b"fractal:exponent".len()).any(|w| w == b"fractal:exponent"),
            "fractal:exponent key missing for multibrot",
        );
    }

    #[test]
    fn text_julia_chunks_present_for_julia() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let m = PngMetadata {
            fractal_kind: "julia",
            exponent: Some(2.0),
            julia_c_re: Some(-0.7),
            julia_c_im: Some(0.27),
            ..meta(300)
        };
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 3, &m).unwrap();
        for key in &[b"fractal:exponent" as &[u8], b"fractal:julia_c_re", b"fractal:julia_c_im"] {
            assert!(
                out.windows(key.len()).any(|w| w == *key),
                "missing key: {}",
                std::str::from_utf8(key).unwrap_or("?"),
            );
        }
    }

    #[test]
    fn text_exponent_absent_for_mandelbrot() {
        let pixels = vec![0.0, 0.0, 0.0, 1.0f32];
        let out = encode_png(&pixels, 1, 1, &white_lut(), &default_params(), 0, &meta(300)).unwrap();
        assert!(
            !out.windows(b"fractal:exponent".len()).any(|w| w == b"fractal:exponent"),
            "fractal:exponent should be absent for mandelbrot",
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

    // ── StreamingPngEncoder ───────────────────────────────────────────────────

    #[test]
    fn streaming_single_strip_matches_encode_png() {
        // A 4×2 all-escaped image encoded as one strip should match encode_png.
        let pixels: Vec<f32> = (0..4 * 2 * 4).map(|i| if i % 4 == 3 { 1.0 } else { 0.0 }).collect();
        let lut = white_lut();
        let params = default_params();
        let m = meta(300);

        let expected = encode_png(&pixels, 4, 2, &lut, &params, 0, &m).unwrap();

        let mut enc = StreamingPngEncoder::new(4, 2, &lut, default_params(), 0, &m).unwrap();
        enc.append_strip(&pixels, 2).unwrap();
        let got = enc.finish().unwrap();

        // Decode both and compare pixel data (not raw bytes — multi-IDAT is valid but differs).
        let decode = |bytes: &[u8]| {
            let dec = png::Decoder::new(std::io::Cursor::new(bytes));
            let mut reader = dec.read_info().unwrap();
            let mut img = vec![0u8; reader.output_buffer_size().unwrap()];
            reader.next_frame(&mut img).unwrap();
            img
        };
        assert_eq!(decode(&expected), decode(&got), "pixel data mismatch");
    }

    #[test]
    fn streaming_two_strips_round_trip() {
        // 2×4 image split into two 2×2 strips, all escaped pixels.
        // Strip 0 = top 2 rows (image rows 0-1), strip 1 = bottom 2 rows (image rows 2-3).
        // Each strip's pixels are in GL bottom-up order.
        let all_pixels: Vec<f32> = (0..2 * 4 * 4).map(|i| if i % 4 == 3 { 1.0 } else { 0.0 }).collect();
        let lut = white_lut();
        let m = meta(300);

        let expected = encode_png(&all_pixels, 2, 4, &lut, &default_params(), 0, &m).unwrap();

        let mut enc = StreamingPngEncoder::new(2, 4, &lut, default_params(), 0, &m).unwrap();
        // strip 0: image rows 0-1 → GL pixels [rows 1,0 of image top half]
        enc.append_strip(&all_pixels[..2 * 2 * 4], 2).unwrap();
        // strip 1: image rows 2-3 → GL pixels [rows 3,2 of image bottom half]
        enc.append_strip(&all_pixels[2 * 2 * 4..], 2).unwrap();
        let got = enc.finish().unwrap();

        let decode = |bytes: &[u8]| {
            let dec = png::Decoder::new(std::io::Cursor::new(bytes));
            let mut reader = dec.read_info().unwrap();
            let mut img = vec![0u8; reader.output_buffer_size().unwrap()];
            reader.next_frame(&mut img).unwrap();
            img
        };
        assert_eq!(decode(&expected), decode(&got), "two-strip pixel data mismatch");
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

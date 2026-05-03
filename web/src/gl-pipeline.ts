/**
 * WebGL 2 pipeline — two-pass architecture.
 *
 * Pass 1 (tile pass): each 256×256 tile is uploaded as RGBA32F and drawn
 *   into the accumFBO (canvas-sized RGBA32F framebuffer). No coloring here —
 *   raw (smooth_t, orbit_min_r, angle, escaped) data is stored.
 *
 * Pass 2 (blit pass): a fullscreen quad reads the accumFBO and applies the
 *   LUT + uCycleLen to produce the final colored output on the canvas.
 *
 * This separation means palette changes only require a blit() call —
 * no tile re-upload or WASM re-render.
 */

const VERT_SRC = /* glsl */ `#version 300 es
out vec2 vUv;
void main() {
    float x = float(gl_VertexID & 1) * 2.0 - 1.0;
    float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    vUv = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
}`;

const TILE_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTile;
out vec4 fragColor;
void main() {
    fragColor = texture(uTile, vUv);
}`;

const BLIT_FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uAccum;
uniform sampler2D uLut;
uniform float uCycleLen;
uniform int   uColorMode;
uniform float uTrapRadius;
uniform float uTrapStrength;
uniform vec3  uTrapColor;
uniform vec3  uInteriorColor;
uniform vec3  uShadingColor;
uniform float uDistanceStrength;
uniform float uDistancePow;
uniform float uAngleStrength;
uniform float uPeriodStrength;
uniform float uPeriodCycleLen;
uniform float uDistWeight;
// Newton uniforms
uniform int   uFractalKind;   // 0 = Mandelbrot, 1 = Newton
uniform int   uNewtonDegree;
uniform vec3  uUnresolvedColor;
out vec4 fragColor;

vec3 hsl_to_rgb(float h, float s, float l) {
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float x = c * (1.0 - abs(mod(h / 60.0, 2.0) - 1.0));
    float m = l - c * 0.5;
    vec3 rgb;
    if      (h < 60.0)  rgb = vec3(c, x, 0.0);
    else if (h < 120.0) rgb = vec3(x, c, 0.0);
    else if (h < 180.0) rgb = vec3(0.0, c, x);
    else if (h < 240.0) rgb = vec3(0.0, x, c);
    else if (h < 300.0) rgb = vec3(x, 0.0, c);
    else                rgb = vec3(c, 0.0, x);
    return rgb + m;
}

void main() {
    vec4 data = texture(uAccum, vUv);

    // ── Newton branch ────────────────────────────────────────────────────────
    if (uFractalKind == 1) {
        if (data.a < 0.5) {
            fragColor = vec4(uUnresolvedColor, 1.0);
            return;
        }
        float hue = (data.b / float(uNewtonDegree)) * 360.0;
        // data.r = log_deriv: accumulated log|N'(z)|, negative deep in basin.
        // Rational sigmoid maps (-inf, 0] → [1, 0]: deep basin = bright, boundary = dark.
        float depth = max(0.0, -data.r);
        float dist_t = depth / (depth + 15.0);
        float lightness = mix(0.15, 0.90, dist_t);
        fragColor = vec4(hsl_to_rgb(hue, 0.85, lightness), 1.0);
        return;
    }

    // ── Mandelbrot / escape-time branch ──────────────────────────────────────
    if (data.a < 0.5) {
        float dist   = pow(clamp(data.g, 0.0, 1.0), uDistancePow);
        vec3  col    = mix(uInteriorColor, uShadingColor, uDistanceStrength * dist);
        float tAngle = fract((data.b + 3.14159265) / 6.28318530);
        col = mix(col, texture(uLut, vec2(tAngle, 0.5)).rgb, uAngleStrength);
        if (data.r >= 0.5 && uPeriodStrength > 0.0) {
            float tPer = fract(data.r / uPeriodCycleLen);
            col = mix(col, texture(uLut, vec2(tPer, 0.5)).rgb, uPeriodStrength);
        }
        fragColor = vec4(col, 1.0);
        return;
    }
    if (uColorMode == 0) {
        float t = fract((data.r + uDistWeight * data.b) / uCycleLen);
        vec3 escapeColor = texture(uLut, vec2(t, 0.5)).rgb;
        float blend = uTrapStrength * smoothstep(uTrapRadius, 0.0, data.g);
        fragColor = vec4(mix(escapeColor, uTrapColor, blend), 1.0);
    } else if (uColorMode == 2) {
        float t = fract(data.b / uCycleLen);
        fragColor = vec4(texture(uLut, vec2(t, 0.5)).rgb, 1.0);
    } else {
        fragColor = vec4(0.5, 0.5, 0.5, 1.0);
    }
}`;

export class GlPipeline {
  readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;

  private readonly tileProgram: WebGLProgram;
  private readonly blitProgram: WebGLProgram;
  private readonly tileTexture: WebGLTexture;   // 256×256 RGBA32F per tile
  private readonly lutTexture: WebGLTexture;    // 4096×1 RGBA32F LUT
  private readonly accumTexture: WebGLTexture;  // canvas-sized RGBA32F
  private readonly accumFBO: WebGLFramebuffer;

  // tile program uniforms
  private readonly uTileTex: WebGLUniformLocation;
  // blit program uniforms
  private readonly uAccum: WebGLUniformLocation;
  private readonly uLut: WebGLUniformLocation;
  private readonly uCycleLen: WebGLUniformLocation;
  private readonly uColorMode: WebGLUniformLocation;
  private readonly uTrapRadius:      WebGLUniformLocation;
  private readonly uTrapStrength:    WebGLUniformLocation;
  private readonly uTrapColor:       WebGLUniformLocation;
  private readonly uInteriorColor:    WebGLUniformLocation;
  private readonly uShadingColor:     WebGLUniformLocation;
  private readonly uDistanceStrength: WebGLUniformLocation;
  private readonly uDistancePow:      WebGLUniformLocation;
  private readonly uAngleStrength:    WebGLUniformLocation;
  private readonly uPeriodStrength:   WebGLUniformLocation;
  private readonly uPeriodCycleLen:   WebGLUniformLocation;
  private readonly uDistWeight:       WebGLUniformLocation;
  private readonly uFractalKind:      WebGLUniformLocation;
  private readonly uNewtonDegree:     WebGLUniformLocation;
  private readonly uUnresolvedColor:  WebGLUniformLocation;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;

    // Required for RGBA32F as FBO color attachment.
    gl.getExtension("EXT_color_buffer_float");
    // Enable float-texture linear filtering for the LUT if available.
    gl.getExtension("OES_texture_float_linear");

    // ── Programs ──────────────────────────────────────────────────────────────
    this.tileProgram = this.buildProgram(VERT_SRC, TILE_FRAG_SRC);
    this.blitProgram = this.buildProgram(VERT_SRC, BLIT_FRAG_SRC);

    // ── Textures ──────────────────────────────────────────────────────────────
    this.tileTexture = this.makeTexture(gl.NEAREST);   // filled per tile
    this.lutTexture = this.makeTexture(gl.LINEAR);     // filled by uploadLut()
    this.accumTexture = this.makeTexture(gl.NEAREST);  // canvas-sized accumulator
    gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.RGBA32F,
      canvas.width, canvas.height,
      0, gl.RGBA, gl.FLOAT, null,
    );

    // ── accumFBO ──────────────────────────────────────────────────────────────
    this.accumFBO = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D, this.accumTexture, 0,
    );
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`accumFBO incomplete (0x${status.toString(16)}) — EXT_color_buffer_float may be unsupported`);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // ── Blit program uniforms ─────────────────────────────────────────────────
    gl.useProgram(this.blitProgram);
    this.uAccum    = gl.getUniformLocation(this.blitProgram, "uAccum")!;
    this.uLut      = gl.getUniformLocation(this.blitProgram, "uLut")!;
    this.uCycleLen = gl.getUniformLocation(this.blitProgram, "uCycleLen")!;
    this.uColorMode = gl.getUniformLocation(this.blitProgram, "uColorMode")!;
    gl.uniform1f(this.uCycleLen, 64.0);
    gl.uniform1i(this.uColorMode, 0);
    this.uTrapRadius   = gl.getUniformLocation(this.blitProgram, "uTrapRadius")!;
    this.uTrapStrength = gl.getUniformLocation(this.blitProgram, "uTrapStrength")!;
    this.uTrapColor    = gl.getUniformLocation(this.blitProgram, "uTrapColor")!;
    gl.uniform1f(this.uTrapRadius,   0.5);
    gl.uniform1f(this.uTrapStrength, 0.0);
    gl.uniform3f(this.uTrapColor,    1.0, 1.0, 0.0); // default: yellow
    this.uInteriorColor    = gl.getUniformLocation(this.blitProgram, "uInteriorColor")!;
    this.uShadingColor     = gl.getUniformLocation(this.blitProgram, "uShadingColor")!;
    this.uDistanceStrength = gl.getUniformLocation(this.blitProgram, "uDistanceStrength")!;
    this.uDistancePow      = gl.getUniformLocation(this.blitProgram, "uDistancePow")!;
    this.uAngleStrength    = gl.getUniformLocation(this.blitProgram, "uAngleStrength")!;
    this.uPeriodStrength   = gl.getUniformLocation(this.blitProgram, "uPeriodStrength")!;
    this.uPeriodCycleLen   = gl.getUniformLocation(this.blitProgram, "uPeriodCycleLen")!;
    gl.uniform3f(this.uInteriorColor, 0.0, 0.0, 0.0); // default: black
    gl.uniform3f(this.uShadingColor,  1.0, 1.0, 1.0); // default: white
    gl.uniform1f(this.uDistanceStrength, 0.0);
    gl.uniform1f(this.uDistancePow,      1.0);
    gl.uniform1f(this.uAngleStrength,    0.0);
    gl.uniform1f(this.uPeriodStrength,   0.0);
    gl.uniform1f(this.uPeriodCycleLen,   8.0);
    this.uDistWeight = gl.getUniformLocation(this.blitProgram, "uDistWeight")!;
    gl.uniform1f(this.uDistWeight, 0.0);
    this.uFractalKind     = gl.getUniformLocation(this.blitProgram, "uFractalKind")!;
    this.uNewtonDegree    = gl.getUniformLocation(this.blitProgram, "uNewtonDegree")!;
    this.uUnresolvedColor = gl.getUniformLocation(this.blitProgram, "uUnresolvedColor")!;
    gl.uniform1i(this.uFractalKind,     0);
    gl.uniform1i(this.uNewtonDegree,    3);
    gl.uniform3f(this.uUnresolvedColor, 0.05, 0.05, 0.05); // near-black

    // ── Tile program uniform ──────────────────────────────────────────────────
    gl.useProgram(this.tileProgram);
    this.uTileTex = gl.getUniformLocation(this.tileProgram, "uTile")!;
  }

  /** Upload the 4096-entry RGBA32F LUT. Called once at startup and on palette change. */
  uploadLut(data: Float32Array): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4096, 1, 0, gl.RGBA, gl.FLOAT, data);
  }

  /** Clear the accumFBO (new generation — old tile data wiped). */
  clear(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.viewport(0, 0, gl.canvas.width as number, gl.canvas.height as number);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Upload a 256×256 tile from the SAB slot into the accumFBO at (tileX, tileY),
   * then blit the accumFBO to the canvas.
   */
  uploadAndDrawTile(
    tileSab: SharedArrayBuffer,
    slotIndex: number,
    tileX: number,
    tileY: number,
  ): void {
    const { gl } = this;
    const byteOffset = slotIndex * 256 * 256 * 4 * 4;
    const data = new Float32Array(new Float32Array(tileSab, byteOffset, 256 * 256 * 4));

    // Upload tile to tileTexture with Y-flip so row 0 = top of tile in texture.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.bindTexture(gl.TEXTURE_2D, this.tileTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, 256, 0, gl.RGBA, gl.FLOAT, data);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

    // Draw tile into accumFBO at the correct canvas region.
    const canvasH = gl.canvas.height as number;
    const x = tileX * 256;
    const y = canvasH - (tileY + 1) * 256; // WebGL Y is bottom-up

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.viewport(x, y, 256, 256);
    gl.scissor(x, y, 256, 256);
    gl.enable(gl.SCISSOR_TEST);

    gl.useProgram(this.tileProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tileTexture);
    gl.uniform1i(this.uTileTex, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disable(gl.SCISSOR_TEST);

    this.blit();
  }

  /**
   * Blit the accumFBO to the canvas, applying the LUT.
   * Called after each tile and will be called directly by session.ts on palette change.
   */
  blit(): void {
    const { gl } = this;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.canvas.width as number, gl.canvas.height as number);
    gl.useProgram(this.blitProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.accumTexture);
    gl.uniform1i(this.uAccum, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.uniform1i(this.uLut, 1);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  /**
   * Resize the accumFBO to match a new canvas size.
   * Call after changing canvas.width/height; triggers a clear.
   */
  resize(width: number, height: number): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.accumTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.accumFBO);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Update the band-width uniform. Call blit() afterwards to see the change. */
  setCycleLen(value: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uCycleLen, value);
  }

  setTrapRadius(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uTrapRadius, v);
  }

  setTrapStrength(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uTrapStrength, v);
  }

  setTrapColor(r: number, g: number, b: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform3f(this.uTrapColor, r, g, b);
  }

  setInteriorColor(r: number, g: number, b: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform3f(this.uInteriorColor, r, g, b);
  }

  setShadingColor(r: number, g: number, b: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform3f(this.uShadingColor, r, g, b);
  }

  setDistanceStrength(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uDistanceStrength, v);
  }

  setDistancePow(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uDistancePow, v);
  }

  setAngleStrength(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uAngleStrength, v);
  }

  setPeriodStrength(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uPeriodStrength, v);
  }

  setPeriodCycleLen(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uPeriodCycleLen, v);
  }

  setColorMode(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1i(this.uColorMode, v);
  }

  setDistWeight(v: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1f(this.uDistWeight, v);
  }

  /** Switch between Mandelbrot (0) and Newton (1) coloring. */
  setFractalKind(kind: 0 | 1): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1i(this.uFractalKind, kind);
  }

  /** Number of roots for Newton coloring — controls hue step size. */
  setNewtonDegree(degree: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform1i(this.uNewtonDegree, degree);
  }

  /** Color for Newton pixels that did not converge (diverged or max-iter). */
  setUnresolvedColor(r: number, g: number, b: number): void {
    const { gl } = this;
    gl.useProgram(this.blitProgram);
    gl.uniform3f(this.uUnresolvedColor, r, g, b);
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  private buildProgram(vertSrc: string, fragSrc: string): WebGLProgram {
    const { gl } = this;
    const vert = this.compileShader(gl.VERTEX_SHADER, vertSrc);
    const frag = this.compileShader(gl.FRAGMENT_SHADER, fragSrc);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, vert);
    gl.attachShader(prog, frag);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Shader link failed: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(vert);
    gl.deleteShader(frag);
    return prog;
  }

  private compileShader(type: number, src: string): WebGLShader {
    const { gl } = this;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(`Shader compile failed: ${gl.getShaderInfoLog(shader)}`);
    }
    return shader;
  }

  private makeTexture(filter: number): WebGLTexture {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }
}

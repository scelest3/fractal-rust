/**
 * WebGL 2 pipeline — Phase 1 smooth-coloring pass.
 *
 * Two textures:
 *   uTile  — 256×256 RGBA32F: (smooth_t, orbit_min_r, angle, escaped) per pixel
 *   uLut   — 4096×1  RGBA32F: pre-built color LUT from build_lut()
 *
 * Fragment shader indexes the LUT by fract(smooth_t / CYCLE_LEN) for smooth
 * cyclic coloring. Interior pixels (escaped == 0) are drawn black.
 */

const VERT_SRC = /* glsl */ `#version 300 es
out vec2 vUv;
void main() {
    float x = float(gl_VertexID & 1) * 2.0 - 1.0;
    float y = float((gl_VertexID >> 1) & 1) * 2.0 - 1.0;
    gl_Position = vec4(x, y, 0.0, 1.0);
    vUv = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
}`;

const FRAG_SRC = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uTile;
uniform sampler2D uLut;
uniform float uCycleLen;
out vec4 fragColor;
void main() {
    vec4 data = texture(uTile, vUv);
    float escaped = data.a;
    if (escaped < 0.5) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float t = fract(data.r / uCycleLen);
        fragColor = vec4(texture(uLut, vec2(t, 0.5)).rgb, 1.0);
    }
}`;

export class GlPipeline {
  readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly tileTexture: WebGLTexture;
  private readonly lutTexture: WebGLTexture;
  private readonly uTile: WebGLUniformLocation;
  private readonly uLut: WebGLUniformLocation;
  private readonly uCycleLen: WebGLUniformLocation;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 not available");
    this.gl = gl;
    this.program = this.buildProgram(VERT_SRC, FRAG_SRC);
    this.tileTexture = this.makeTexture(gl.LINEAR);
    this.lutTexture = this.makeTexture(gl.LINEAR);

    gl.useProgram(this.program);
    this.uTile = gl.getUniformLocation(this.program, "uTile")!;
    this.uLut = gl.getUniformLocation(this.program, "uLut")!;
    this.uCycleLen = gl.getUniformLocation(this.program, "uCycleLen")!;
    gl.uniform1f(this.uCycleLen, 64.0);
  }

  /** Upload the 4096-entry RGBA32F LUT, called once after worker init. */
  uploadLut(data: Float32Array): void {
    const { gl } = this;
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 4096, 1, 0, gl.RGBA, gl.FLOAT, data);
  }

  /** Clear the canvas to black. Called at the start of each new dispatch generation. */
  clear(): void {
    const { gl } = this;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  /**
   * Upload a 256×256 tile from the SAB slot and draw it at (tileX, tileY)
   * in tile-grid coordinates (tileY=0 is the top of the canvas).
   */
  uploadAndDrawTile(
    tileSab: SharedArrayBuffer,
    slotIndex: number,
    tileX: number,
    tileY: number,
  ): void {
    const { gl } = this;
    const byteOffset = slotIndex * 256 * 256 * 4 * 4;
    const data = new Float32Array(tileSab, byteOffset, 256 * 256 * 4);

    // Upload tile data.
    gl.bindTexture(gl.TEXTURE_2D, this.tileTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, 256, 0, gl.RGBA, gl.FLOAT, data);

    // WebGL Y is bottom-up; tileY=0 is the top row, so flip.
    const canvasH = gl.canvas.height as number;
    const x = tileX * 256;
    const y = canvasH - (tileY + 1) * 256;

    gl.viewport(x, y, 256, 256);
    gl.scissor(x, y, 256, 256);
    gl.enable(gl.SCISSOR_TEST);

    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tileTexture);
    gl.uniform1i(this.uTile, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.lutTexture);
    gl.uniform1i(this.uLut, 1);

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disable(gl.SCISSOR_TEST);
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

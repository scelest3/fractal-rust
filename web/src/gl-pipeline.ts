/**
 * WebGL 2 pipeline — Phase 0 hello-world.
 *
 * Uploads a 256×256 RGBA f32 tile from a SharedArrayBuffer directly into a
 * GPU texture (zero copy from SAB to GPU), then renders it to the canvas via
 * a fullscreen quad.
 *
 * Phase 1+ will add the accumFBO, LUT texture, and multi-pass coloring.
 */

// ── Shaders ────────────────────────────────────────────────────────────────

// Fullscreen quad via gl_VertexID — no vertex buffers needed.
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
out vec4 fragColor;
void main() {
    fragColor = texture(uTile, vUv);
}`;

// ── GlPipeline ─────────────────────────────────────────────────────────────

export class GlPipeline {
  private readonly gl: WebGL2RenderingContext;
  private readonly program: WebGLProgram;
  private readonly texture: WebGLTexture;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext("webgl2");
    if (!gl) throw new Error("WebGL2 not available in this browser");
    this.gl = gl;
    this.program = this.createProgram(VERT_SRC, FRAG_SRC);
    this.texture = this.createTexture();
  }

  /**
   * Upload one 256×256 RGBA f32 tile from a SharedArrayBuffer into the GPU
   * texture. Reads directly from the SAB — zero copy from SAB to GPU.
   *
   * @param tileSab  The shared tile ring buffer.
   * @param slotIndex  Which ring slot to read (slot 0 = offset 0).
   * @param tileSlotBytes  Bytes per slot (constant: 256×256×4×4 = 1 048 576).
   */
  uploadTile(tileSab: SharedArrayBuffer, slotIndex: number, tileSlotBytes: number): void {
    const { gl } = this;
    const byteOffset = slotIndex * tileSlotBytes;
    const data = new Float32Array(tileSab, byteOffset, 256 * 256 * 4);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 256, 256, 0, gl.RGBA, gl.FLOAT, data);
  }

  /** Render the tile texture to the full canvas. */
  draw(): void {
    const { gl } = this;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.uniform1i(gl.getUniformLocation(this.program, "uTile"), 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  // ── Private helpers ──────────────────────────────────────────────────────

  private createProgram(vertSrc: string, fragSrc: string): WebGLProgram {
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

  private createTexture(): WebGLTexture {
    const { gl } = this;
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // NEAREST filtering: no OES_texture_float_linear extension required.
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }
}

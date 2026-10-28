export type LiquidRenderer = "webgl" | "canvas2d" | "static";

const DROP_COUNT = 12;
const EXCLUDED_TARGETS = ".crystal-stage, input, textarea, select";
const VERTEX = "attribute vec2 position;void main(){gl_Position=vec4(position,0.,1.);}";
const FRAGMENT = `precision mediump float;
uniform vec2 resolution;
uniform float time;
uniform vec4 drops[12];
void main(){
  vec2 uv=vec2(gl_FragCoord.x/resolution.x,1.-gl_FragCoord.y/resolution.y);
  float aspect=resolution.x/resolution.y;
  vec2 q=uv*vec2(aspect,1.);
  vec2 bend=vec2(0.);float ripple=0.;float energy=0.;
  for(int i=0;i<12;i++){
    float age=time-drops[i].z;
    if(age<0.||age>7.||drops[i].w<=0.)continue;
    vec2 delta=q-drops[i].xy*vec2(aspect,1.);
    float r=length(delta);float fade=exp(-age*.65)*drops[i].w;
    float edge=(r-age*.105)*12.;float ring=exp(-edge*edge);
    float w=sin(r*42.-age*6.5)*exp(-r*3.8)*fade;
    bend+=delta/(r+.08)*w*.13;
    ripple+=w*ring;energy+=ring*fade*.035;
  }
  vec2 p=(q+bend)*3.6;
  p+=.27*sin(p.yx*1.45+vec2(time*.13,-time*.11));
  float f=sin(p.x+sin(p.y*.87-time*.13))+cos(p.y*1.16+sin(p.x*.72+time*.11));
  float g=sin(p.x*1.41-p.y*.45+sin(p.y+time*.08));
  float line=.5+.5*sin(f*4.1+g*.35+ripple);line*=line;line*=line;line*=line;line*=line;
  float line2=.5+.5*cos(f*2.8-g*.55);line2*=line2;line2*=line2;line2*=line2;line2=line2*line2*line2;
  float pool=.5+.5*sin(f*1.08);
  float focus=.35+.65*smoothstep(.15,.85,uv.x);
  vec3 color=vec3(.021,.043,.062);
  color+=vec3(.015,.08,.105)*pool*focus;
  color+=vec3(.04,.29,.29)*line*focus*.46;
  color+=vec3(.06,.13,.36)*line2*.29;
  float pink=smoothstep(.25,.95,uv.y)*(1.-smoothstep(.12,.8,uv.x));
  color+=vec3(.16,.025,.13)*line2*pink*.35;
  color+=vec3(.13,.65,.57)*energy+vec3(.025,.12,.14)*max(ripple,0.);
  vec2 edge=uv-vec2(.5);color*=1.-.2*dot(edge,edge);
  gl_FragColor=vec4(color,1.);
}`;

function smoothstep(low: number, high: number, value: number) {
  const x = Math.max(0, Math.min(1, (value - low) / (high - low)));
  return x * x * (3 - 2 * x);
}

/** Owns only the decorative canvases, never the crystal viewers' contexts. */
export function createLiquidSurface(
  canvas: HTMLCanvasElement,
  fallbackCanvas: HTMLCanvasElement,
  onRenderer: (renderer: LiquidRenderer) => void,
) {
  let enabled = false;
  let disposed = false;
  let frame = 0;
  let last = 0;
  let time = 0;
  let cursor = 0;
  let lastSplat = 0;
  let lastPoint: { x: number; y: number } | null = null;
  let renderer: LiquidRenderer = "static";
  let gl: WebGLRenderingContext | null = null;
  let program: WebGLProgram | null = null;
  let buffer: WebGLBuffer | null = null;
  const shaders: WebGLShader[] = [];
  let locations: { resolution: WebGLUniformLocation | null; time: WebGLUniformLocation | null; drops: WebGLUniformLocation | null } | null = null;
  let fallbackContext: CanvasRenderingContext2D | null = null;
  let fallbackImage: ImageData | null = null;
  let fallbackInterval = 1000 / 15;
  const viewerPointers = new Set<number>();
  const trails = new Float32Array(DROP_COUNT * 4);

  function clearTrails() {
    trails.fill(0);
    for (let i = 0; i < DROP_COUNT; i++) trails[i * 4 + 2] = -100;
    lastPoint = null;
  }

  function releaseGL() {
    if (gl) {
      if (buffer) gl.deleteBuffer(buffer);
      if (program) gl.deleteProgram(program);
      for (const shader of shaders) gl.deleteShader(shader);
    }
    shaders.length = 0;
    buffer = null;
    program = null;
    locations = null;
  }

  function startFallback() {
    releaseGL();
    try {
      fallbackContext = fallbackCanvas.getContext("2d", { alpha: false });
    } catch {
      fallbackContext = null;
    }
    renderer = fallbackContext ? "canvas2d" : "static";
    onRenderer(renderer);
  }

  function compile(type: number, source: string) {
    const shader = gl!.createShader(type);
    if (!shader) throw new Error("Liquid shader unavailable");
    shaders.push(shader);
    gl!.shaderSource(shader, source);
    gl!.compileShader(shader);
    if (!gl!.getShaderParameter(shader, gl!.COMPILE_STATUS)) throw new Error("Liquid shader unsupported");
    return shader;
  }

  try {
    gl = canvas.getContext("webgl", { alpha: false, antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false, powerPreference: "low-power" });
    if (!gl || gl.isContextLost()) throw new Error("Liquid WebGL unavailable");
    program = gl.createProgram();
    if (!program) throw new Error("Liquid program unavailable");
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("Liquid program unsupported");
    gl.useProgram(program);
    buffer = gl.createBuffer();
    if (!buffer) throw new Error("Liquid buffer unavailable");
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(position);
    gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
    locations = { resolution: gl.getUniformLocation(program, "resolution"), time: gl.getUniformLocation(program, "time"), drops: gl.getUniformLocation(program, "drops") };
    renderer = "webgl";
    onRenderer(renderer);
  } catch {
    startFallback();
  }

  function drawFallback() {
    if (!fallbackContext || !fallbackImage) return;
    const width = fallbackCanvas.width;
    const height = fallbackCanvas.height;
    const aspect = window.innerWidth / Math.max(1, window.innerHeight);
    const data = fallbackImage.data;
    const active: number[][] = [];
    for (let i = 0; i < DROP_COUNT; i++) {
      const age = time - trails[i * 4 + 2];
      if (age >= 0 && age < 7 && trails[i * 4 + 3] > 0) {
        active.push([trails[i * 4] * aspect, trails[i * 4 + 1], age, Math.exp(-age * .65) * trails[i * 4 + 3]]);
      }
    }
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const ux = x / width, uy = y / height, qx = ux * aspect;
      let bx = 0, by = 0, ripple = 0, energy = 0;
      for (const drop of active) {
        const dx = qx - drop[0], dy = uy - drop[1], r = Math.hypot(dx, dy);
        const edge = (r - drop[2] * .105) * 12, ring = Math.exp(-edge * edge);
        const wave = Math.sin(r * 42 - drop[2] * 6.5) * Math.exp(-r * 3.8) * drop[3];
        bx += dx / (r + .08) * wave * .13;
        by += dy / (r + .08) * wave * .13;
        ripple += wave * ring;
        energy += ring * drop[3] * .035;
      }
      let px = (qx + bx) * 3.6, py = (uy + by) * 3.6;
      const originalX = px;
      px += .27 * Math.sin(py * 1.45 + time * .13);
      py += .27 * Math.sin(originalX * 1.45 - time * .11);
      const f = Math.sin(px + Math.sin(py * .87 - time * .13)) + Math.cos(py * 1.16 + Math.sin(px * .72 + time * .11));
      const g = Math.sin(px * 1.41 - py * .45 + Math.sin(py + time * .08));
      let line = .5 + .5 * Math.sin(f * 4.1 + g * .35 + ripple);
      line *= line; line *= line; line *= line; line *= line;
      let line2 = .5 + .5 * Math.cos(f * 2.8 - g * .55);
      line2 *= line2; line2 *= line2; line2 *= line2; line2 = line2 * line2 * line2;
      const pool = .5 + .5 * Math.sin(f * 1.08), focus = .35 + .65 * smoothstep(.15, .85, ux);
      const pink = smoothstep(.25, .95, uy) * (1 - smoothstep(.12, .8, ux));
      const vignette = 1 - .2 * ((ux - .5) ** 2 + (uy - .5) ** 2), lit = Math.max(ripple, 0);
      const i = (y * width + x) * 4;
      data[i] = 255 * vignette * (.021 + .015 * pool * focus + .04 * line * focus * .46 + .06 * line2 * .29 + .16 * line2 * pink * .35 + .13 * energy + .025 * lit);
      data[i + 1] = 255 * vignette * (.043 + .08 * pool * focus + .29 * line * focus * .46 + .13 * line2 * .29 + .025 * line2 * pink * .35 + .65 * energy + .12 * lit);
      data[i + 2] = 255 * vignette * (.062 + .105 * pool * focus + .29 * line * focus * .46 + .36 * line2 * .29 + .13 * line2 * pink * .35 + .57 * energy + .14 * lit);
      data[i + 3] = 255;
    }
    fallbackContext.putImageData(fallbackImage, 0, 0);
  }

  function draw() {
    if (disposed || document.hidden) return;
    if (gl && program && locations) {
      gl.uniform2f(locations.resolution, canvas.width, canvas.height);
      gl.uniform1f(locations.time, time);
      gl.uniform4fv(locations.drops, trails);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    } else {
      drawFallback();
    }
  }

  function resize() {
    const width = Math.max(1, window.innerWidth), height = Math.max(1, window.innerHeight);
    const scale = Math.min(.5, 480 / width, 360 / height);
    canvas.width = Math.max(1, Math.round(width * scale));
    canvas.height = Math.max(1, Math.round(height * scale));
    if (gl && program) gl.viewport(0, 0, canvas.width, canvas.height);
    if (fallbackContext) {
      const mobile = width < 700;
      const ratio = Math.min((mobile ? 120 : 180) / width, (mobile ? 90 : 130) / height);
      fallbackCanvas.width = Math.max(1, Math.round(width * ratio));
      fallbackCanvas.height = Math.max(1, Math.round(height * ratio));
      fallbackImage = fallbackContext.createImageData(fallbackCanvas.width, fallbackCanvas.height);
      fallbackInterval = 1000 / (mobile ? 12 : 15);
    }
    draw();
  }

  function canAnimate() {
    return !disposed && enabled && !document.hidden && viewerPointers.size === 0 && renderer !== "static";
  }

  function tick(now: number) {
    frame = 0;
    if (!canAnimate()) return;
    if (now - last >= (renderer === "webgl" ? 1000 / 30 : fallbackInterval)) {
      time += Math.min((now - last) / 1000, .1);
      last = now;
      draw();
    }
    frame = requestAnimationFrame(tick);
  }

  function updateAnimation() {
    cancelAnimationFrame(frame);
    frame = 0;
    last = performance.now();
    lastPoint = null;
    if (canAnimate()) frame = requestAnimationFrame(tick);
  }

  function splat(x: number, y: number, strength: number) {
    const i = (cursor++ % DROP_COUNT) * 4;
    trails[i] = x / Math.max(1, window.innerWidth);
    trails[i + 1] = y / Math.max(1, window.innerHeight);
    trails[i + 2] = time;
    trails[i + 3] = strength;
  }

  function targetElement(event: PointerEvent) {
    return event.target instanceof Element ? event.target : null;
  }

  function pointerMove(event: PointerEvent) {
    if (!canAnimate() || event.pointerType !== "mouse" || targetElement(event)?.closest(EXCLUDED_TARGETS)) {
      lastPoint = null;
      return;
    }
    const now = performance.now();
    if (now - lastSplat < 55) return;
    const speed = lastPoint ? Math.hypot(event.clientX - lastPoint.x, event.clientY - lastPoint.y) : 10;
    if (speed > 3) {
      splat(event.clientX, event.clientY, Math.min(.9, .18 + speed / 130));
      lastPoint = { x: event.clientX, y: event.clientY };
      lastSplat = now;
    }
  }

  function pointerDown(event: PointerEvent) {
    const target = targetElement(event);
    if (target?.closest(".crystal-stage")) {
      viewerPointers.add(event.pointerId);
      updateAnimation();
    } else if (canAnimate() && event.pointerType === "mouse" && !target?.closest(EXCLUDED_TARGETS)) {
      splat(event.clientX, event.clientY, .9);
    }
  }

  function pointerUp(event: PointerEvent) {
    if (viewerPointers.delete(event.pointerId)) updateAnimation();
  }

  function blur() {
    viewerPointers.clear();
    updateAnimation();
  }

  function visibilityChanged() {
    viewerPointers.clear();
    updateAnimation();
    if (!enabled) draw();
  }

  function contextLost(event: Event) {
    event.preventDefault();
    if (disposed) return;
    startFallback();
    resize();
    updateAnimation();
  }

  clearTrails();
  document.addEventListener("pointermove", pointerMove, { passive: true });
  document.addEventListener("pointerdown", pointerDown, { passive: true });
  window.addEventListener("pointerup", pointerUp, { passive: true });
  window.addEventListener("pointercancel", pointerUp, { passive: true });
  window.addEventListener("blur", blur);
  document.addEventListener("visibilitychange", visibilityChanged);
  window.addEventListener("resize", resize, { passive: true });
  canvas.addEventListener("webglcontextlost", contextLost);
  resize();

  return {
    setEnabled(value: boolean) {
      enabled = value;
      if (!enabled) {
        clearTrails();
        draw();
      }
      updateAnimation();
    },
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      document.removeEventListener("pointermove", pointerMove);
      document.removeEventListener("pointerdown", pointerDown);
      window.removeEventListener("pointerup", pointerUp);
      window.removeEventListener("pointercancel", pointerUp);
      window.removeEventListener("blur", blur);
      document.removeEventListener("visibilitychange", visibilityChanged);
      window.removeEventListener("resize", resize);
      canvas.removeEventListener("webglcontextlost", contextLost);
      releaseGL();
      // Do not force context loss: React StrictMode can reuse this canvas.
      gl = null;
      fallbackContext = null;
      fallbackImage = null;
      viewerPointers.clear();
    },
  };
}

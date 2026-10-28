"use client";
import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Upload,
  Download,
  RotateCcw,
  Play,
  Pause,
  X,
  ChevronDown,
  Check,
  Loader2,
  FlaskConical,
  Braces,
  Sparkles,
  Atom,
} from "lucide-react";
declare global {
  interface Window {
    $3Dmol: any;
  }
}
type VData = { atoms: any[]; lattice: number[][] };
type Preview = { cif: string; summary: any; viewer: VData };
type Result = {
  operation: string;
  input_summary: any;
  output_summary: any;
  output_viewer: VData | null;
  input_cif: string;
  output_cif: string | null;
  latent: number[];
  optimized_latent: number[] | null;
  initial_score: number;
  final_score: number;
  history: { step: number; score: number }[];
  metadata: any;
};
async function api(path: string, body?: unknown) {
  const r = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  let data: any;
  try {
    data = await r.json();
  } catch {
    throw Error("The model service is unavailable. Please try again shortly.");
  }
  if (!r.ok) throw Error(data.error || "The request could not be completed.");
  return data;
}
let molPromise: Promise<void> | null = null;
function loadMol() {
  if (!molPromise)
    molPromise = new Promise((resolve, reject) => {
      if (window.$3Dmol) {
        resolve();
        return;
      }
      const s = document.createElement("script");
      s.src = "/3Dmol-min.js";
      s.onload = () => resolve();
      s.onerror = () => {
        molPromise = null;
        reject(Error("The crystal viewer could not load."));
      };
      document.head.appendChild(s);
    });
  return molPromise;
}
function Mark({ large = false }: { large?: boolean }) {
  return (
    <svg
      className={large ? "mark large" : "mark"}
      viewBox="0 0 64 64"
      fill="none"
      aria-hidden="true"
    >
      <path d="M32 4 56.2 18v28L32 60 7.8 46V18L32 4Z" stroke="currentColor" />
      <path
        d="m32 14 15.6 9v18L32 50l-15.6-9V23L32 14Zm0-10v56M7.8 18l48.4 28M56.2 18 7.8 46"
        stroke="currentColor"
        opacity=".5"
      />
      <circle cx="32" cy="32" r="5" fill="currentColor" />
      <circle cx="32" cy="4" r="2" fill="currentColor" />
      <circle cx="56.2" cy="46" r="2" fill="currentColor" />
      <circle cx="7.8" cy="46" r="2" fill="currentColor" />
    </svg>
  );
}
function Formula({ text }: { text: string }) {
  return (
    <>
      {text
        .split(/(\d+)/)
        .map((p, i) =>
          /^\d+$/.test(p) ? <sub key={i}>{p}</sub> : <span key={i}>{p}</span>,
        )}
    </>
  );
}
function Crystal({
  data,
  busy = false,
}: {
  data: VData | null;
  busy?: boolean;
}) {
  const host = useRef<HTMLDivElement>(null),
    viewer = useRef<any>(null),
    initial = useRef<any>(null);
  const [spin, setSpin] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    let alive = true;
    let observer: ResizeObserver | undefined;
    setSpin(false);
    setError("");
    if (!data || !host.current) return;
    loadMol()
      .then(() => {
        if (!alive || !host.current) return;
        host.current.innerHTML = "";
        const v = window.$3Dmol.createViewer(host.current, {
          backgroundColor: "#0a111a",
          antialias: true,
        });
        viewer.current = v;
        const m = v.addModel();
        m.addAtoms(data.atoms);
        v.setStyle({}, { sphere: { scale: 0.35, colorscheme: "Jmol" } });
        const vertices = Array.from({ length: 8 }, (_, bits) => {
          const a = [0, 0, 0];
          for (let k = 0; k < 3; k++)
            if (bits & (1 << k))
              for (let j = 0; j < 3; j++) a[j] += data.lattice[k][j];
          return { x: a[0], y: a[1], z: a[2] };
        });
        for (let b = 0; b < 8; b++)
          for (let k = 0; k < 3; k++)
            if (!(b & (1 << k)))
              v.addLine({
                start: vertices[b],
                end: vertices[b | (1 << k)],
                color: b === 0 ? "#5ce3da" : "#375464",
                linewidth: 1.4,
              });
        ["a", "b", "c"].forEach((a, i) =>
          v.addLabel(a, {
            position: vertices[1 << i],
            fontColor: "#73a5b3",
            fontSize: 11,
            showBackground: false,
            inFront: true,
          }),
        );
        const bounds = v.addModel();
        bounds.addAtoms(vertices.map((p) => ({ ...p, elem: "X" })));
        bounds.setStyle({}, {});
        let label: any;
        v.setHoverable(
          {},
          true,
          (a: any) => {
            label = v.addLabel(`${a.elem} · site ${a.serial}`, {
              position: a,
              fontColor: "#e5eff6",
              backgroundColor: "#152632",
              fontSize: 12,
            });
            v.render();
          },
          () => {
            if (label) v.removeLabel(label);
            v.render();
          },
        );
        v.zoomTo();
        v.rotate(18, "x");
        v.rotate(-20, "y");
        v.zoom(1.1);
        v.render();
        initial.current = v.getView();
        observer = new ResizeObserver(() => {
          v.resize();
          v.render();
        });
        observer.observe(host.current);
      })
      .catch((e) => alive && setError(e.message));
    return () => {
      alive = false;
      observer?.disconnect();
      viewer.current?.spin(false);
      viewer.current?.clear();
      viewer.current = null;
    };
  }, [data]);
  return (
    <div className="crystal-stage">
      <div className="corner tl" />
      <div className="corner tr" />
      <div className="corner bl" />
      <div className="corner br" />
      {data ? (
        <>
          <div className="mol-canvas" ref={host} />
          <div className="canvas-tag">
            <span />
            UNIT CELL
          </div>
          <div className="viewer-controls">
            <span>Drag to rotate · Scroll to zoom</span>
            <button
              aria-label="Reset view"
              onClick={() => {
                viewer.current?.setView(initial.current);
                viewer.current?.render();
              }}
            >
              <RotateCcw size={14} />
            </button>
            <button
              aria-label={spin ? "Stop rotation" : "Rotate crystal"}
              onClick={() => {
                viewer.current?.spin(!spin ? "y" : false, 0.5, true);
                setSpin(!spin);
              }}
            >
              {spin ? <Pause size={14} /> : <Play size={14} />}
            </button>
          </div>
          <div className="element-legend">
            {[...new Set(data.atoms.map((a) => a.elem))].map((e) => (
              <span key={e}>{e}</span>
            ))}
          </div>
        </>
      ) : (
        <div className={`empty-crystal ${busy ? "working" : ""}`}>
          <div className="orbit one" />
          <div className="orbit two" />
          <Mark large />
          <strong>
            {busy ? "Finding a new form" : "A possibility, not yet formed"}
          </strong>
          <p>
            {busy
              ? "Exploring the latent space of your crystal."
              : "Your generated candidate will appear here."}
          </p>
        </div>
      )}
      {error && (
        <div className="viewer-error">{error} Downloads remain available.</div>
      )}
    </div>
  );
}
function save(name: string, data: BlobPart, mime = "application/json") {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function npy(values: number[]) {
  let h = `{'descr': '<f4', 'fortran_order': False, 'shape': (${values.length},), }`;
  h += " ".repeat((16 - ((10 + h.length + 1) % 16)) % 16) + "\n";
  const b = new ArrayBuffer(10 + h.length + values.length * 4),
    u = new Uint8Array(b),
    v = new DataView(b);
  u.set([147, 78, 85, 77, 80, 89, 1, 0]);
  v.setUint16(8, h.length, true);
  u.set(new TextEncoder().encode(h), 10);
  values.forEach((n, i) => v.setFloat32(10 + h.length + i * 4, n, true));
  return b;
}
function History({ points }: { points: { step: number; score: number }[] }) {
  const ys = points.map((p) => p.score),
    hi = Math.max(...ys),
    span = Math.max(hi - Math.min(...ys), 1e-5),
    last = Math.max(points.at(-1)?.step || 1, 1),
    path = points
      .map(
        (p, i) =>
          `${i ? "L" : "M"}${48 + (p.step / last) * 650},${18 + ((hi - p.score) / span) * 110}`,
      )
      .join(" ");
  return (
    <svg
      className="history"
      viewBox="0 0 720 166"
      role="img"
      aria-label={`Predicted objective from ${ys[0]?.toFixed(5)} to ${ys.at(-1)?.toFixed(5)}`}
    >
      <defs>
        <linearGradient id="chart-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#5ce3da" stopOpacity=".14" />
          <stop offset="1" stopColor="#5ce3da" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 0.5, 1].map((f) => (
        <g key={f}>
          <line
            x1="48"
            x2="698"
            y1={18 + 110 * f}
            y2={18 + 110 * f}
            stroke="#20303b"
            strokeDasharray="3 5"
          />
          <text x="2" y={22 + 110 * f}>
            {(hi - span * f).toFixed(3)}
          </text>
        </g>
      ))}
      <path d={`${path} L698,132 L48,132Z`} fill="url(#chart-fill)" />
      <path d={path} fill="none" stroke="#5ce3da" strokeWidth="2" />
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <text key={f} x={48 + 650 * f} y="150" textAnchor="middle">
          {Math.round(last * f)}
        </text>
      ))}
      <text x="374" y="165" textAnchor="middle">
        Optimization step
      </text>
    </svg>
  );
}

export default function Home() {
  const [config, setConfig] = useState<any>(null),
    [online, setOnline] = useState(false),
    [mode, setMode] = useState("examples"),
    [cif, setCif] = useState(""),
    [name, setName] = useState(""),
    [preview, setPreview] = useState<Preview | null>(null),
    [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState(""),
    [loading, setLoading] = useState(false),
    [job, setJob] = useState<string | null>(null),
    [progress, setProgress] = useState(0),
    [message, setMessage] = useState(""),
    [operation, setOperation] = useState("optimize"),
    [budget, setBudget] = useState("quick"),
    [seed, setSeed] = useState(42),
    [detail, setDetail] = useState("overview"),
    [method, setMethod] = useState(false),
    [drag, setDrag] = useState(false),
    [notice, setNotice] = useState(""),
    [latentView, setLatentView] = useState("optimized");
  const file = useRef<HTMLInputElement>(null),
    generation = useRef(0),
    jobRef = useRef<string | null>(null),
    finished = useRef(0);
  const busy = !!job;
  function invalidate() {
    generation.current++;
    if (jobRef.current)
      api(`jobs/${jobRef.current}/cancel`, {}).catch(() => {});
    jobRef.current = null;
    setJob(null);
    setResult(null);
    setPreview(null);
    setError("");
    setNotice("");
    setProgress(0);
  }
  async function prepare(text: string, label: string) {
    invalidate();
    const v = generation.current;
    setCif(text);
    setName(label);
    setLoading(true);
    try {
      if (new TextEncoder().encode(text).length > 100000)
        throw Error("Please choose a CIF smaller than 100 KB.");
      const p = await api("preview", { cif: text });
      if (v === generation.current) {
        setPreview(p);
        setOnline(true);
      }
    } catch (e: any) {
      if (v === generation.current) setError(e.message);
    } finally {
      if (v === generation.current) setLoading(false);
    }
  }
  async function connect() {
    setError("");
    try {
      const c = await api("config");
      setConfig(c);
      setOnline(true);
      if (!cif && c.examples.length)
        prepare(c.examples[0].cif, c.examples[0].name);
    } catch (e: any) {
      setOnline(false);
      setError(e.message);
    }
  }
  useEffect(() => {
    connect();
    return () => {
      generation.current++;
    };
  }, []);
  useEffect(() => {
    jobRef.current = job;
  }, [job]);
  useEffect(() => {
    if (!job) return;
    let stop = false,
      timer: ReturnType<typeof setTimeout>,
      failures = 0;
    async function poll() {
      try {
        const s = await api(`jobs/${job}`);
        if (stop) return;
        failures = 0;
        setProgress(s.progress);
        setMessage(s.message || "Working");
        if (s.status === "completed") {
          setResult(s.result);
          finished.current = Date.now();
          setJob(null);
          setDetail("overview");
          return;
        }
        if (s.status === "failed" || s.status === "cancelled") {
          setJob(null);
          if (s.status === "failed")
            setError(s.error || "The model could not complete this run.");
          else
            setNotice(
              "Run cancelled. Your input is ready for another exploration.",
            );
          return;
        }
      } catch (e: any) {
        if (stop) return;
        if (++failures >= 4) {
          api(`jobs/${job}/cancel`, {}).catch(() => {});
          setJob(null);
          setOnline(false);
          setError(e.message);
          return;
        }
        setMessage("Reconnecting to the model…");
      }
      timer = setTimeout(poll, 1500);
    }
    poll();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [job]);
  useEffect(() => {
    const t = setInterval(() => {
      if (finished.current && Date.now() - finished.current > 1800000) {
        setResult(null);
        finished.current = 0;
        setNotice("This result has expired. Run the model again to continue.");
      }
    }, 15000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (!method) return;
    const previous = document.activeElement as HTMLElement;
    const priorOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const f = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMethod(false);
      if (e.key === "Tab") {
        const nodes = Array.from(
          document.querySelectorAll<HTMLElement>(
            ".method-modal button, .method-modal a[href]",
          ),
        );
        const first = nodes[0],
          last = nodes.at(-1);
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last?.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first?.focus();
        }
      }
    };
    window.addEventListener("keydown", f);
    return () => {
      window.removeEventListener("keydown", f);
      document.body.style.overflow = priorOverflow;
      previous?.focus();
    };
  }, [method]);
  async function start(op: string) {
    if (!preview || busy || loading) return;
    if (!Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
      setError("Choose a whole-number seed between 0 and 2147483647.");
      return;
    }
    const v = generation.current;
    setError("");
    setNotice("");
    setResult(null);
    setOperation(op);
    setLoading(true);
    try {
      const s = await api("jobs", { cif, operation: op, budget, seed });
      if (v !== generation.current) {
        api(`jobs/${s.job_id}/cancel`, {}).catch(() => {});
        return;
      }
      setProgress(0);
      setMessage("Preparing the model");
      setJob(s.job_id);
    } catch (e: any) {
      if (v === generation.current) setError(e.message);
    } finally {
      if (v === generation.current) setLoading(false);
    }
  }
  async function upload(f: File | undefined) {
    if (!f) return;
    setMode("upload");
    if (f.size > 100000) {
      invalidate();
      setCif("");
      setError("Please choose a CIF smaller than 100 KB.");
      return;
    }
    try {
      await prepare(
        new TextDecoder("utf-8", { fatal: true }).decode(await f.arrayBuffer()),
        f.name,
      );
    } catch {
      invalidate();
      setError("The CIF must be a UTF-8 text file.");
    }
  }
  const output = result?.output_summary,
    latent =
      (latentView === "optimized" ? result?.optimized_latent : null) ||
      result?.latent;
  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#">
          <Mark />
          <span>
            Clique<span className="brand-light">Flowmer</span>
            <small>COMPUTATIONAL ALCHEMY</small>
          </span>
        </a>
        <nav>
          <a href="#workspace" className="nav-active">
            Workspace
          </a>
          <button onClick={() => setMethod(true)}>The method</button>
          <a
            href="https://arxiv.org/abs/2603.06082"
            target="_blank"
            rel="noreferrer"
          >
            Read the paper <ArrowUpRight size={13} />
          </a>
        </nav>
        <div className={`connection ${online ? "on" : ""}`}>
          <span />
          {online ? "MODEL CONNECTED" : error ? "MODEL OFFLINE" : "CONNECTING"}
        </div>
      </header>
      <main>
        <section className="intro">
          <div>
            <div className="eyebrow">
              <span />
              MATTER, REIMAGINED
            </div>
            <h1>
              A new direction
              <br className="mobile-break" /> for <em>discovery.</em>
            </h1>
            <p>
              Explore a crystal’s latent space. Transform its representation.
              <br className="desktop-break" /> Bring a new candidate into view.
            </p>
          </div>
          <div className="intro-diagram" aria-hidden="true">
            <div className="diagram-ring" />
            <Mark large />
            <span className="diagram-dot d1" />
            <span className="diagram-dot d2" />
            <span className="diagram-dot d3" />
            <div className="diagram-coordinate">z ∈ ℝ¹²¹</div>
          </div>
        </section>
        <section
          className="workspace"
          id="workspace"
          aria-label="Crystal workspace"
        >
          <div className="workspace-bar">
            <div>
              <FlaskConical size={16} />
              <h2>Crystal workspace</h2>
              <span className="version">01</span>
            </div>
            <div className="pipeline">
              <span className={preview ? "active" : ""}>INPUT</span>
              <ArrowRight size={12} />
              <span className={busy || result ? "active" : ""}>
                LATENT SPACE
              </span>
              <ArrowRight size={12} />
              <span className={output ? "active" : ""}>CANDIDATE</span>
            </div>
          </div>
          <div className="work-grid">
            <aside className="input-panel">
              <div className="panel-heading">
                <span className="step-number">01</span>
                <h3>Start with a crystal</h3>
              </div>
              <div
                className="input-tabs"
                role="tablist"
                aria-label="Input source"
              >
                {[
                  ["examples", "Examples"],
                  ["upload", "Upload"],
                  ["paste", "Paste CIF"],
                ].map(([id, label]) => (
                  <button
                    key={id}
                    role="tab"
                    aria-selected={mode === id}
                    className={mode === id ? "selected" : ""}
                    onClick={() => setMode(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              {mode === "examples" ? (
                <div className="example-list">
                  {config?.examples.map((e: any, i: number) => (
                    <button
                      key={e.name}
                      className={`example ${name === e.name ? "chosen" : ""}`}
                      onClick={() => prepare(e.cif, e.name)}
                    >
                      <span className="example-glyph">
                        <Atom size={20} />
                      </span>
                      <span>
                        <strong>
                          {["Strontium titanate", "Silicon", "Copper"][i] ||
                            e.name}
                        </strong>
                        <small>
                          {["SrTiO₃ · 5 sites", "Si · 2 sites", "Cu · 4 sites"][
                            i
                          ] || "Example crystal"}
                        </small>
                      </span>
                      {name === e.name ? (
                        <Check size={15} />
                      ) : (
                        <ArrowRight size={14} />
                      )}
                    </button>
                  ))}
                </div>
              ) : mode === "upload" ? (
                <>
                  <input
                    ref={file}
                    className="visually-hidden"
                    type="file"
                    accept=".cif"
                    aria-label="Upload CIF file"
                    onChange={(e) => upload(e.target.files?.[0])}
                  />
                  <button
                    className={`drop-zone ${drag ? "dragging" : ""}`}
                    onClick={() => file.current?.click()}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDrag(true);
                    }}
                    onDragLeave={() => setDrag(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDrag(false);
                      upload(e.dataTransfer.files[0]);
                    }}
                  >
                    <Upload size={24} />
                    <strong>Drop your CIF here</strong>
                    <span>or click to browse</span>
                    <small>.cif · up to 100 KB</small>
                  </button>
                  {name && <div className="file-name">{name}</div>}
                </>
              ) : (
                <div className="paste-input">
                  <textarea
                    aria-label="CIF contents"
                    spellCheck={false}
                    value={cif}
                    placeholder={"data_crystal\n_cell_length_a …"}
                    onChange={(e) => {
                      invalidate();
                      setCif(e.target.value);
                      setName("Pasted CIF");
                      setLoading(false);
                    }}
                  />
                  <button
                    className="text-button"
                    onClick={() => prepare(cif, "Pasted CIF")}
                    disabled={!cif.trim() || loading}
                  >
                    Load structure <ArrowRight size={13} />
                  </button>
                </div>
              )}
              <p className="input-hint">
                One ordered crystal. <br />
                1–20 atoms in the uploaded cell.
              </p>
              <div className="panel-divider" />
              <div className="panel-heading">
                <span className="step-number copper">02</span>
                <h3>Choose a direction</h3>
              </div>
              <div className="objective-card">
                <span className="objective-icon">
                  <Sparkles size={16} />
                </span>
                <div>
                  <strong>Lower the model objective</strong>
                  <small>Formation-energy checkpoint</small>
                </div>
              </div>
              <div className="budget-control" aria-label="Optimization budget">
                {[
                  ["quick", "Quick", "100 steps"],
                  ["standard", "Extended", "500 steps"],
                ].map(([v, t, s]) => (
                  <button
                    key={v}
                    className={budget === v ? "chosen" : ""}
                    disabled={busy}
                    onClick={() => setBudget(v)}
                  >
                    <strong>{t}</strong>
                    <small>{s}</small>
                  </button>
                ))}
              </div>
              <details className="advanced">
                <summary>
                  Run settings <ChevronDown size={13} />
                </summary>
                <label>
                  Random seed
                  <input
                    type="number"
                    min="0"
                    max="2147483647"
                    value={seed}
                    disabled={busy}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  />
                </label>
                <p>
                  Same input, settings, and environment give a repeatable run.
                </p>
              </details>
              <button
                className="primary-button"
                onClick={() => start("optimize")}
                disabled={!preview || busy || loading}
              >
                {busy ? (
                  <Loader2 size={17} className="spin" />
                ) : (
                  <Sparkles size={17} />
                )}
                <span>{busy ? "Exploring…" : "Optimize crystal"}</span>
                <ArrowRight size={16} />
              </button>
              <div className="secondary-actions">
                <button
                  onClick={() => start("encode")}
                  disabled={!preview || busy || loading}
                >
                  <Braces size={14} />
                  Encode
                </button>
                <button
                  onClick={() => start("reconstruct")}
                  disabled={!preview || busy || loading}
                >
                  <RotateCcw size={14} />
                  Reconstruct
                </button>
              </div>
              <div className="compute-note">
                <span />
                CPU inference · No installation
              </div>
            </aside>
            <div className="visualization-panel">
              <div className="viewers">
                <div className="crystal-card">
                  <div className="crystal-heading">
                    <span>INPUT STRUCTURE</span>
                    <span className="tag">ORIGINAL</span>
                  </div>
                  <div className="crystal-title">
                    <h3>
                      {preview ? (
                        <Formula text={preview.summary.formula} />
                      ) : loading ? (
                        "Reading crystal…"
                      ) : (
                        "Your crystal"
                      )}
                    </h3>
                    <span>
                      {preview
                        ? `${preview.summary.n_sites} sites · ${preview.summary.volume.toFixed(2)} Å³`
                        : "Upload or choose an example"}
                    </span>
                  </div>
                  <Crystal data={preview?.viewer || null} />
                </div>
                <div
                  className={`crystal-card candidate ${output ? "has-result" : ""}`}
                >
                  <div className="crystal-heading">
                    <span>
                      {operation === "reconstruct"
                        ? "RECONSTRUCTION"
                        : "GENERATED CANDIDATE"}
                    </span>
                    <span className="tag copper">
                      {output ? "GENERATED" : "UNEXPLORED"}
                    </span>
                  </div>
                  <div className="crystal-title">
                    <h3>
                      {output ? (
                        <Formula text={output.formula} />
                      ) : result?.operation === "encode" ? (
                        "Representation ready"
                      ) : busy ? (
                        "Transforming…"
                      ) : (
                        "What comes next?"
                      )}
                    </h3>
                    <span>
                      {output
                        ? `${output.n_sites} sites · ${output.volume.toFixed(2)} Å³`
                        : result?.operation === "encode"
                          ? "121 coordinates, ready to explore"
                          : "A new point in material space"}
                    </span>
                  </div>
                  <Crystal data={result?.output_viewer || null} busy={busy} />
                </div>
              </div>
              {busy && (
                <div className="progress-panel" role="status">
                  <div>
                    <Loader2 size={14} className="spin" />
                    <span>{message}</span>
                    <b>{Math.round(progress * 100)}%</b>
                    <button
                      onClick={() =>
                        api(`jobs/${job}/cancel`, {})
                          .then(() => setMessage("Cancelling…"))
                          .catch((e: any) => setError(e.message))
                      }
                    >
                      Cancel <X size={12} />
                    </button>
                  </div>
                  <div className="progress-track">
                    <i style={{ width: `${progress * 100}%` }} />
                  </div>
                </div>
              )}
              {error && (
                <div className="alert error" role="alert">
                  <span>{error}</span>
                  {!online && <button onClick={connect}>Reconnect</button>}
                  <button
                    aria-label="Dismiss error"
                    onClick={() => setError("")}
                  >
                    <X size={14} />
                  </button>
                </div>
              )}
              {notice && (
                <div className="alert" role="status">
                  {notice}
                </div>
              )}
              <div className="inspector">
                <div className="inspector-nav">
                  <div>
                    <button
                      className={detail === "overview" ? "active" : ""}
                      onClick={() => setDetail("overview")}
                    >
                      Overview
                    </button>
                    <button
                      className={detail === "representation" ? "active" : ""}
                      onClick={() => setDetail("representation")}
                    >
                      Representation <span>121D</span>
                    </button>
                  </div>
                  {result && (
                    <button
                      className="download-link"
                      onClick={() =>
                        save(
                          "cliqueflowmer-result.json",
                          JSON.stringify(result, null, 2),
                        )
                      }
                    >
                      <Download size={13} />
                      Run data
                    </button>
                  )}
                </div>
                {detail === "overview" ? (
                  result ? (
                    <>
                      <div className="score-grid">
                        <div>
                          <span>STARTING SCORE</span>
                          <strong>{result.initial_score.toFixed(5)}</strong>
                        </div>
                        <div>
                          <span>FINAL SCORE</span>
                          <strong className="cyan">
                            {result.final_score.toFixed(5)}
                          </strong>
                        </div>
                        <div>
                          <span>CHANGE</span>
                          <strong className="change">
                            {result.final_score - result.initial_score > 0
                              ? "+"
                              : ""}
                            {(
                              result.final_score - result.initial_score
                            ).toFixed(5)}
                          </strong>
                        </div>
                      </div>
                      {result.history.length > 1 && (
                        <History points={result.history} />
                      )}
                      <p className="result-note">
                        Raw latent model scores · Lower is the optimization
                        target. Generated structures are unrelaxed; composition
                        may change.
                      </p>
                    </>
                  ) : (
                    <div className="inspector-empty">
                      <div className="mini-chart">
                        <svg viewBox="0 0 120 40">
                          <path
                            d="M0 30h120M0 15h120M0 0v40M40 0v40M80 0v40M120 0v40"
                            stroke="currentColor"
                            fill="none"
                          />
                        </svg>
                      </div>
                      <div>
                        <strong>A clearer view of the transformation</strong>
                        <p>
                          Run the model to compare scores and explore its
                          optimization trajectory.
                        </p>
                      </div>
                    </div>
                  )
                ) : latent ? (
                  <div className="latent-panel">
                    <div className="latent-toolbar">
                      <span>
                        {latentView === "optimized" && result?.optimized_latent
                          ? "Optimized representation"
                          : "Encoded representation"}
                      </span>
                      {result?.optimized_latent && (
                        <div>
                          {["encoded", "optimized"].map((v) => (
                            <button
                              key={v}
                              className={latentView === v ? "active" : ""}
                              onClick={() => setLatentView(v)}
                            >
                              {v === "encoded" ? "Original" : "Optimized"}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="latent-label">
                      <strong>Eight connected cliques</strong>
                      <span>
                        16 coordinates each · 1 shared coordinate between
                        neighbors
                      </span>
                    </div>
                    <div className="clique-grid">
                      {Array.from({ length: 8 }, (_, row) => (
                        <div className="clique-row" key={row}>
                          <span>C{row + 1}</span>
                          {Array.from({ length: 16 }, (_, col) => {
                            const idx = row * 15 + col,
                              value = latent[idx],
                              a = Math.min(
                                0.92,
                                0.15 +
                                  (Math.abs(value) /
                                    Math.max(...latent.map(Math.abs), 0.01)) *
                                    0.77,
                              );
                            return (
                              <div
                                key={col}
                                title={`Clique ${row + 1}, z[${idx}] = ${value.toPrecision(7)}`}
                                style={{
                                  background: `rgba(${value >= 0 ? "91,227,217" : "210,156,111"},${a})`,
                                }}
                              />
                            );
                          })}
                        </div>
                      ))}
                    </div>
                    <p className="result-note">
                      Cyan: positive · Copper: negative. Shared coordinates are
                      outlined. These learned features do not correspond to
                      individual atoms or known physical properties.
                    </p>
                  </div>
                ) : (
                  <div className="inspector-empty">
                    <Braces size={30} />
                    <div>
                      <strong>The structure beneath the structure</strong>
                      <p>
                        Encode a crystal to reveal its 121-dimensional latent
                        representation.
                      </p>
                    </div>
                  </div>
                )}
              </div>
              {result && (
                <div className="result-downloads">
                  <span>
                    <Check size={13} />
                    {result.operation === "encode"
                      ? "Representation encoded"
                      : "Exploration complete"}
                  </span>
                  {result.output_cif && (
                    <button
                      onClick={() =>
                        save(
                          "cliqueflowmer-candidate.cif",
                          result.output_cif!,
                          "chemical/x-cif",
                        )
                      }
                    >
                      <Download size={14} />
                      Candidate CIF
                    </button>
                  )}
                  <button
                    onClick={() =>
                      save(
                        "cliqueflowmer-latent.npy",
                        npy(result.latent),
                        "application/octet-stream",
                      )
                    }
                  >
                    <Download size={14} />
                    Encoded latent
                  </button>
                  {result.optimized_latent && (
                    <button
                      onClick={() =>
                        save(
                          "cliqueflowmer-optimized-latent.npy",
                          npy(result.optimized_latent!),
                          "application/octet-stream",
                        )
                      }
                    >
                      <Download size={14} />
                      Optimized latent
                    </button>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>
        <div className="below-workspace">
          <span>
            <Atom size={14} />
            An experiment in learned crystal representations.
          </span>
          <button onClick={() => setMethod(true)}>
            Understand the model <ArrowUpRight size={13} />
          </button>
        </div>
      </main>
      <footer>
        <span className="footer-brand">CliqueFlowmer</span>
        <p>Research, made explorable.</p>
        <a
          href="https://github.com/znowu/CliqueFlowmer"
          target="_blank"
          rel="noreferrer"
        >
          GitHub <ArrowUpRight size={12} />
        </a>
        <a
          href="https://huggingface.co/iamkuba/CliqueFlowmer"
          target="_blank"
          rel="noreferrer"
        >
          Model weights <ArrowUpRight size={12} />
        </a>
      </footer>
      {method && (
        <div className="modal-backdrop" onClick={() => setMethod(false)}>
          <section
            className="method-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="method-title"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              className="modal-close"
              autoFocus
              aria-label="Close method"
              onClick={() => setMethod(false)}
            >
              <X size={20} />
            </button>
            <div className="eyebrow">THE METHOD</div>
            <h2 id="method-title">
              A different space
              <br />
              for material discovery.
            </h2>
            <div className="method-steps">
              <div>
                <span>01</span>
                <h3>Represent</h3>
                <p>
                  The encoder maps your crystal to 121 latent coordinates,
                  organized into eight overlapping cliques.
                </p>
              </div>
              <div>
                <span>02</span>
                <h3>Explore</h3>
                <p>
                  Evolution strategies optimize the learned objective in latent
                  space, using 100 or 500 steps.
                </p>
              </div>
              <div>
                <span>03</span>
                <h3>Generate</h3>
                <p>
                  The decoder produces a candidate crystal using 1,000 flow
                  steps. One geometry retry is allowed.
                </p>
              </div>
            </div>
            <p className="method-note">
              The published formation-energy checkpoint lacks target scaling
              metadata. Scores are raw latent model outputs, not calibrated
              eV/atom or independently evaluated candidate energies. Structures
              are unrelaxed and are not DFT-validated.
            </p>
            <p className="method-note">
              Uploads stay in temporary server memory. Results expire after 30
              minutes. One CPU job runs at a time.
            </p>
            <a
              className="paper-link"
              href="https://arxiv.org/abs/2603.06082"
              target="_blank"
              rel="noreferrer"
            >
              Read the paper <ArrowUpRight size={15} />
            </a>
          </section>
        </div>
      )}
    </div>
  );
}

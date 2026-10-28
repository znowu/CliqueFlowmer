"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createLiquidSurface, type LiquidRenderer } from "../lib/liquid-surface";

const MOTION_PREFERENCE = "cliqueflowmer-liquid-motion";

function savedMotion(): boolean | null {
  try {
    const value = window.localStorage.getItem(MOTION_PREFERENCE);
    return value === "on" ? true : value === "off" ? false : null;
  } catch {
    return null;
  }
}

export function useLiquidMotion() {
  // A still first render also makes server rendering and hydration agree.
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const applyPreference = () => setEnabled(!reducedMotion.matches && (savedMotion() ?? true));
    applyPreference();
    reducedMotion.addEventListener("change", applyPreference);
    return () => reducedMotion.removeEventListener("change", applyPreference);
  }, []);

  const toggle = useCallback(() => {
    setEnabled((previous) => {
      const next = !previous;
      try {
        window.localStorage.setItem(MOTION_PREFERENCE, next ? "on" : "off");
      } catch {
        // The current visit still works when browser storage is unavailable.
      }
      return next;
    });
  }, []);

  return { enabled, toggle };
}

export function LiquidSurface({ enabled }: { enabled: boolean }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const fallback = useRef<HTMLCanvasElement>(null);
  const engine = useRef<ReturnType<typeof createLiquidSurface> | null>(null);
  const [renderer, setRenderer] = useState<LiquidRenderer>("static");

  useEffect(() => {
    if (!canvas.current || !fallback.current) return;
    const surface = createLiquidSurface(canvas.current, fallback.current, setRenderer);
    engine.current = surface;
    return () => {
      engine.current = null;
      surface.dispose();
    };
  }, []);

  useEffect(() => engine.current?.setEnabled(enabled), [enabled]);

  return (
    <div className="liquid-surface" aria-hidden="true" data-renderer={renderer} data-motion={enabled ? "on" : "off"} style={{ pointerEvents: "none" }}>
      <canvas ref={canvas} className="liquid-canvas" style={{ display: renderer === "webgl" ? "block" : "none" }} />
      <canvas ref={fallback} className="liquid-canvas liquid-fallback" style={{ display: renderer === "canvas2d" ? "block" : "none" }} />
      <div className="liquid-vignette" />
    </div>
  );
}

"use client";

import { useState } from "react";

/** Both vectors use identical ordering and a shared, zero-centered color scale. */
export function LatentComparison({
  original,
  optimized,
}: {
  original: number[];
  optimized: number[] | null;
}) {
  const [selected, setSelected] = useState(0);
  const index = Math.min(selected, original.length - 1);
  const scale = Math.max(
    ...original.map(Math.abs),
    ...(optimized?.map(Math.abs) || []),
    0.000001,
  );
  const rows = [
    { label: "Original", values: original },
    ...(optimized ? [{ label: "Optimized", values: optimized }] : []),
  ];
  const number = (value: number) => value.toPrecision(6);

  return (
    <section
      className="latent-comparison"
      aria-label="Latent representation comparison"
    >
      <div className="representation-heading">
        <div>
          <h3>
            {optimized
              ? "Original and optimized representations"
              : "Your crystal, in latent space."}
          </h3>
          <p>
            {original.length} coordinates in order
            {optimized ? " · One shared color scale" : ""}
          </p>
        </div>
        <div
          className="latent-legend"
          aria-label="Copper is negative; cyan is positive"
        >
          <span>
            <i className="negative" />
            Negative
          </span>
          <span>
            <i className="positive" />
            Positive
          </span>
        </div>
      </div>
      <div
        className="latent-scroll"
        aria-label="Representation rows; scroll horizontally on small screens"
      >
        <div className="latent-rows">
          {rows.map(({ label, values }) => (
            <div className="latent-vector" key={label}>
              <div className="vector-label">
                <strong>{label}</strong>
                <span>{values.length}D</span>
              </div>
              <div
                className="latent-blocks"
                role="group"
                aria-label={`${label} latent coordinates`}
                style={{
                  gridTemplateColumns: `repeat(${values.length}, minmax(0, 1fr))`,
                }}
              >
                {values.map((value, i) => (
                  <button
                    type="button"
                    className={index === i ? "selected-coordinate" : ""}
                    key={i}
                    data-index={i}
                    tabIndex={index === i ? 0 : -1}
                    aria-label={`${label} coordinate ${i}: ${number(value)}`}
                    aria-pressed={index === i}
                    title={`z[${i}] = ${value.toPrecision(9)}`}
                    style={{
                      backgroundColor: `rgba(${value >= 0 ? "91,227,217" : "210,156,111"},${0.12 + (0.83 * Math.abs(value)) / scale})`,
                    }}
                    onMouseEnter={() => setSelected(i)}
                    onFocus={() => setSelected(i)}
                    onClick={() => setSelected(i)}
                    onKeyDown={(event) => {
                      const next =
                        event.key === "ArrowRight"
                          ? Math.min(i + 1, values.length - 1)
                          : event.key === "ArrowLeft"
                            ? Math.max(i - 1, 0)
                            : event.key === "Home"
                              ? 0
                              : event.key === "End"
                                ? values.length - 1
                                : null;
                      if (next === null) return;
                      event.preventDefault();
                      setSelected(next);
                      const block =
                        event.currentTarget.parentElement?.querySelector<HTMLButtonElement>(
                          `[data-index="${next}"]`,
                        );
                      block?.focus({ preventScroll: true });
                      block?.scrollIntoView({
                        block: "nearest",
                        inline: "nearest",
                      });
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
          <div className="latent-axis" aria-hidden="true">
            {[0, 0.25, 0.5, 0.75, 1].map((f) => (
              <span key={f}>z[{Math.round((original.length - 1) * f)}]</span>
            ))}
          </div>
        </div>
      </div>
      <p className="latent-scroll-hint">
        Swipe or scroll to see all {original.length} coordinates.
      </p>
      <div className="coordinate-inspector">
        <div>
          <span>COORDINATE</span>
          <strong>z[{index}]</strong>
        </div>
        <div>
          <span>ORIGINAL</span>
          <strong>{number(original[index])}</strong>
        </div>
        {optimized && (
          <>
            <div>
              <span>OPTIMIZED</span>
              <strong className="cyan">{number(optimized[index])}</strong>
            </div>
            <div>
              <span>CHANGE</span>
              <strong>{number(optimized[index] - original[index])}</strong>
            </div>
          </>
        )}
      </div>
      <p className="coordinate-help">
        Hover, tap, or use arrow keys to inspect a coordinate. Color intensity
        shows magnitude.
      </p>
      <p className="representation-note">
        These learned coordinates describe the whole crystal; they are not
        individual atoms or physical properties.
      </p>
    </section>
  );
}

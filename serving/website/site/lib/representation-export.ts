export type RepresentationFormat = "npy" | "csv" | "json";
export type RepresentationLabel = "original" | "optimized";
export type RepresentationFile = {
  name: string;
  data: string | ArrayBuffer;
  mime: string;
};

/** Encode a one-dimensional float32 array using the NumPy 1.0 file format. */
function numpyArray(values: number[]): ArrayBuffer {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${values.length},), }`;
  // The ten-byte preamble plus header must end on a 16-byte boundary.
  header += " ".repeat((16 - ((10 + header.length + 1) % 16)) % 16) + "\n";
  const buffer = new ArrayBuffer(10 + header.length + values.length * 4);
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  bytes.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
  view.setUint16(8, header.length, true);
  bytes.set(new TextEncoder().encode(header), 10);
  values.forEach((value, index) => {
    view.setFloat32(10 + header.length + index * 4, value, true);
  });
  return buffer;
}

/**
 * Export coordinates in their original order. Text formats retain the full
 * numeric precision received from the model; NPY stores its native float32.
 * This helper creates no DOM elements and does not mutate the supplied array.
 */
export function representationFile(
  values: number[],
  label: RepresentationLabel,
  format: RepresentationFormat,
): RepresentationFile {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error("The representation must contain numeric coordinates.");
  }
  for (const value of values) {
    if (!Number.isFinite(value) || !Number.isFinite(Math.fround(value))) {
      throw new Error(
        "Representation coordinates must be finite float32 values.",
      );
    }
  }
  if (label !== "original" && label !== "optimized") {
    throw new Error("Choose the original or optimized representation.");
  }
  const name = `cliqueflowmer-${label}-latent.${format}`;
  switch (format) {
    case "npy":
      return {
        name,
        data: numpyArray(values),
        mime: "application/octet-stream",
      };
    case "csv":
      return {
        name,
        data:
          "index,value\r\n" +
          values
            .map(
              (value, index) =>
                `${index},${Object.is(value, -0) ? "-0" : value}`,
            )
            .join("\r\n") +
          "\r\n",
        mime: "text/csv;charset=utf-8",
      };
    case "json":
      return {
        name,
        data:
          JSON.stringify(
            { representation: label, dimension: values.length, values },
            null,
            2,
          ) + "\n",
        mime: "application/json",
      };
    default:
      throw new Error("Choose NumPy, CSV, or JSON for the representation.");
  }
}

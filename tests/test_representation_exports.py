"""Cross-language checks for browser exports consumed by scientific tools.

Needs the website's installed TypeScript dependency and Node.js, plus NumPy.
The test skips when frontend dependencies are not installed.
"""
import base64
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "serving/website/site"
NODE = shutil.which("node")
TYPESCRIPT = SITE / "node_modules/typescript"

# Transpile the actual browser helper, then export through its public API.
RUN_EXPORTS = r"""
const fs = require('node:fs');
const ts = require(process.argv[1]);
const source = fs.readFileSync(process.argv[2], 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020},
}).outputText;
const library = {};
new Function('exports', compiled)(library);
const fixture = JSON.parse(fs.readFileSync(0, 'utf8'));
const files = [];
for (const label of ['original', 'optimized']) {
  for (const format of ['npy', 'csv', 'json']) {
    const file = library.representationFile(fixture[label], label, format);
    files.push({...file, label, format, data: typeof file.data === 'string'
      ? file.data : Buffer.from(file.data).toString('base64')});
  }
}
const rejected = [[], [NaN], [Infinity], [-Infinity], [3.5e38], [undefined], Array(2)]
  .map(values => {
    try {library.representationFile(values, 'original', 'npy'); return false;}
    catch {return true;}
  });
process.stdout.write(JSON.stringify({files, rejected}));
"""


@unittest.skipUnless(NODE and TYPESCRIPT.exists(), "Frontend dependencies are not installed")
class RepresentationExportTests(unittest.TestCase):
    def test_browser_exports_round_trip_in_numpy_csv_and_json(self):
        # Exercise precision, ordering, zero, negative values, and float32 limits.
        original = np.linspace(-3, 3, 121, dtype=np.float32)
        original[:5] = [0, -0.0, np.finfo(np.float32).tiny,
                        np.finfo(np.float32).max, -np.finfo(np.float32).max]
        fixture = {"original": original.tolist(), "optimized": original[::-1].tolist()}
        result = subprocess.run(
            [NODE, "-e", RUN_EXPORTS, str(TYPESCRIPT),
             str(SITE / "lib/representation-export.ts")],
            input=json.dumps(fixture), text=True, capture_output=True,
            timeout=30, check=True,
        )
        exported = json.loads(result.stdout)
        self.assertTrue(all(exported["rejected"]))
        for file in exported["files"]:
            with self.subTest(label=file["label"], format=file["format"]):
                expected = fixture[file["label"]]
                self.assertEqual(file["name"],
                                 f"cliqueflowmer-{file['label']}-latent.{file['format']}")
                if file["format"] == "npy":
                    actual = np.load(io.BytesIO(base64.b64decode(file["data"])),
                                     allow_pickle=False)
                    self.assertEqual(actual.dtype, np.dtype("float32"))
                    self.assertEqual(actual.shape, (121,))
                elif file["format"] == "csv":
                    rows = list(csv.DictReader(io.StringIO(file["data"])))
                    self.assertEqual([int(row["index"]) for row in rows], list(range(121)))
                    actual = [float(row["value"]) for row in rows]
                else:
                    document = json.loads(file["data"])
                    self.assertEqual(document["representation"], file["label"])
                    self.assertEqual(document["dimension"], 121)
                    actual = document["values"]
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()

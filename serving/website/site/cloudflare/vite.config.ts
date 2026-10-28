import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The public deployment uses the same UI as the Sites preview, without SSR or
// Sites authentication. Static assets do not spend the free Worker's CPU budget.
export default defineConfig({
  root: fileURLToPath(new URL(".", import.meta.url)),
  publicDir: fileURLToPath(new URL("../public", import.meta.url)),
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL("../dist/cloudflare/public", import.meta.url)),
    emptyOutDir: true,
  },
});

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Development proxy: requests under /api are forwarded to the local FastAPI backend.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});

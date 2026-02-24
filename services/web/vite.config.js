import { defineConfig } from "vite";

export default defineConfig({
  server: {
    port: 3000,
    proxy: {
      "/detections": "http://localhost:8000",
      "/known-sites": "http://localhost:8000",
      "/evidence": "http://localhost:8000",
      "/tiles": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});

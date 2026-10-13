import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Dev: the browser talks to Vite on :5173; Vite forwards /api and /ws to the Express server.
const API = process.env.VITE_API_TARGET || 'http://localhost:4000';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    chunkSizeWarningLimit: 900,
    rollupOptions: { output: { manualChunks: { charts: ['recharts'], motion: ['framer-motion'] } } },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': API,
      '/ws': { target: API.replace(/^http/, 'ws'), ws: true },
    },
  },
});

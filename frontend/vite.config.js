import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to Flask backend during development
      '/api':     { target: 'http://localhost:5000', changeOrigin: true },
      '/predict': { target: 'http://localhost:5000', changeOrigin: true },
      '/compare':  { target: 'http://localhost:5000', changeOrigin: true },
      '/health':   { target: 'http://localhost:5000', changeOrigin: true },
      '/circuit':  { target: 'http://localhost:5000', changeOrigin: true },
      '/report':   { target: 'http://localhost:5000', changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    assetsDir: 'assets',
  },
})

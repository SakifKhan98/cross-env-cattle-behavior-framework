import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/jobs': {
        target: 'http://localhost:8000',
        // Browser navigations send Accept: text/html — serve the SPA instead of proxying
        bypass(req) {
          if (req.headers.accept?.includes('text/html')) return '/index.html'
        },
      },
      '/results': 'http://localhost:8000',
    },
  },
})

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const backendUrl = process.env.VITE_BACKEND_URL || 'http://127.0.0.1:8002';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 8501,
    host: '0.0.0.0',
    proxy: {
      '/auth': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/chat': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/history': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/health': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/feedback': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/approvals': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/audit': {
        target: backendUrl,
        changeOrigin: true,
      },
      '/stats': {
        target: backendUrl,
        changeOrigin: true,
      },
    },
  },
});

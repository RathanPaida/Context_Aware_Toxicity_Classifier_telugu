import path from 'node:path';
import { fileURLToPath } from 'node:url';

/** @type {import('next').NextConfig} */
// Backend location is configurable: BACKEND_URL=http://host:port npm run dev
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';

const nextConfig = {
  // Pin the project root (a stray package-lock.json higher up otherwise confuses Next.js).
  turbopack: { root: path.dirname(fileURLToPath(import.meta.url)) },
  // `npm run dev` blocks its own scripts when opened as 127.0.0.1 instead of localhost.
  allowedDevOrigins: ['127.0.0.1'],
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${BACKEND_URL}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;

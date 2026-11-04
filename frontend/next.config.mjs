/** @type {import('next').NextConfig} */
// Backend location is configurable: BACKEND_URL=http://host:port npm run dev
const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:7860';

const nextConfig = {
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

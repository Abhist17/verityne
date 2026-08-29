/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // Proxy the API and its static assets through the dashboard origin so the
  // browser never has to deal with CORS or mixed origins for heatmap images.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: `${API}/:path*` },
      { source: "/static/:path*", destination: `${API}/static/:path*` },
    ];
  },
};
export default nextConfig;

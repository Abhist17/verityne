/** @type {import('next').NextConfig} */
const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig = {
  reactStrictMode: true,
  // The Gauntlet's live results arrive over Server-Sent Events. Next compresses
  // proxied responses by default, and gzip holds a stream in its buffer until it
  // has enough bytes to emit a block - so every `result` event sat in that buffer
  // and the whole run landed at once, after ~75s of an apparently frozen page.
  // The backend already sends `X-Accel-Buffering: no`; that instructs nginx, not
  // this proxy. `curl` without `Accept-Encoding` streamed correctly the whole
  // time, which is exactly why this survived: only a browser reproduces it.
  //
  // Responses here are small JSON and already-compressed JPEG, so this costs
  // very little. If a CDN or reverse proxy fronts this app in production, let it
  // compress the static assets instead.
  compress: false,
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

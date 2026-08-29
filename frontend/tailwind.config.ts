import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: { 950: "#07090f", 900: "#0b0e16", 850: "#10141f", 800: "#161b28", 700: "#1e2432", 600: "#2a3142" },
        edge: "#232a3a",
        pass: { DEFAULT: "#2dd4a7", dim: "#0f3a30" },
        review: { DEFAULT: "#f5b53d", dim: "#3d2f10" },
        reject: { DEFAULT: "#fb5e6d", dim: "#3d1520" },
        accent: { DEFAULT: "#5b8cff", dim: "#16233f" },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      keyframes: {
        shimmer: { "0%": { backgroundPosition: "-200% 0" }, "100%": { backgroundPosition: "200% 0" } },
        pulseRing: { "0%": { transform: "scale(0.9)", opacity: "0.7" }, "100%": { transform: "scale(1.6)", opacity: "0" } },
      },
      animation: { shimmer: "shimmer 1.8s linear infinite", pulseRing: "pulseRing 1.4s ease-out infinite" },
    },
  },
  plugins: [],
};
export default config;

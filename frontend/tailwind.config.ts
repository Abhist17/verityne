import type { Config } from "tailwindcss";

/**
 * The palette is deliberately narrow. This is a forensics console: colour is a
 * signal, not decoration, so the base is a single neutral ramp and hue is spent
 * only where it carries meaning — a verdict, a threshold, an interactive
 * affordance. Anything that is merely structure is a value step, not a colour.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      screens: {
        // Tailwind's `lg` is 1024px, which is just above the ~1000px viewport a
        // 13" laptop actually reports. Every two-column split in this app was
        // collapsing there, so the splits key off this instead.
        wide: "960px",
      },
      colors: {
        // Surface ramp: even perceptual steps, faintly cool so the accent reads
        // as the same family rather than a sticker on top of grey.
        ink: {
          950: "#08090c",
          900: "#0c0e13",
          850: "#101319",
          800: "#161a22",
          750: "#1c212b",
          700: "#232935",
          600: "#2f3644",
        },
        edge: {
          DEFAULT: "#1e2430", // hairline between surfaces
          strong: "#2b3342", // hairline that needs to be seen
        },
        // Verdicts. Desaturated from the usual traffic-light set: at 12% alpha on
        // a near-black ground the saturated versions glow, and a glow reads as
        // urgency the score has not earned.
        pass: { DEFAULT: "#3ddc97", dim: "#0e2a20" },
        review: { DEFAULT: "#e8b04b", dim: "#2c2211" },
        reject: { DEFAULT: "#f4626f", dim: "#301218" },
        accent: { DEFAULT: "#6d8cff", dim: "#151c33" },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        // A tighter ramp than Tailwind's default. An ops console is read at a
        // glance and at density; the default 16px body is one step too loose.
        "2xs": ["10px", { lineHeight: "14px", letterSpacing: "0.04em" }],
        xs: ["11px", { lineHeight: "16px" }],
        sm: ["12.5px", { lineHeight: "18px" }],
        base: ["13.5px", { lineHeight: "20px" }],
        lg: ["15px", { lineHeight: "22px" }],
        xl: ["18px", { lineHeight: "24px", letterSpacing: "-0.011em" }],
        "2xl": ["22px", { lineHeight: "28px", letterSpacing: "-0.018em" }],
        "3xl": ["28px", { lineHeight: "34px", letterSpacing: "-0.022em" }],
      },
      keyframes: {
        shimmer: { "0%": { transform: "translateX(-100%)" }, "100%": { transform: "translateX(300%)" } },
        sweep: { "0%": { opacity: "0.25" }, "50%": { opacity: "1" }, "100%": { opacity: "0.25" } },
        rise: { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "none" } },
      },
      animation: {
        shimmer: "shimmer 1.6s ease-in-out infinite",
        sweep: "sweep 1.4s ease-in-out infinite",
        rise: "rise 0.35s cubic-bezier(0.2, 0.8, 0.2, 1) both",
      },
    },
  },
  plugins: [],
};
export default config;

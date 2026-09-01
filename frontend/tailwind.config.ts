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
        // Surface ramp. Warmed very slightly off true blue-black: a pure cool
        // grey next to the ochre and red-earth verdicts reads as two unrelated
        // palettes stacked, and the small amount of red in the ground is what
        // makes them look chosen together.
        ink: {
          950: "#0a0a0c",
          900: "#0e0e12",
          850: "#131318",
          800: "#191920",
          750: "#1f2028",
          700: "#272833",
          600: "#343542",
        },
        edge: {
          DEFAULT: "#20212a", // hairline between surfaces
          strong: "#2d2f3b", // hairline that needs to be seen
        },
        // Verdicts, and the one interactive hue.
        //
        // Taken from pigment rather than from a screen: verdigris, yellow ochre,
        // and red earth, against an ink blue. The previous set was mint, gold and
        // coral at full luminosity, which on a near-black ground *emits* rather
        // than sits — everything glowed, so nothing was emphatic, and a REJECT
        // could not look more serious than a PASS because both were already at
        // maximum. These are all held around 55-62% lightness and under 45%
        // saturation, which is where print keeps its warning colours, and leaves
        // headroom to brighten a single element when something is genuinely wrong.
        //
        // Hue assignment is forced, not chosen: the verdict ramp has to be the
        // traffic-light set because that is what a reviewer already reads without
        // being taught. That spoken for, there was no hue left for the accent
        // worth having — the traffic lights occupy green through red, and cyan,
        // violet and magenta are the ones that read as a default template rather
        // than a decision.
        //
        // So the accent carries no hue at all. It is warm bone, and it is the
        // brightest value in the interface: **colour means a verdict, brightness
        // means you can act on it.** That is a cleaner rule than any fourth hue
        // would have given, and it means a chart series or a progress bar can
        // never be mistaken for a state. It reads warm against the slate text
        // ramp, which is cool, so the two stay separable side by side.
        pass: { DEFAULT: "#5a9e79", dim: "#12241c" },
        review: { DEFAULT: "#c39a4e", dim: "#26200f" },
        reject: { DEFAULT: "#c15f66", dim: "#2a1417" },
        accent: { DEFAULT: "#e8e2d5", dim: "#1e1c18" },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
      fontSize: {
        // Editorial scale: a wide gap between the labels and the numbers, because
        // in this design the number IS the interface. Nothing sits in the middle
        // of the ramp — text is either quiet chrome or the thing you came to read.
        // No tracking on the size itself. This step is used for two different
        // things — uppercase micro-labels and the explanatory sentence under a
        // chart — and 0.11em is correct for the first and actively unreadable
        // for the second. `.label` carries its own tracking, so letter-spacing
        // now belongs to the *role* rather than to the size, and prose at this
        // step reads as prose.
        "2xs": ["10.5px", { lineHeight: "15px" }],
        xs: ["12px", { lineHeight: "18px" }],
        sm: ["13px", { lineHeight: "20px" }],
        base: ["14px", { lineHeight: "22px" }],
        lg: ["16px", { lineHeight: "24px", letterSpacing: "-0.006em" }],
        xl: ["19px", { lineHeight: "26px", letterSpacing: "-0.014em" }],
        "2xl": ["26px", { lineHeight: "32px", letterSpacing: "-0.02em" }],
        "3xl": ["34px", { lineHeight: "38px", letterSpacing: "-0.025em" }],
        // The heroes.
        display: ["58px", { lineHeight: "58px", letterSpacing: "-0.035em" }],
        "display-lg": ["76px", { lineHeight: "74px", letterSpacing: "-0.04em" }],
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

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
        // Surface ramp: true black, and even steps up from it.
        //
        // The ground used to be #0a0a0c — a near-black with a little blue in it,
        // on the reasoning that a UI you read for an hour wants some lift under
        // the text. That is true of a *document*. It is not true of an
        // instrument: the moment the accent became a saturated orange, every
        // millilumen in the ground was light competing with the one colour that
        // is supposed to be the brightest thing on the page. Black gives the
        // orange the whole dynamic range, and it is what makes the hairline grid
        // read as structure rather than as noise.
        //
        // Neutral now, not cool. A blue-black ground under a warm accent reads
        // as two palettes stacked; at zero saturation the ground has no opinion
        // and the accent is the only hue in the interface that is not a verdict.
        ink: {
          1000: "#000000",
          950: "#000000", // the ground, hero and app alike
          900: "#0c0c0c", // one step up: the raised block, a card, a table head
          850: "#121212",
          800: "#181818",
          750: "#202020",
          700: "#2a2a2a",
          600: "#3d3d3d",
        },
        edge: {
          DEFAULT: "#1c1c1c", // hairline between surfaces, and the page grid
          strong: "#2e2e2e", // hairline that needs to be seen
        },
        // Verdicts, and the one interactive hue.
        //
        // The verdict triad is pigment rather than screen — verdigris, yellow
        // ochre, red earth — held around 55-62% lightness and under 45%
        // saturation. That is where print keeps its warning colours, and on a
        // near-black ground it is the difference between colours that *sit* and
        // colours that emit. It is unchanged, because the traffic-light set is
        // what a reviewer already reads without being taught.
        //
        // The accent is not. It used to be warm bone — no hue at all — on the
        // argument that the verdicts had taken green through red and every hue
        // left over read as a default template. The way out of that was never a
        // different hue; it was a different *axis*. This orange is at 100%
        // saturation and the verdicts are all under 45%, so the two can share a
        // neighbourhood on the wheel and still never be confused: **saturation
        // means you can act on it, hue means a verdict.** A vivid orange next to
        // an ochre REVIEW does not read as a brighter REVIEW, it reads as a
        // different kind of thing — which is exactly what it is.
        pass: { DEFAULT: "#5a9e79", dim: "#12241c" },
        review: { DEFAULT: "#c39a4e", dim: "#26200f" },
        reject: { DEFAULT: "#c15f66", dim: "#2a1417" },
        accent: {
          DEFAULT: "#ff4d17",
          soft: "#ff7a4d", // hover, and the one step brighter a link gets
          dim: "#2a0f06", // a wash behind accented text; never a fill on its own
        },
        // The text ramp, neutral to match the ground.
        //
        // These override Tailwind's `slate`, which is a blue-grey — correct
        // against the old blue-black ground, and visibly cold against a true
        // black one with an orange accent. Same names, same steps, no hue, so
        // every `text-slate-500` in the app moved without being touched.
        slate: {
          50: "#fcfcfc",
          100: "#f2f2f2",
          200: "#dedede",
          300: "#c2c2c2",
          400: "#9c9c9c",
          500: "#7a7a7a",
          600: "#585858",
          700: "#3f3f3f",
          800: "#2a2a2a",
          900: "#1a1a1a",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
        // The display face: a bitmap monospace, used at title sizes and at the
        // 10-11px micro-label size it was actually drawn for. Never in a table —
        // it has no tabular figures worth the name, and a column of numbers is a
        // job for JetBrains Mono.
        display: ["var(--font-display)", "var(--font-mono)", "ui-monospace", "monospace"],
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

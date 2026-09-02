/**
 * The palette, in the one place it is allowed to be a hex string.
 *
 * Tailwind owns colour everywhere it can, but three things cannot read a
 * Tailwind class and need a literal: Recharts props, SVG `stroke`/`fill`
 * attributes, and canvas. Those had each grown their own copy of the values,
 * which is how the charts ended up still drawing the previous palette after the
 * tokens moved. They import from here now; `tailwind.config.ts` is the only
 * other definition, and the two are meant to be read side by side.
 *
 * The verdict hues are pigment rather than screen - verdigris, yellow ochre,
 * red earth - held around 55-62% lightness and under 45% saturation. That is
 * where print keeps warning colours, and it is what leaves a genuine alarm
 * somewhere brighter to go.
 *
 * The accent is the opposite: fully saturated orange, on a ground with no
 * saturation at all. That is the whole separation rule - **saturation means you
 * can act on it, hue means a verdict** - and it is why a vivid orange edge in
 * the ring graph is never mistaken for a muted red-earth REJECT node sitting
 * next to it, despite the two being neighbours on the wheel.
 */
export const INK = {
  1000: "#000000",
  950: "#000000",
  900: "#0c0c0c",
  850: "#121212",
  800: "#181818",
  750: "#202020",
} as const;

export const EDGE = "#1c1c1c";
export const EDGE_STRONG = "#2e2e2e";

export const ACCENT = "#ff4d17";
export const ACCENT_SOFT = "#ff7a4d";
export const PASS = "#5a9e79";
export const REVIEW = "#c39a4e";
export const REJECT = "#c15f66";

export const VERDICT_HEX = { PASS, REVIEW, REJECT } as const;

/**
 * Categorical series, for the ROC panel where each line is one detector.
 *
 * Six nominal categories with no order between them, so the ramp is hue-varied
 * at near-constant lightness - no series reads as "more" than another, which a
 * light-to-dark ramp would have implied. It deliberately excludes violet and
 * magenta: at this lightness they are the two that look synthetic next to earth
 * pigments, and they are also the pair most likely to be read as a state rather
 * than a category.
 *
 * The accent leads it, which makes the fused curve the figure and the
 * per-detector curves the ground - what the panel is actually asking you to
 * compare. Every other member is desaturated, so the lead never has a rival.
 */
export const SERIES = [ACCENT, PASS, REVIEW, "#b0725a", "#8b8b8b", "#6f8f8a"] as const;

/** Chart chrome. Both sit deliberately below the data in contrast. */
export const AXIS_STROKE = "#4a4a4a";
export const GRID_STROKE = "#1a1a1a";

/** Text ramp, for the same three consumers that cannot read a class. Mirrors
 *  the neutral `slate` override in `tailwind.config.ts`. */
export const TEXT = {
  100: "#f2f2f2",
  300: "#c2c2c2",
  400: "#9c9c9c",
  500: "#7a7a7a",
  600: "#585858",
} as const;

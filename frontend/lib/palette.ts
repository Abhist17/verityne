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
 * The hues are pigment rather than screen — verdigris, yellow ochre, red earth,
 * ink blue — held around 55-62% lightness and under 45% saturation. That is
 * where print keeps warning colours, and it is what leaves a genuine alarm
 * somewhere brighter to go.
 */
export const INK = {
  950: "#0a0a0c",
  900: "#0e0e12",
  850: "#131318",
  800: "#191920",
  750: "#1f2028",
} as const;

export const EDGE = "#20212a";
export const EDGE_STRONG = "#2d2f3b";

export const ACCENT = "#5b83a8";
export const PASS = "#5a9e79";
export const REVIEW = "#c39a4e";
export const REJECT = "#c15f66";

export const VERDICT_HEX = { PASS, REVIEW, REJECT } as const;

/**
 * Categorical series, for the ROC panel where each line is one detector.
 *
 * Six nominal categories with no order between them, so the ramp is hue-varied
 * at near-constant lightness — no series reads as "more" than another, which a
 * light-to-dark ramp would have implied. It deliberately excludes violet and
 * magenta: at this lightness they are the two that look synthetic next to earth
 * pigments, and they are also the pair most likely to be read as a state rather
 * than a category.
 */
export const SERIES = [ACCENT, PASS, REVIEW, "#b0725a", "#8a8f9e", "#6f8f8a"] as const;

/** Chart chrome. Both sit deliberately below the data in contrast. */
export const AXIS_STROKE = "#4a4d5a";
export const GRID_STROKE = "#1b1c24";

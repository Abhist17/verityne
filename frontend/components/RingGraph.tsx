"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ThreatEdge, ThreatGraph, ThreatNode, Verdict } from "@/lib/api";
import { ACCENT, EDGE, TEXT, VERDICT_HEX } from "@/lib/palette";

/**
 * Fraud rings as a node-link diagram, drawn as a schematic rather than a scene.
 *
 * **The panel is about the rings, and it has to look like it.** The first
 * version laid every submission out in one force cloud, which meant that on a
 * window with two rings and twenty-one unlinked applicants, the twenty-one
 * carried nine tenths of the ink and the answer was two faint pairs somewhere
 * inside the confetti. Unlinked submissions are the *absence* of the finding, so
 * they are separated out: rings own the canvas, and everything unlinked is a
 * quiet band along the bottom that says how many there are and otherwise gets
 * out of the way.
 *
 * **Nothing here glows.** The version before this one drew every node with a
 * Gaussian bloom, wrapped each cluster in a radial-gradient halo, breathed the
 * proven ones in and out and ran a lit particle down each proven edge. Each of
 * those was defensible on its own and together they were a screensaver: the
 * blur put a soft ramp around every hard fact, the halos made cluster
 * *boundaries* — the one thing a linkage graph has to state precisely — into
 * gradients with no edge at all, and the four simultaneous animations meant the
 * eye never settled anywhere long enough to read a label. A forensic claim
 * should be drawn the way a plan is drawn: hard edges, one weight of line, and
 * every mark on a lattice.
 *
 * So: every coordinate snaps to a 4px grid, every node is a square with
 * `crispEdges` on it, every edge is a straight 1px line, and the canvas carries
 * a faint dot lattice underneath so the snapping reads as *registration* rather
 * than as a rounding error. There is no animation at all, which also retires
 * the reduced-motion special case this file used to need — SMIL cannot see a
 * media query, so the preference had to be read in JS and the elements simply
 * not rendered.
 *
 * **Edge style is the evidence type, and that is the point of the panel.** A
 * ring built on byte-identical files and one built on face similarity look the
 * same in a plain graph and are not the same claim: a shared SHA-256 is a fact,
 * while a face match is a similarity search whose false-accept rate compounds
 * with database size (see linkage.SAME_PERSON). Exact matches are drawn solid
 * in the accent and pull tighter; similarity links are dotted and grey. A ring
 * resting on at least one exact match gets accent corner brackets; an inferred
 * one gets grey.
 *
 * Verdict is carried on the node fill — a *status* encoding, which is a reserved
 * role, so it never doubles as a series colour, and it is never colour-alone:
 * every node names its verdict on hover and the legend spells the states out.
 * The accent cannot be confused with any of the three despite sitting between
 * ochre and red-earth on the wheel, because it is the only saturated thing on
 * the canvas.
 */

const VERDICT_FILL: Record<Verdict, string> = {
  PASS: VERDICT_HEX.PASS,
  REVIEW: VERDICT_HEX.REVIEW,
  REJECT: VERDICT_HEX.REJECT,
};

interface Sim extends ThreatNode {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/** The drawing lattice, in screen pixels. Every mark lands on it. */
const LATTICE = 4;
const snap = (v: number) => Math.round(v / LATTICE) * LATTICE;

/**
 * Node side from risk score, in lattice units.
 *
 * Area-proportional within a readable band, so a 0.9 does not swamp a 0.3 the
 * way a linear side would, then quantised: a square whose side is 13.7px has a
 * soft edge on a fractional device pixel, which is the exact thing this drawing
 * is not allowed to have. The result is five discrete sizes, 8px to 24px, which
 * is also a more honest read of a score than a continuous ramp nobody can
 * measure by eye.
 */
const sideOf = (score: number, k = 1) => {
  const r = 5 + Math.sqrt(Math.max(0, Math.min(1, score))) * 7;
  return Math.max(8, snap(r * 2 * Math.min(1.25, k)));
};

/** Padding between the outermost node and its ring's bracket, in screen px. */
const BOX_PAD = 22;
/** Arm length of a corner bracket. */
const BRACKET = 11;

export function RingGraph({
  graph,
  height = 460,
  onSelect,
  selected,
}: {
  graph: ThreatGraph;
  height?: number;
  onSelect?: (id: string | null) => void;
  selected?: string | null;
}) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(760);
  const [hover, setHover] = useState<{ node: Sim; x: number; y: number } | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /** Linked submissions carry the finding; unlinked ones are its absence. */
  const linked = useMemo(() => graph.nodes.filter((n) => n.ring !== null), [graph.nodes]);
  const isolated = useMemo(() => graph.nodes.filter((n) => n.ring === null), [graph.nodes]);

  const ringIds = useMemo(
    () => Array.from(new Set(linked.map((n) => n.ring as number))).sort((a, b) => a - b),
    [linked]
  );

  const byId = useMemo(() => {
    const m = new Map<string, number>();
    linked.forEach((n, i) => m.set(n.id, i));
    return m;
  }, [linked]);

  /**
   * The settled layout, computed synchronously and only over the linked nodes.
   *
   * Each ring is given its own anchor on a circle and its members are pulled
   * toward it, so clusters land apart from one another instead of being
   * negotiated into one blob by repulsion alone. Rings are placed largest-first
   * so the biggest cluster gets the most room.
   *
   * It runs inline rather than across `requestAnimationFrame`: rAF does not fire
   * in an unpainted tab, so a page opened in the background — or screenshotted
   * by a test — used to render an empty graph and stay empty. It is also
   * deterministic, seeded from ring index and node order rather than
   * `Math.random`, because a graph that rearranges itself between reloads is
   * useless for walking a reviewer through a case.
   */
  const nodes = useMemo<Sim[]>(() => {
    const n = linked.length;
    if (!n) return [];

    const anchors = new Map<number, { x: number; y: number }>();
    // Anchors start at angle 0, not -90 degrees. Panels are wider than they are
    // tall, and two rings placed at -90/+90 stack vertically — which wasted the
    // whole horizontal axis and shrank both clusters to fit the height.
    // Spread scales with how many clusters there are. A fixed large value threw
    // two small rings to opposite edges with an empty middle, because the fit
    // then zoomed the whole cloud to the panel width.
    const spread = ringIds.length === 1 ? 0 : 60 + ringIds.length * 22;
    ringIds.forEach((r, i) => {
      const a = (i / ringIds.length) * Math.PI * 2;
      anchors.set(r, { x: Math.cos(a) * spread, y: Math.sin(a) * spread * 0.55 });
    });

    const pts: Sim[] = linked.map((node, i) => {
      const anchor = anchors.get(node.ring as number)!;
      const a = (i % 8) * (Math.PI / 4);
      return {
        ...node,
        x: anchor.x + Math.cos(a) * 30,
        y: anchor.y + Math.sin(a) * 30,
        vx: 0,
        vy: 0,
      };
    });

    const links = graph.edges
      .map((e) => ({ s: byId.get(e.source), t: byId.get(e.target), kind: e.kind }))
      .filter((l): l is { s: number; t: number; kind: ThreatEdge["kind"] } =>
        l.s !== undefined && l.t !== undefined
      );

    const iterations = Math.max(140, Math.min(420, Math.round((420 * 40) / Math.max(1, n))));
    let alpha = 1;

    for (let step = 0; step < iterations; step++) {
      alpha *= 0.986;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = pts[i];
          const b = pts[j];
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let d2 = dx * dx + dy * dy;
          if (d2 < 1e-4) {
            dx = (i % 7) - 3;
            dy = (j % 7) - 3;
            d2 = dx * dx + dy * dy || 1;
          }
          const d = Math.sqrt(d2);
          // Members of different rings push apart harder, so clusters separate.
          const cross = a.ring === b.ring ? 1 : 2.6;
          const rep = (760 * alpha * cross) / d2;
          a.vx -= (dx / d) * rep;
          a.vy -= (dy / d) * rep;
          b.vx += (dx / d) * rep;
          b.vy += (dy / d) * rep;
        }
      }
      for (const l of links) {
        const a = pts[l.s];
        const b = pts[l.t];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1;
        // A proven cluster should read tighter than an inferred one, reinforcing
        // the solid/dotted distinction with spacing as well as stroke.
        const rest = l.kind === "asset_exact" ? 44 : 70;
        const k = (d - rest) * 0.05 * alpha;
        a.vx += (dx / d) * k;
        a.vy += (dy / d) * k;
        b.vx -= (dx / d) * k;
        b.vy -= (dy / d) * k;
      }
      for (const p of pts) {
        const anchor = anchors.get(p.ring as number)!;
        p.vx += (anchor.x - p.x) * 0.02 * alpha;
        p.vy += (anchor.y - p.y) * 0.02 * alpha;
        p.vx *= 0.82;
        p.vy *= 0.82;
        p.x += p.vx;
        p.y += p.vy;
      }
    }
    return pts;
  }, [linked, ringIds, graph.edges, byId]);

  /** The band along the bottom is reserved for unlinked submissions. */
  const bandH = isolated.length ? 62 : 0;
  const graphH = height - bandH;

  const view = useMemo(() => {
    if (!nodes.length) return { tx: width / 2, ty: graphH / 2, k: 1 };
    // Padding covers the bracket drawn around a cluster and the label above it,
    // both of which extend past the outermost node centre. Fitting to centres
    // alone clipped the top label off the canvas.
    const pad = BOX_PAD + 34;
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const k = Math.min(
      (width - pad * 2) / Math.max(1, maxX - minX),
      (graphH - pad * 2) / Math.max(1, maxY - minY),
      // Capped low: a two-node graph zoomed to fill the panel reads as a poster,
      // not a measurement, and leaves the clusters marooned at the edges.
      1.5
    );
    return { tx: width / 2 - ((minX + maxX) / 2) * k, ty: graphH / 2 - ((minY + maxY) / 2) * k, k };
  }, [nodes, width, graphH]);

  /** Layout space → screen, then onto the lattice. Everything downstream — node
   *  corners, edge endpoints, bracket boxes — derives from this, so one snap
   *  here is what keeps the whole drawing registered. */
  const pos = useCallback(
    (n: Sim) => ({ x: snap(n.x * view.k + view.tx), y: snap(n.y * view.k + view.ty) }),
    [view]
  );

  const posById = useMemo(() => {
    const m = new Map<string, { x: number; y: number }>();
    nodes.forEach((n) => m.set(n.id, pos(n)));
    return m;
  }, [nodes, pos]);

  /**
   * One box per ring: an axis-aligned bounding rectangle over its members.
   *
   * A rectangle rather than the circular hull this used to draw. A circle around
   * a cluster of four is mostly empty canvas — it claims a lot of area to
   * enclose very little, and two neighbouring rings' circles overlap long before
   * their members do. A bounding box claims exactly what the ring occupies, it
   * sits square on the lattice, and its corners are the natural place to hang a
   * bracket.
   */
  const boxes = useMemo(() => {
    return ringIds.map((r) => {
      const members = nodes.filter((n) => n.ring === r);
      const px = members.map((m) => {
        const p = pos(m);
        const half = sideOf(m.score, view.k) / 2;
        return { ...p, half };
      });
      const x0 = snap(Math.min(...px.map((p) => p.x - p.half)) - BOX_PAD);
      const x1 = snap(Math.max(...px.map((p) => p.x + p.half)) + BOX_PAD);
      const y0 = snap(Math.min(...px.map((p) => p.y - p.half)) - BOX_PAD);
      const y1 = snap(Math.max(...px.map((p) => p.y + p.half)) + BOX_PAD);
      const meta = graph.rings.find((x) => x.id === r);
      // The label is monospace, so its width is arithmetic — 10px at Departure
      // Mono's advance plus 0.1em of tracking is a shade under 8px a character.
      // The estimate is deliberately generous: it is only used to decide whether
      // the label still fits left-aligned to its box, and over-estimating flips
      // it to the right-hand edge a little early, which is correct-looking.
      // Under-estimating would run it off the canvas, which is not.
      const label =
        `RING ${r + 1} / ${members.length}` +
        (meta && meta.merchants.length > 1 ? ` / ${meta.merchants.length} MERCHANTS` : "") +
        (meta?.has_exact_asset_reuse ? " / PROVEN" : "");
      return { r, x0, y0, x1, y1, meta, size: members.length, labelW: label.length * 7.8 };
    });
  }, [ringIds, nodes, pos, view.k, graph.rings]);

  /** Unlinked nodes, evenly spaced along the band. */
  const bandNodes = useMemo(() => {
    if (!isolated.length) return [];
    const inset = 18;
    const usable = Math.max(1, width - inset * 2);
    const step = isolated.length > 1 ? usable / (isolated.length - 1) : 0;
    const y = snap(graphH + bandH / 2 + 4);
    return isolated.map((n, i) => ({
      node: n,
      x: snap(isolated.length > 1 ? inset + i * step : width / 2),
      y,
    }));
  }, [isolated, width, graphH, bandH]);

  const dim = (id: string) =>
    selected != null && selected !== id && !graph.edges.some(
      (e) =>
        (e.source === selected && e.target === id) || (e.target === selected && e.source === id)
    );

  const empty = !nodes.length;

  return (
    // `w-full min-w-0` is load-bearing. The SVG takes a pixel width measured
    // from this element, and a grid or flex child defaults to min-content width
    // — so on a narrow viewport the box never shrank, the observer kept
    // reporting the 760px seed, and the graph pushed the whole page 394px wide.
    <div ref={wrapRef} className="relative w-full min-w-0">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={
          `Fraud-ring graph: ${graph.rings.length} ring(s) across ${linked.length} linked ` +
          `submissions, ${isolated.length} unlinked`
        }
        className="block touch-none select-none"
        shapeRendering="crispEdges"
        onMouseLeave={() => setHover(null)}
      >
        {/* The plotting surface: one pixel every 8, in the same hairline colour
            as the page grid behind it. It is barely visible and it is the reason
            the snapping reads as deliberate — without it, squares that happen to
            align look like a coincidence rather than a register. */}
        <defs>
          <pattern id="rg-lattice" width="8" height="8" patternUnits="userSpaceOnUse">
            <rect x="0" y="0" width="1" height="1" fill={EDGE} />
          </pattern>
        </defs>
        <rect x={0} y={0} width={width} height={graphH} fill="url(#rg-lattice)" />

        {/* ---- ring boxes, behind everything ---------------------------
             A hairline rectangle for the extent, and four corner brackets for
             the claim. The brackets are the encoding: accent means at least one
             byte-identical file holds this cluster together, grey means the
             whole thing rests on similarity. */}
        <g>
          {boxes.map((b) => {
            const proven = b.meta?.has_exact_asset_reuse;
            const faded = selected != null && !nodes.some((n) => n.ring === b.r && n.id === selected);
            const tone = proven ? ACCENT : TEXT[600];
            const overflows = b.x0 + b.labelW > width - 2;
            const corners: [number, number, number, number][] = [
              // x, y, dx, dy — the direction each arm runs from the corner.
              [b.x0, b.y0, 1, 1],
              [b.x1, b.y0, -1, 1],
              [b.x0, b.y1, 1, -1],
              [b.x1, b.y1, -1, -1],
            ];
            return (
              <g key={b.r} opacity={faded ? 0.3 : 1}>
                <rect
                  x={b.x0}
                  y={b.y0}
                  width={b.x1 - b.x0}
                  height={b.y1 - b.y0}
                  fill="none"
                  stroke={EDGE}
                  strokeWidth={1}
                />
                {corners.map(([x, y, dx, dy], i) => (
                  <path
                    key={i}
                    d={`M ${x + dx * BRACKET} ${y} L ${x} ${y} L ${x} ${y + dy * BRACKET}`}
                    fill="none"
                    stroke={tone}
                    strokeWidth={proven ? 1.5 : 1}
                    strokeOpacity={proven ? 1 : 0.8}
                  />
                ))}
                {/* The label sits on the top rule, left-aligned to the box, the
                    way a callout sits on a drawing — until it would not fit,
                    at which point it anchors to the right-hand edge of the
                    canvas instead of being slid leftward by a guess.

                    Sliding is what the first attempt did, against a constant
                    190px inset. That was right for "RING 5 / 2" and wrong for
                    "RING 1 / 3 / 3 MERCHANTS / PROVEN" at 33 characters, so the
                    longest label — on the ring carrying the most evidence — was
                    the one that ran off the edge. Switching the anchor cannot
                    clip whatever the estimate says, because the end anchor is
                    measured by the renderer rather than by us. */}
                <text
                  x={overflows ? width - 2 : Math.max(2, b.x0)}
                  textAnchor={overflows ? "end" : "start"}
                  y={Math.max(11, b.y0 - 8)}
                  className="fill-slate-500"
                  style={{
                    fontSize: 10,
                    fontFamily: "var(--font-display), var(--font-mono), monospace",
                    letterSpacing: "0.1em",
                  }}
                >
                  <tspan fill={proven ? ACCENT : TEXT[500]}>{`RING ${b.r + 1}`}</tspan>
                  <tspan>{` / ${b.size}`}</tspan>
                  {b.meta && b.meta.merchants.length > 1 && (
                    <tspan>{` / ${b.meta.merchants.length} MERCHANTS`}</tspan>
                  )}
                  {proven && <tspan fill={ACCENT}>{" / PROVEN"}</tspan>}
                </text>
              </g>
            );
          })}
        </g>

        {/* ---- edges ----------------------------------------------------
             Straight, one pixel, no arc. The bowed chords the previous version
             drew were there to separate parallel edges between the same pair —
             a real problem, solved by a curve that made every link look like a
             flight path. Parallel edges are rare enough here that the honest
             fix is to draw them straight and let the pair overlap; what matters
             is which *kind* of evidence connects two nodes, and that is carried
             by the stroke. */}
        <g shapeRendering="auto">
          {graph.edges.map((e, i) => {
            const a = posById.get(e.source);
            const b = posById.get(e.target);
            if (!a || !b) return null;
            const exact = e.kind === "asset_exact";
            const faded = selected != null && e.source !== selected && e.target !== selected;
            return (
              <line
                key={i}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={exact ? ACCENT : TEXT[600]}
                strokeWidth={1}
                strokeDasharray={exact ? undefined : "1 3"}
                strokeOpacity={faded ? 0.12 : exact ? 0.85 : 0.7}
              />
            );
          })}
        </g>

        {/* ---- linked nodes --------------------------------------------- */}
        <g>
          {nodes.map((n) => {
            const p = pos(n);
            const s = sideOf(n.score, view.k);
            const faded = dim(n.id);
            const isSel = selected === n.id;
            const isHot = hover?.node.id === n.id;
            // The marker on a selected node, lifted straight from the way a
            // survey drawing calls out a station: a square outline standing off
            // the mark, with a tick on each side. It reads as "this one" at a
            // glance and it adds no colour the palette does not already own.
            const halo = s / 2 + 6;
            return (
              <g key={n.id}>
                <rect
                  x={p.x - s / 2}
                  y={p.y - s / 2}
                  width={s}
                  height={s}
                  fill={VERDICT_FILL[n.verdict]}
                  fillOpacity={faded ? 0.18 : 1}
                  stroke={isHot ? "#ffffff" : "none"}
                  strokeWidth={1}
                  className="cursor-pointer"
                  onMouseEnter={() => setHover({ node: n, x: p.x, y: p.y })}
                  onClick={() => onSelect?.(isSel ? null : n.id)}
                />
                {isSel && (
                  <g pointerEvents="none">
                    <rect
                      x={p.x - halo}
                      y={p.y - halo}
                      width={halo * 2}
                      height={halo * 2}
                      fill="none"
                      stroke={ACCENT}
                      strokeWidth={1}
                    />
                    {[
                      [p.x, p.y - halo - 5, p.x, p.y - halo],
                      [p.x, p.y + halo, p.x, p.y + halo + 5],
                      [p.x - halo - 5, p.y, p.x - halo, p.y],
                      [p.x + halo, p.y, p.x + halo + 5, p.y],
                    ].map(([x1, y1, x2, y2], i) => (
                      <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={ACCENT} strokeWidth={1} />
                    ))}
                  </g>
                )}
              </g>
            );
          })}
        </g>

        {/* ---- the unlinked band ----------------------------------------
             Deliberately small, dim and on one line. These submissions are the
             absence of a finding; they belong on screen for honesty — the window
             is not all rings — but they must not out-weigh the rings. */}
        {isolated.length > 0 && (
          <g>
            <line
              x1={0}
              y1={graphH + 6}
              x2={width}
              y2={graphH + 6}
              stroke={EDGE}
              strokeWidth={1}
            />
            <text
              x={0}
              y={graphH + 24}
              className="fill-slate-600"
              style={{
                fontSize: 10,
                fontFamily: "var(--font-display), var(--font-mono), monospace",
                letterSpacing: "0.12em",
              }}
            >
              {`${isolated.length} UNLINKED`}
            </text>
            {bandNodes.map(({ node, x, y }) => (
              <rect
                key={node.id}
                x={x - 3}
                y={y - 3}
                width={6}
                height={6}
                fill={VERDICT_FILL[node.verdict]}
                fillOpacity={selected != null && selected !== node.id ? 0.2 : 0.55}
                className="cursor-pointer"
                onMouseEnter={() =>
                  setHover({ node: { ...node, x: 0, y: 0, vx: 0, vy: 0 }, x, y })
                }
                onClick={() => onSelect?.(selected === node.id ? null : node.id)}
              />
            ))}
          </g>
        )}

        {empty && (
          <text
            x={width / 2}
            y={graphH / 2}
            textAnchor="middle"
            className="fill-slate-600"
            style={{ fontSize: 12 }}
            shapeRendering="auto"
          >
            No linked submissions in this window — nothing shares a face or a file.
          </text>
        )}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute z-10 w-56 border border-edge-strong bg-ink-950/95 p-2 backdrop-blur-sm"
          style={{
            left: Math.min(Math.max(0, hover.x + 12), Math.max(0, width - 232)),
            top: Math.max(0, hover.y - 8),
          }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="truncate text-xs font-medium text-slate-200">
              {hover.node.claimed_name ?? "unnamed"}
            </span>
            <span className="num text-2xs" style={{ color: VERDICT_FILL[hover.node.verdict] }}>
              {hover.node.verdict}
            </span>
          </div>
          <div className="num mt-1 space-y-0.5 text-2xs text-slate-500">
            <div>risk {hover.node.score.toFixed(2)}</div>
            <div className="truncate">{hover.node.merchant_id}</div>
            {hover.node.generator_guess && (
              <div className="truncate text-slate-400">{hover.node.generator_guess}</div>
            )}
            <div>{hover.node.ring !== null ? `ring ${hover.node.ring + 1}` : "unlinked"}</div>
          </div>
        </div>
      )}
    </div>
  );
}

/** Spelled out in words, because the graph encodes two different things in
 *  colour and stroke and neither is guessable. The swatches are squares and the
 *  rules are hairlines, so the key is drawn in the same hand as the drawing. */
export function RingGraphLegend({ threshold }: { threshold: ThreatGraph["threshold"] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-2xs text-slate-500">
      {(["PASS", "REVIEW", "REJECT"] as Verdict[]).map((v) => (
        <span key={v} className="flex items-center gap-1.5">
          <span className="h-2 w-2" style={{ background: VERDICT_FILL[v] }} />
          {v.toLowerCase()}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <svg width="18" height="6" aria-hidden>
          <line x1="0" y1="3" x2="18" y2="3" stroke={ACCENT} strokeWidth="1" />
        </svg>
        byte-identical file — a fact
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="18" height="6" aria-hidden>
          <line x1="0" y1="3" x2="18" y2="3" stroke={TEXT[600]} strokeWidth="1" strokeDasharray="1 3" />
        </svg>
        similarity — an inference
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="12" height="12" aria-hidden shapeRendering="crispEdges">
          <path d="M 5 0 L 0 0 L 0 5" fill="none" stroke={ACCENT} strokeWidth="1.5" />
          <path d="M 7 0 L 12 0 L 12 5" fill="none" stroke={ACCENT} strokeWidth="1.5" />
          <path d="M 5 12 L 0 12 L 0 7" fill="none" stroke={ACCENT} strokeWidth="1.5" />
          <path d="M 7 12 L 12 12 L 12 7" fill="none" stroke={ACCENT} strokeWidth="1.5" />
        </svg>
        ring holds on a fact
      </span>
      <span className="num ml-auto">edges at cos ≥ {threshold.same_person.toFixed(4)}</span>
    </div>
  );
}

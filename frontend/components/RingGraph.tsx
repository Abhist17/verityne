"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import type { ThreatEdge, ThreatGraph, ThreatNode, Verdict } from "@/lib/api";
import { AXIS_STROKE, INK, REJECT, VERDICT_HEX } from "@/lib/palette";

/**
 * Fraud rings as a node-link diagram.
 *
 * **The panel is about the rings, and it has to look like it.** The first
 * version laid every submission out in one force cloud, which meant that on a
 * window with two rings and twenty-one unlinked applicants, the twenty-one
 * carried nine tenths of the ink and the answer was two faint pairs somewhere
 * inside the confetti. Unlinked submissions are the *absence* of the finding, so
 * they are now separated out: rings own the canvas, laid out one cluster per
 * ring with a hull drawn behind them, and everything unlinked is a quiet band
 * along the bottom that says how many there are and otherwise gets out of the
 * way.
 *
 * **Edge style is the evidence type, and that is the point of the panel.** A
 * ring built on byte-identical files and one built on face similarity look the
 * same in a plain graph and are not the same claim: a shared SHA-256 is a fact,
 * while a face match is a similarity search whose false-accept rate compounds
 * with database size (see linkage.SAME_PERSON). Exact matches are drawn solid
 * and pull tighter; similarity links are dashed. A ring resting on at least one
 * exact match is outlined; an inferred one is not.
 *
 * Verdict is carried on the node fill — a *status* encoding, which is a reserved
 * role, so it never doubles as a series colour, and it is never colour-alone:
 * every node names its verdict on hover and the legend spells the states out.
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

/** Node radius from risk score. Area-proportional within a readable band, so a
 *  0.9 does not swamp a 0.3 the way a linear radius would. */
const radiusOf = (score: number) => 5 + Math.sqrt(Math.max(0, Math.min(1, score))) * 7;

/** Ring hull padding, in layout units. */
const HULL_PAD = 26;

/** Whether this viewer has asked for less motion.
 *
 *  SVG animation elements do not honour `prefers-reduced-motion` the way a CSS
 *  animation does — there is no media query that reaches inside SMIL — so the
 *  preference has to be read and the elements simply not rendered. The graph is
 *  built to be fully legible without any of it: the pulses and the breathing
 *  halo are emphasis, never the encoding. */
function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const sync = () => setReduced(mq.matches);
    sync();
    mq.addEventListener?.("change", sync);
    return () => mq.removeEventListener?.("change", sync);
  }, []);
  return reduced;
}

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
  const stillness = useReducedMotion();

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

    const order = new Map(ringIds.map((r, i) => [r, i]));
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
        // the solid/dashed distinction with spacing as well as stroke.
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
    // Padding covers the hull drawn around a cluster and the label above it,
    // both of which extend past the outermost node centre. Fitting to centres
    // alone clipped the top label off the canvas.
    const pad = HULL_PAD + 34;
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

  const pos = useCallback(
    (n: Sim) => ({ x: n.x * view.k + view.tx, y: n.y * view.k + view.ty }),
    [view]
  );

  const posById = useMemo(() => {
    const m = new Map<string, { x: number; y: number }>();
    nodes.forEach((n) => m.set(n.id, pos(n)));
    return m;
  }, [nodes, pos]);

  /** One hull per ring: centre, radius, and whether the ring is proven. */
  const hulls = useMemo(() => {
    return ringIds.map((r) => {
      const members = nodes.filter((n) => n.ring === r);
      const px = members.map((m) => pos(m));
      const cx = px.reduce((s, p) => s + p.x, 0) / px.length;
      const cy = px.reduce((s, p) => s + p.y, 0) / px.length;
      const rad =
        Math.max(...px.map((p, i) => Math.hypot(p.x - cx, p.y - cy) + radiusOf(members[i].score))) +
        HULL_PAD;
      const meta = graph.rings.find((x) => x.id === r);
      return { r, cx, cy, rad, meta, size: members.length };
    });
  }, [ringIds, nodes, pos, graph.rings]);

  /** Unlinked nodes, evenly spaced along the band. */
  const bandNodes = useMemo(() => {
    if (!isolated.length) return [];
    const inset = 18;
    const usable = Math.max(1, width - inset * 2);
    const step = isolated.length > 1 ? usable / (isolated.length - 1) : 0;
    const y = graphH + bandH / 2 + 4;
    return isolated.map((n, i) => ({
      node: n,
      x: isolated.length > 1 ? inset + i * step : width / 2,
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
        onMouseLeave={() => setHover(null)}
      >
        {/* Bloom, hull gradients and a vignette.
            The glow is a real filter rather than a stack of translucent circles:
            one blur pass merged under the source keeps the node edge crisp while
            the light falls off around it, which is what makes a dark network
            read as lit rather than as flat dots on a background. */}
        <defs>
          <filter id="rg-glow" x="-120%" y="-120%" width="340%" height="340%">
            <feGaussianBlur stdDeviation="4.5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="rg-glow-soft" x="-160%" y="-160%" width="420%" height="420%">
            <feGaussianBlur stdDeviation="9" />
          </filter>
          <radialGradient id="rg-hull-proven">
            <stop offset="0%" stopColor={REJECT} stopOpacity="0.20" />
            <stop offset="70%" stopColor={REJECT} stopOpacity="0.06" />
            <stop offset="100%" stopColor={REJECT} stopOpacity="0" />
          </radialGradient>
          <radialGradient id="rg-hull-inferred">
            <stop offset="0%" stopColor="#8a8f9e" stopOpacity="0.13" />
            <stop offset="70%" stopColor="#8a8f9e" stopOpacity="0.04" />
            <stop offset="100%" stopColor="#8a8f9e" stopOpacity="0" />
          </radialGradient>
          <radialGradient id="rg-vignette">
            <stop offset="0%" stopColor="#20212a" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#20212a" stopOpacity="0" />
          </radialGradient>
        </defs>

        <rect x={0} y={0} width={width} height={graphH} fill="url(#rg-vignette)" />

        {/* ---- ring hulls, behind everything --------------------------- */}
        <g>
          {hulls.map((h) => {
            const proven = h.meta?.has_exact_asset_reuse;
            const faded = selected != null && !nodes.some((n) => n.ring === h.r && n.id === selected);
            return (
              <g key={h.r} opacity={faded ? 0.35 : 1}>
                <circle
                  cx={h.cx}
                  cy={h.cy}
                  r={h.rad}
                  fill={proven ? "url(#rg-hull-proven)" : "url(#rg-hull-inferred)"}
                />
                <circle
                  cx={h.cx}
                  cy={h.cy}
                  r={h.rad}
                  fill="none"
                  stroke={proven ? REJECT : AXIS_STROKE}
                  strokeOpacity={proven ? 0.55 : 0.28}
                  strokeWidth={proven ? 1.4 : 1}
                  // A proven ring is outlined solid; an inferred one is dashed,
                  // the same grammar the edges use. The dash creeps slowly so an
                  // inferred cluster reads as unsettled rather than merely thin.
                  strokeDasharray={proven ? undefined : "5 6"}
                >
                  {!proven && !stillness && (
                    <animate attributeName="stroke-dashoffset" from="0" to="-22"
                             dur="3.4s" repeatCount="indefinite" />
                  )}
                </circle>
                {/* A proven ring breathes: the only motion reserved for a claim
                    that rests on a byte-identical file rather than a similarity. */}
                {proven && !stillness && (
                  <circle cx={h.cx} cy={h.cy} r={h.rad} fill="none" stroke={REJECT}
                          strokeWidth={1} opacity={0}>
                    <animate attributeName="r" values={`${h.rad};${h.rad + 16};${h.rad}`}
                             dur="4.2s" repeatCount="indefinite" />
                    <animate attributeName="opacity" values="0.34;0;0.34"
                             dur="4.2s" repeatCount="indefinite" />
                  </circle>
                )}
                <text
                  // Kept inside the canvas on both sides; a cluster near an edge
                  // would otherwise have half its label cut off.
                  x={Math.min(Math.max(h.rad * 0.35 + 46, h.cx), width - 46)}
                  // Above the hull, but never off the top edge — a cluster that
                  // settles high would otherwise lose its label entirely.
                  y={Math.max(12, h.cy - h.rad - 9)}
                  textAnchor="middle"
                  className="fill-slate-500"
                  style={{ fontSize: 10, fontFamily: "var(--font-mono)", letterSpacing: "0.08em" }}
                >
                  {`RING ${h.r + 1} · ${h.size}`}
                  {h.meta && h.meta.merchants.length > 1 ? ` · ${h.meta.merchants.length} merchants` : ""}
                  {proven ? " · PROVEN" : ""}
                </text>
              </g>
            );
          })}
        </g>

        {/* ---- edges ---------------------------------------------------- */}
        <g>
          {graph.edges.map((e, i) => {
            const a = posById.get(e.source);
            const b = posById.get(e.target);
            if (!a || !b) return null;
            const exact = e.kind === "asset_exact";
            const faded = selected != null && e.source !== selected && e.target !== selected;
            // A gentle arc rather than a straight chord. In a dense cluster two
            // straight edges between the same pair of neighbours overlap into one
            // line; curving them apart keeps the count of links legible.
            const mx = (a.x + b.x) / 2;
            const my = (a.y + b.y) / 2;
            const nx = -(b.y - a.y);
            const ny = b.x - a.x;
            const len = Math.hypot(nx, ny) || 1;
            const bow = Math.min(26, len * 0.16) * (i % 2 === 0 ? 1 : -1);
            const d = `M ${a.x} ${a.y} Q ${mx + (nx / len) * bow} ${my + (ny / len) * bow} ${b.x} ${b.y}`;
            return (
              <g key={i} opacity={faded ? 0.14 : 1}>
                {exact && (
                  <path d={d} fill="none" stroke={REJECT} strokeWidth={5}
                        strokeOpacity={0.28} filter="url(#rg-glow-soft)" />
                )}
                <path
                  d={d}
                  fill="none"
                  stroke={exact ? REJECT : AXIS_STROKE}
                  strokeWidth={exact ? 1.9 : 1.2}
                  strokeDasharray={exact ? undefined : "3 4"}
                  strokeOpacity={exact ? 0.95 : 0.55}
                >
                  {!exact && !stillness && (
                    <animate attributeName="stroke-dashoffset" from="0" to="-14"
                             dur="1.6s" repeatCount="indefinite" />
                  )}
                </path>
                {/* Evidence travelling down a proven link. Motion is spent only
                    where the claim is a fact, so the eye is drawn to the rings
                    that are not an inference. */}
                {exact && !faded && !stillness && (
                  <circle r={2.2} fill={REJECT} filter="url(#rg-glow)">
                    <animateMotion dur="2.6s" repeatCount="indefinite" path={d} />
                    <animate attributeName="opacity" values="0;1;1;0"
                             dur="2.6s" repeatCount="indefinite" />
                  </circle>
                )}
              </g>
            );
          })}
        </g>

        {/* ---- linked nodes --------------------------------------------- */}
        <g>
          {nodes.map((n) => {
            const p = pos(n);
            const r = radiusOf(n.score) * Math.min(1.25, view.k);
            const faded = dim(n.id);
            return (
              <g key={n.id}>
                {!faded && (
                  <circle cx={p.x} cy={p.y} r={r * 1.9} fill={VERDICT_FILL[n.verdict]}
                          opacity={0.16} filter="url(#rg-glow-soft)" />
                )}
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={r}
                  fill={VERDICT_FILL[n.verdict]}
                  fillOpacity={faded ? 0.15 : 0.95}
                  stroke={selected === n.id ? "#f1f5f9" : INK[900]}
                  strokeWidth={2}
                  filter={faded ? undefined : "url(#rg-glow)"}
                  className="cursor-pointer"
                  onMouseEnter={() => setHover({ node: n, x: p.x, y: p.y })}
                  onClick={() => onSelect?.(selected === n.id ? null : n.id)}
                />
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
              stroke={AXIS_STROKE}
              strokeOpacity={0.25}
            />
            <text
              x={0}
              y={graphH + 24}
              className="fill-slate-600"
              style={{ fontSize: 10, fontFamily: "var(--font-mono)", letterSpacing: "0.11em" }}
            >
              {`${isolated.length} UNLINKED`}
            </text>
            {bandNodes.map(({ node, x, y }) => (
              <circle
                key={node.id}
                cx={x}
                cy={y}
                r={3.5}
                fill={VERDICT_FILL[node.verdict]}
                fillOpacity={selected != null && selected !== node.id ? 0.18 : 0.45}
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
          >
            No linked submissions in this window — nothing shares a face or a file.
          </text>
        )}
      </svg>

      {hover && (
        <div
          className="pointer-events-none absolute z-10 w-56 rounded border border-edge-strong bg-ink-950/95 p-2 shadow-lg backdrop-blur-sm"
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
 *  colour and stroke and neither is guessable. */
export function RingGraphLegend({ threshold }: { threshold: ThreatGraph["threshold"] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-2xs text-slate-500">
      {(["PASS", "REVIEW", "REJECT"] as Verdict[]).map((v) => (
        <span key={v} className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full" style={{ background: VERDICT_FILL[v] }} />
          {v.toLowerCase()}
        </span>
      ))}
      <span className="flex items-center gap-1.5">
        <svg width="18" height="6" aria-hidden>
          <line x1="0" y1="3" x2="18" y2="3" stroke={REJECT} strokeWidth="2" />
        </svg>
        byte-identical file — a fact
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="18" height="6" aria-hidden>
          <line x1="0" y1="3" x2="18" y2="3" stroke="#4b5565" strokeWidth="1.25" strokeDasharray="3 3" />
        </svg>
        similarity — an inference
      </span>
      <span className={clsx("ml-auto num")}>
        edges at cos ≥ {threshold.same_person.toFixed(4)}
      </span>
    </div>
  );
}

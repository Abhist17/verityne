"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import type { ThreatEdge, ThreatGraph, ThreatNode, Verdict } from "@/lib/api";
import { AXIS_STROKE, INK, REJECT, VERDICT_HEX } from "@/lib/palette";

/**
 * Fraud rings as a node-link diagram.
 *
 * **Edge style is the evidence type, and that is the whole point of the panel.**
 * A ring built on byte-identical files and a ring built on face similarity look
 * identical in a plain graph, and they are not the same claim: a shared SHA-256
 * is a fact, while a face match is a similarity search whose false-accept rate
 * compounds with the size of the database (see linkage.SAME_PERSON). So an exact
 * asset match is drawn solid and a similarity link is drawn dashed, and a
 * reviewer can tell at a glance whether a cluster is proven or inferred without
 * reading a single number.
 *
 * Verdict is carried on the node fill. That is a *status* encoding, which is a
 * reserved role, so it never doubles as a series colour — and it is never
 * colour-alone: every node names its verdict in the tooltip and the legend
 * spells the three states out in words.
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

  const byId = useMemo(() => {
    const m = new Map<string, number>();
    graph.nodes.forEach((n, i) => m.set(n.id, i));
    return m;
  }, [graph.nodes]);

  /**
   * The settled layout, computed synchronously.
   *
   * This used to run the simulation across `requestAnimationFrame` so you could
   * watch it settle. That is a liability rather than a feature: rAF does not
   * fire while the tab is unpainted, so a page opened in a background tab - or
   * screenshotted by a test - rendered an empty graph and stayed empty, because
   * the one scheduled frame was cancelled by the effect cleanup and nothing
   * rescheduled it. A few hundred iterations over this many nodes costs a few
   * milliseconds, so it runs inline and the layout simply exists by first paint.
   *
   * It is also deterministic: seeded from ring index and node order, never from
   * `Math.random`, so the same submissions lay out the same way on every reload.
   * A graph that rearranges itself between refreshes is useless for talking a
   * reviewer through a case.
   */
  const nodes = useMemo<Sim[]>(() => {
    const n = graph.nodes.length;
    if (!n) return [];

    const ringCount = Math.max(1, new Set(graph.nodes.map((x) => x.ring ?? -1)).size);
    const pts: Sim[] = graph.nodes.map((node, i) => {
      const ring = node.ring ?? -1;
      const a = ((ring + 1) / ringCount) * Math.PI * 2 + (i % 5) * 0.21;
      const jitter = ((i * 37) % 17) / 17 - 0.5;
      return {
        ...node,
        x: Math.cos(a) * 120 + jitter * 60,
        y: Math.sin(a) * 120 + jitter * 60,
        vx: 0,
        vy: 0,
      };
    });

    const links = graph.edges
      .map((e) => ({ s: byId.get(e.source), t: byId.get(e.target), kind: e.kind }))
      .filter((l): l is { s: number; t: number; kind: ThreatEdge["kind"] } =>
        l.s !== undefined && l.t !== undefined
      );

    // O(n^2) per iteration, so trade iterations against node count to keep the
    // whole layout inside a few milliseconds even on a large window.
    const iterations = Math.max(120, Math.min(400, Math.round((400 * 60) / Math.max(1, n))));
    let alpha = 1;

    for (let step = 0; step < iterations; step++) {
      alpha *= 0.985;
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
          const rep = (900 * alpha) / d2;
          const fx = (dx / d) * rep;
          const fy = (dy / d) * rep;
          a.vx -= fx;
          a.vy -= fy;
          b.vx += fx;
          b.vy += fy;
        }
      }
      for (const l of links) {
        const a = pts[l.s];
        const b = pts[l.t];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) || 1;
        // Exact-asset links pull tighter: a proven cluster should read as tighter
        // on screen than an inferred one, reinforcing the dash/solid distinction.
        const rest = l.kind === "asset_exact" ? 46 : 74;
        const k = (d - rest) * 0.035 * alpha;
        const fx = (dx / d) * k;
        const fy = (dy / d) * k;
        a.vx += fx;
        a.vy += fy;
        b.vx -= fx;
        b.vy -= fy;
      }
      for (const p of pts) {
        p.vx -= p.x * 0.012 * alpha;
        p.vy -= p.y * 0.012 * alpha;
        p.vx *= 0.82;
        p.vy *= 0.82;
        p.x += p.vx;
        p.y += p.vy;
      }
    }
    return pts;
  }, [graph.nodes, graph.edges, byId]);

  // Fit the settled cloud into the viewport rather than assuming a scale.
  const view = useMemo(() => {
    if (!nodes.length) return { tx: width / 2, ty: height / 2, k: 1 };
    const pad = 34;
    const xs = nodes.map((n) => n.x);
    const ys = nodes.map((n) => n.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    // Cap the zoom so a two-node graph does not blow up to fill the panel, but
    // leave enough headroom that a settled cloud actually uses the canvas.
    const k = Math.min(
      (width - pad * 2) / Math.max(1, maxX - minX),
      (height - pad * 2) / Math.max(1, maxY - minY),
      2.4
    );
    return { tx: width / 2 - ((minX + maxX) / 2) * k, ty: height / 2 - ((minY + maxY) / 2) * k, k };
  }, [nodes, width, height]);

  const pos = useCallback(
    (n: Sim) => ({ x: n.x * view.k + view.tx, y: n.y * view.k + view.ty }),
    [view]
  );

  const posById = useMemo(() => {
    const m = new Map<string, { x: number; y: number }>();
    nodes.forEach((n) => m.set(n.id, pos(n)));
    return m;
  }, [nodes, pos]);

  const dim = (id: string) =>
    selected != null && selected !== id && !graph.edges.some(
      (e) =>
        (e.source === selected && e.target === id) || (e.target === selected && e.source === id)
    );

  return (
    <div ref={wrapRef} className="relative">
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Fraud-ring graph: ${graph.nodes.length} submissions, ${graph.edges.length} links`}
        className="block touch-none select-none"
        onMouseLeave={() => setHover(null)}
      >
        <g>
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
                stroke={exact ? REJECT : AXIS_STROKE}
                strokeWidth={exact ? 1.6 : 1}
                // Dashed = a similarity claim. Solid = a byte-identical file.
                strokeDasharray={exact ? undefined : "3 3"}
                strokeOpacity={faded ? 0.12 : exact ? 0.85 : 0.5}
              />
            );
          })}
        </g>
        <g>
          {nodes.map((n) => {
            const p = pos(n);
            const r = radiusOf(n.score) * Math.min(1.25, view.k);
            const faded = dim(n.id);
            return (
              <circle
                key={n.id}
                cx={p.x}
                cy={p.y}
                r={r}
                fill={VERDICT_FILL[n.verdict]}
                fillOpacity={faded ? 0.15 : 0.9}
                // A 2px surface ring keeps overlapping nodes legible.
                stroke={selected === n.id ? "#f1f5f9" : INK[900]}
                strokeWidth={2}
                className="cursor-pointer"
                onMouseEnter={() => setHover({ node: n, x: p.x, y: p.y })}
                onClick={() => onSelect?.(selected === n.id ? null : n.id)}
              />
            );
          })}
        </g>
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
            {/* Verdict in words, never colour alone. */}
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
            {hover.node.ring !== null && <div>ring #{hover.node.ring}</div>}
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
          <line x1="0" y1="3" x2="18" y2="3" stroke={REJECT} strokeWidth="1.6" />
        </svg>
        byte-identical file — a fact
      </span>
      <span className="flex items-center gap-1.5">
        <svg width="18" height="6" aria-hidden>
          <line x1="0" y1="3" x2="18" y2="3" stroke="#4b5565" strokeWidth="1" strokeDasharray="3 3" />
        </svg>
        similarity — an inference
      </span>
      <span className={clsx("ml-auto num")}>
        edges at cos ≥ {threshold.same_person.toFixed(4)}
      </span>
    </div>
  );
}

"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { api, fmtPct, type ThreatGraph, type Verdict } from "@/lib/api";
import { RingGraph, RingGraphLegend } from "@/components/RingGraph";
import { Empty, ErrorBox, PageHeader, Section, StatTile, VerdictBadge } from "@/components/ui";

const WINDOWS = [
  { hours: 24, label: "24h" },
  { hours: 168, label: "7d" },
  { hours: 720, label: "30d" },
  { hours: 8760, label: "all" },
];

const REFRESH_MS = 6000;

interface Row {
  submission_id: string;
  merchant_id: string;
  created_at: string;
  verdict: Verdict | null;
  score: number | null;
  attack_pattern: string | null;
  generator_guess: string | null;
  claimed_name: string | null;
  top_reasons?: string[];
}

const ago = (iso: string) => {
  const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(0)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(0)}h`;
  return `${(s / 86400).toFixed(0)}d`;
};

export default function ThreatIntelPage() {
  const [hours, setHours] = useState(168);
  const [live, setLive] = useState(false);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [graph, setGraph] = useState<ThreatGraph | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const poll = useRef<any>(null);

  const load = useCallback(async () => {
    try {
      const feed = await api.submissions({ limit: 300 });
      setRows(feed.items ?? []);
      setError(null);
    } catch (e: any) {
      setError(e.message ?? String(e));
    }
    // The graph is a separate endpoint and a separate failure: the rest of the
    // page stays useful when that call fails on its own.
    try {
      setGraph(await api.threatGraph(hours));
      setGraphError(null);
    } catch (e: any) {
      setGraph(null);
      setGraphError(e.message ?? String(e));
    }
  }, [hours]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!live) return;
    poll.current = setInterval(load, REFRESH_MS);
    return () => clearInterval(poll.current);
  }, [live, load]);

  const windowed = useMemo(() => {
    if (!rows) return [];
    const cutoff = Date.now() - hours * 3600 * 1000;
    return rows.filter((r) => new Date(r.created_at).getTime() >= cutoff);
  }, [rows, hours]);

  const flagged = useMemo(
    () => windowed.filter((r) => r.verdict === "REVIEW" || r.verdict === "REJECT"),
    [windowed]
  );

  /** Generator families over the window. One measure across nominal categories,
   *  so it is a bar chart in a single colour — the category is the axis, and
   *  spending hue on it would encode nothing the axis does not already say. */
  const generators = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of windowed) {
      if (!r.generator_guess) continue;
      counts.set(r.generator_guess, (counts.get(r.generator_guess) ?? 0) + 1);
    }
    const out = [...counts.entries()].map(([label, n]) => ({ label, n })).sort((a, b) => b.n - a.n);
    const max = Math.max(1, ...out.map((d) => d.n));
    return { rows: out, max };
  }, [windowed]);

  const patterns = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of flagged) {
      const k = r.attack_pattern && r.attack_pattern !== "clean" ? r.attack_pattern : "unclassified";
      counts.set(k, (counts.get(k) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6);
  }, [flagged]);

  const merchants = useMemo(() => new Set(windowed.map((r) => r.merchant_id)).size, [windowed]);

  const provenRings = graph?.rings.filter((r) => r.has_exact_asset_reuse).length ?? 0;

  const feed = useMemo(() => {
    const list = selected
      ? windowed.filter(
          (r) =>
            r.submission_id === selected ||
            (graph?.edges ?? []).some(
              (e) =>
                (e.source === selected && e.target === r.submission_id) ||
                (e.target === selected && e.source === r.submission_id)
            )
        )
      : windowed;
    return list.slice(0, 60);
  }, [windowed, selected, graph]);

  return (
    <div className="space-y-14">
      <PageHeader
        title="Threat Intelligence"
        actions={
          <>
            <div className="segment" role="group" aria-label="Time window">
              {WINDOWS.map((w) => (
                <button key={w.hours} data-active={hours === w.hours} onClick={() => setHours(w.hours)}>
                  {w.label}
                </button>
              ))}
            </div>
            <button
              onClick={() => setLive((v) => !v)}
              className={clsx("btn", live && "border-pass/40 bg-pass/10 text-pass hover:text-pass")}
              aria-pressed={live}
            >
              <span
                className={clsx("h-1.5 w-1.5 rounded-full", live ? "bg-pass animate-sweep" : "bg-slate-600")}
                aria-hidden
              />
              {live ? "Live" : "Paused"}
            </button>
          </>
        }
      >
        One fake merchant is an incident; forty sharing a face or a file is a ring, and only the linking makes
        that visible. Everything here is derived from what the detectors already cached — no extra scoring pass.
      </PageHeader>

      {error && <ErrorBox error={error} />}

      <div className="grid grid-cols-[repeat(auto-fit,minmax(178px,1fr))] gap-x-8 gap-y-8">
        <StatTile
          label="Flagged in window"
          value={flagged.length}
          sub={`of ${windowed.length} submissions · ${fmtPct(windowed.length ? flagged.length / windowed.length : 0, 0)}`}
          tone={flagged.length ? "review" : "default"}
        />
        <StatTile
          label="Linked clusters"
          value={graph ? graph.rings.length : "—"}
          sub={graph ? `${provenRings} with byte-identical reuse` : "graph unavailable"}
          tone={provenRings > 0 ? "reject" : "default"}
        />
        <StatTile label="Merchants seen" value={merchants} sub="distinct in this window" />
        <StatTile
          label="Top generator"
          value={generators.rows[0] ? generators.rows[0].n : "—"}
          sub={generators.rows[0]?.label ?? "no generator attributed yet"}
        />
      </div>

      <div className="grid gap-x-12 gap-y-14 wide:grid-cols-[minmax(0,1fr)_320px]">
        {/* ---------------- fraud rings ---------------- */}
        <Section
          title="Fraud rings"
          hint="Submissions linked by a shared face or a shared file. Click a node to filter the feed to it and everything it touches."
        >
          {graph && graph.nodes.length > 0 ? (
            <div className="space-y-3">
              <RingGraph graph={graph} onSelect={setSelected} selected={selected} />
              <div className="border-t border-edge pt-2.5">
                <RingGraphLegend threshold={graph.threshold} />
                <p className="mt-2 text-2xs leading-relaxed text-slate-600">
                  {graph.threshold.fitted_on}
                </p>
              </div>
            </div>
          ) : graph ? (
            <div>
              <Empty
                title="No links in this window"
                hint="Every submission here is isolated — nothing shares a face or a file with anything else. Widen the window, or run the Gauntlet to load fixtures that do."
              />
            </div>
          ) : (
            <div>
              <Empty
                title="Ring graph could not be loaded"
                hint={
                  <>
                    This panel reads <code className="num text-accent">GET /threat/graph</code>, which did not
                    answer. It is left empty on purpose: a fabricated ring in a fraud tool is worse than a
                    blank panel, so nothing is drawn unless the real edges came back.
                    {graphError && (
                      <span className="num mt-2 block text-slate-600">{graphError.slice(0, 120)}</span>
                    )}
                  </>
                }
              />
            </div>
          )}
        </Section>

        {/* ---------------- side panels ---------------- */}
        <div className="space-y-10">
          <Section
            title="Generator fingerprint"
            hint="Which synthesis family the spectral fingerprint attributed each flagged image to, over the window."
          >
            {generators.rows.length ? (
              <ul className="space-y-2.5">
                {generators.rows.map((g) => (
                  <li key={g.label}>
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="truncate text-xs text-slate-300" title={g.label}>
                        {g.label}
                      </span>
                      <span className="num shrink-0 text-xs text-slate-400">{g.n}</span>
                    </div>
                    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-ink-800">
                      <div
                        className="h-full rounded-full bg-accent transition-all duration-500"
                        style={{ width: `${(g.n / generators.max) * 100}%` }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-slate-600">
                No generator attributed in this window. The fingerprint only reports above 0.5 confidence.
              </p>
            )}
          </Section>

          <Section title="Attack patterns">
            {patterns.length ? (
              <ul className="space-y-1.5">
                {patterns.map(([k, n]) => (
                  <li key={k} className="flex items-baseline justify-between gap-2 text-xs">
                    <span className="truncate text-slate-400">{k.replace(/_/g, " ")}</span>
                    <span className="num text-slate-300">{n}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-xs text-slate-600">Nothing flagged in this window.</p>
            )}
          </Section>
        </div>
      </div>

      {/* ---------------- live feed ---------------- */}
      <Section
        title={
          <>
            Feed
            {selected && (
              <span className="ml-2 text-2xs font-normal text-slate-500">
                filtered to one cluster ·{" "}
                <button onClick={() => setSelected(null)} className="text-accent hover:underline">
                  clear
                </button>
              </span>
            )}
          </>
        }
        right={
          <span className="num">
            {feed.length} of {windowed.length} shown
          </span>
        }
      >
        {feed.length ? (
          <div className="scroll-x">
            <table className="w-full min-w-[760px] text-left text-sm">
              <thead className="border-b border-edge text-2xs uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-2 pr-4 font-medium">When</th>
                  <th className="py-2 pr-4 font-medium">Applicant</th>
                  <th className="py-2 pr-4 font-medium">Merchant</th>
                  <th className="py-2 pr-4 font-medium">Verdict</th>
                  <th className="py-2 pr-4 text-right font-medium">Risk</th>
                  <th className="py-2 pr-4 font-medium">Pattern</th>
                  <th className="py-2 pr-4 font-medium">Generator</th>
                </tr>
              </thead>
              <tbody>
                {feed.map((r) => (
                  <tr
                    key={r.submission_id}
                    onClick={() => setSelected(selected === r.submission_id ? null : r.submission_id)}
                    className={clsx(
                      "cursor-pointer border-b border-edge/60 transition-colors last:border-0 hover:bg-ink-900",
                      selected === r.submission_id && "bg-accent/[0.06]"
                    )}
                  >
                    <td className="num py-2 pr-4 text-2xs text-slate-500">{ago(r.created_at)}</td>
                    <td className="py-2 pr-4 text-xs text-slate-300">{r.claimed_name ?? "—"}</td>
                    <td className="num py-2 pr-4 text-2xs text-slate-500">{r.merchant_id}</td>
                    <td className="py-2 pr-4">
                      {r.verdict ? <VerdictBadge verdict={r.verdict} size="sm" /> : <span className="text-slate-600">—</span>}
                    </td>
                    <td className="num py-2 pr-4 text-right text-xs text-slate-300">
                      {r.score !== null && r.score !== undefined ? r.score.toFixed(2) : "—"}
                    </td>
                    <td className="py-2 pr-4 text-2xs text-slate-500">
                      {r.attack_pattern && r.attack_pattern !== "clean"
                        ? r.attack_pattern.replace(/_/g, " ")
                        : "—"}
                    </td>
                    <td className="max-w-[220px] truncate py-2 text-2xs text-slate-500">
                      {r.generator_guess ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-10 text-center text-xs text-slate-600">
            {rows === null ? "Loading…" : "Nothing in this window."}
          </div>
        )}
      </Section>
    </div>
  );
}

"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import clsx from "clsx";
import { api, fmtInr, fmtPct, type Ablation } from "@/lib/api";
import { Empty, ErrorBox, Spinner, StatTile } from "@/components/ui";

const SERIES_COLORS = ["#5b8cff", "#2dd4a7", "#f5b53d", "#c084fc", "#fb7185"];
const AXIS = { stroke: "#3a4258", fontSize: 11 };
const GRID = "#1e2432";

const tooltipStyle = {
  contentStyle: { background: "#0b0e16", border: "1px solid #232a3a", borderRadius: 10, fontSize: 12 },
  labelStyle: { color: "#94a3b8" },
};

function Section({ title, hint, children }: { title: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card-pad">
      <div className="flex flex-wrap items-baseline justify-between gap-3">
        <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
        {hint && <p className="max-w-2xl text-[11px] leading-relaxed text-slate-500">{hint}</p>}
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

export default function MetricsPage() {
  const [data, setData] = useState<any>(null);
  const [cost, setCost] = useState<any>(null);
  const [ablation, setAblation] = useState<Ablation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [costError, setCostError] = useState<string | null>(null);

  const [fraudLoss, setFraudLoss] = useState(85000);
  const [ltv, setLtv] = useState(42000);
  const [abandon, setAbandon] = useState(0.35);
  const [baseRate, setBaseRate] = useState(0.03);

  useEffect(() => { api.metrics().then(setData).catch((e) => setError(e.message)); }, []);
  useEffect(() => { api.ablation().then(setAblation).catch(() => setAblation(null)); }, []);

  useEffect(() => {
    const t = setTimeout(() => {
      api.costCurve({ avg_fraud_loss_inr: fraudLoss, merchant_ltv_inr: ltv, abandon_prob: abandon, fraud_base_rate: baseRate })
        .then((c) => { setCost(c); setCostError(null); })
        .catch((e) => setCostError(e.message));
    }, 220);
    return () => clearTimeout(t);
  }, [fraudLoss, ltv, abandon, baseRate]);

  const ev = data?.evaluation;

  const rocData = useMemo(() => {
    if (!ev) return [];
    const grid = Array.from({ length: 51 }, (_, i) => i / 50);
    const series: Record<string, number>[] = grid.map((fpr) => ({ fpr }));
    const attach = (key: string, roc: any[]) => {
      if (!roc?.length) return;
      grid.forEach((fpr, i) => {
        let best = 0;
        for (const p of roc) if (p.fpr <= fpr) best = Math.max(best, p.tpr);
        series[i][key] = best;
      });
    };
    attach("fusion", ev.fusion?.roc);
    Object.entries(ev.per_detector ?? {}).forEach(([k, d]: any) => attach(k, d.roc));
    return series;
  }, [ev]);

  const attackData = useMemo(() => {
    if (!ev?.per_attack_type) return [];
    return Object.entries(ev.per_attack_type).map(([k, v]: any) => ({
      attack: k.replace(/_/g, " "),
      recall: v.caught_at_review,
      auc: v.auc_vs_genuine ?? 0,
      n: v.n,
    })).sort((a, b) => a.recall - b.recall);
  }, [ev]);

  const latencyData = useMemo(() => {
    const h = data?.live?.latency_ms?.histogram;
    if (!h) return [];
    return h.counts.map((c: number, i: number) => ({ bucket: `${Math.round(h.edges[i])}`, count: c }));
  }, [data]);

  if (error) return <ErrorBox error={error} />;
  if (!data) return <div className="py-24"><Spinner label="Loading metrics…" /></div>;

  if (!ev) {
    return (
      <Empty
        title="No held-out evaluation report yet"
        hint={<>Build the corpus and run the eval pipeline:{" "}
          <code className="font-mono text-accent">make pipeline</code>{" "}
          (build_dataset → benchmark → calibrate → score_corpus → train_fusion → evaluate).</>}
      />
    );
  }

  const cm = ev.fusion?.at_reject_threshold ?? {};
  const cmReview = ev.fusion?.at_review_threshold ?? {};

  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-xl font-semibold tracking-tight text-slate-100">Metrics</h1>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-slate-400">
          Everything below is computed on an <strong className="text-slate-200">identity-disjoint held-out split</strong> —
          no face that trained the fusion layer appears in these numbers. The uncomfortable numbers are here too.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        <StatTile label="Fusion ROC-AUC" value={ev.fusion?.roc_auc?.toFixed(3) ?? "—"}
          sub={`${ev.n_packets} held-out packets · ${ev.fusion_model}`} tone="pass" />
        <StatTile label="Recall @ reject" value={fmtPct(cm.recall, 0)}
          sub={`threshold ${cm.threshold} · precision ${fmtPct(cm.precision, 0)}`} />
        <StatTile label="False accept rate" value={fmtPct(cm.false_accept_rate, 1)}
          sub="fraud that would be approved" tone={cm.false_accept_rate <= 0.15 ? "pass" : "reject"} />
        <StatTile label="False reject rate" value={fmtPct(cm.false_reject_rate, 1)}
          sub="genuine merchants blocked" tone={cm.false_reject_rate <= 0.1 ? "pass" : "review"} />
        <StatTile label="Latency p50 / p99"
          value={`${data.live?.latency_ms?.p50?.toFixed(0) ?? "—"} / ${data.live?.latency_ms?.p99?.toFixed(0) ?? "—"}`}
          sub="ms, live API traffic on this instance" />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="ROC — fusion vs each detector"
          hint="The fusion curve should dominate. Where a single detector beats it, the fusion weights are wrong.">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rocData} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={AXIS} tickFormatter={(v) => v.toFixed(1)}
                label={{ value: "false positive rate", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 10 }} />
              <YAxis type="number" domain={[0, 1]} tick={AXIS} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => (typeof v === "number" ? v.toFixed(3) : v)} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="fusion" stroke="#5b8cff" strokeWidth={2.5} dot={false} name="fusion" />
              {Object.keys(ev.per_detector ?? {}).map((k, i) => (
                <Line key={k} type="monotone" dataKey={k} stroke={SERIES_COLORS[(i + 1) % SERIES_COLORS.length]}
                  strokeWidth={1.3} dot={false} strokeOpacity={0.75} name={k.replace(/_/g, " ")} />
              ))}
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#3a4258" strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Per-detector performance"
          hint="“On target attacks” grades each detector only against the fraud it is designed to catch — the all-rows AUC is diluted by attacks it cannot see.">
          <div className="scroll-x">
            <table className="w-full min-w-[520px] text-left text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="pb-2 font-medium">Detector</th>
                  <th className="pb-2 text-right font-medium">AUC (all)</th>
                  <th className="pb-2 text-right font-medium">AUC (on target)</th>
                  <th className="pb-2 text-right font-medium">Genuine</th>
                  <th className="pb-2 text-right font-medium">Fraud</th>
                  <th className="pb-2 text-right font-medium">Coverage</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {Object.entries(ev.per_detector).map(([k, d]: any) => (
                  <tr key={k} className="border-t border-edge/60">
                    <td className="py-2 pr-3 font-sans text-slate-300">{d.label}</td>
                    <td className={clsx("py-2 text-right tabular-nums",
                      d.auc_all_rows >= 0.7 ? "text-pass" : d.auc_all_rows >= 0.55 ? "text-review" : "text-reject")}>
                      {d.auc_all_rows?.toFixed(3) ?? "—"}
                    </td>
                    <td className={clsx("py-2 text-right tabular-nums",
                      d.auc_on_target_attacks >= 0.7 ? "text-pass" : d.auc_on_target_attacks >= 0.55 ? "text-review" : "text-slate-500")}>
                      {d.auc_on_target_attacks?.toFixed(3) ?? "—"}
                    </td>
                    <td className="py-2 text-right tabular-nums text-slate-400">{d.mean_genuine?.toFixed(2) ?? "—"}</td>
                    <td className="py-2 text-right tabular-nums text-slate-400">{d.mean_fraud?.toFixed(2) ?? "—"}</td>
                    <td className="py-2 text-right tabular-nums text-slate-500">{fmtPct(d.coverage, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.fusion_weights && Object.keys(data.fusion_weights).length > 0 && (
            <div className="mt-4 border-t border-edge pt-3">
              <div className="label">Fusion weights (positive raises risk)</div>
              <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px] sm:grid-cols-3">
                {Object.entries(data.fusion_weights)
                  .sort((a: any, b: any) => Math.abs(b[1]) - Math.abs(a[1]))
                  .slice(0, 9)
                  .map(([k, v]: any) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span className="truncate text-slate-500">{k.replace(/_score|_conf/, (m: string) => m === "_conf" ? "·c" : "")}</span>
                      <span className={v >= 0 ? "text-reject" : "text-pass"}>{v >= 0 ? "+" : ""}{Number(v).toFixed(2)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </Section>
      </div>

      {ablation && (
        <Section
          title="What the headline number is made of"
          hint="Fusion refit with one detector muted at a time, scored on the same held-out split. A single AUC cannot tell you whether five detectors earned it or one did."
        >
          <div className="scroll-x">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="pb-2 font-medium">Detector muted</th>
                  <th className="pb-2 text-right font-medium">Held-out AUC</th>
                  <th className="pb-2 text-right font-medium">Change</th>
                  <th className="pb-2 font-medium pl-6">Share of above-chance AUC</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                <tr className="border-t border-edge/60">
                  <td className="py-2 pr-3 font-sans text-slate-400">— none (full model)</td>
                  <td className="py-2 text-right tabular-nums text-slate-200">{ablation.full_model.roc_auc.toFixed(3)}</td>
                  <td className="py-2 text-right text-slate-600">—</td>
                  <td className="pl-6" />
                </tr>
                {ablation.ranked.map((r) => {
                  const share = r.share_of_headline_auc_above_chance;
                  const helps = r.auc_drop > 0;
                  return (
                    <tr key={r.detector} className="border-t border-edge/60">
                      <td className="py-2 pr-3 font-sans text-slate-300">{r.detector.replace(/_/g, " ")}</td>
                      <td className="py-2 text-right tabular-nums text-slate-300">{r.roc_auc.toFixed(3)}</td>
                      <td className={clsx("py-2 text-right tabular-nums", helps ? "text-reject" : "text-pass")}>
                        {r.auc_drop >= 0 ? "−" : "+"}{Math.abs(r.auc_drop).toFixed(3)}
                      </td>
                      <td className="py-2 pl-6">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-32 overflow-hidden rounded-full bg-ink-800">
                            <div
                              className={clsx("h-full rounded-full", helps ? "bg-reject" : "bg-pass")}
                              style={{ width: `${Math.min(100, Math.abs(share) * 100)}%` }}
                            />
                          </div>
                          <span className={clsx("tabular-nums text-[11px]", helps ? "text-slate-300" : "text-pass")}>
                            {share >= 0 ? "" : "−"}{Math.abs(share * 100).toFixed(1)}%
                          </span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <p className="mt-3 text-[11px] leading-relaxed text-slate-500">
            A <span className="text-pass">negative</span> share means muting that detector made the
            model <em>better</em> — it was contributing noise, not signal.
          </p>

          <div className="mt-4 border-t border-edge pt-3">
            <div className="label">Corpus leak audit</div>
            {ablation.corpus_clean ? (
              <p className="mt-2 text-[11px] leading-relaxed text-pass">
                No generator setting lands on one class only without a physical reason to. Every
                detector above is reading the packet rather than a label written into the file.
              </p>
            ) : (
              <>
                <p className="mt-2 text-[11px] leading-relaxed text-reject">
                  {ablation.leaks.length} corpus value{ablation.leaks.length === 1 ? "" : "s"} appear
                  on one class only. A detector reading one of these is reading the answer key, and
                  every number above it is inflated by an unknown amount.
                </p>
                <div className="mt-2 grid gap-1 font-mono text-[11px] sm:grid-cols-2">
                  {ablation.leaks.map((l) => (
                    <div key={`${l.field}-${l.value}`} className="flex justify-between gap-2 rounded border border-reject/30 bg-reject/8 px-2 py-1">
                      <span className="truncate text-slate-400">{l.field}={l.value}</span>
                      <span className="text-reject">{l.n} packets, all {l.class}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
            {ablation.one_sided_but_expected?.length > 0 && (
              <div className="mt-3">
                <p className="text-[11px] leading-relaxed text-slate-500">
                  One-sided by construction, and allowed to be:
                </p>
                <div className="mt-1 grid gap-1 font-mono text-[11px] sm:grid-cols-2">
                  {ablation.one_sided_but_expected.map((l) => (
                    <div key={`${l.field}-${l.value}`} className="rounded border border-edge bg-ink-850 px-2 py-1">
                      <div className="flex justify-between gap-2">
                        <span className="truncate text-slate-400">{l.field}={l.value}</span>
                        <span className="text-slate-500">{l.n} packets</span>
                      </div>
                      {l.justification && (
                        <div className="mt-0.5 font-sans text-[10px] leading-snug text-slate-600">{l.justification}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Section>
      )}

      <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
        <Section title="Recall by attack type"
          hint="Sorted worst-first on purpose. A single headline AUC hides which attack we are actually bad at.">
          <ResponsiveContainer width="100%" height={Math.max(220, attackData.length * 34)}>
            <BarChart data={attackData} layout="vertical" margin={{ top: 4, right: 40, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tick={AXIS} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="attack" tick={{ ...AXIS, fontSize: 10 }} width={150} />
              <Tooltip {...tooltipStyle}
                formatter={(v: any, n: any) => [n === "recall" ? fmtPct(v, 0) : Number(v).toFixed(3), n]} />
              <Bar dataKey="recall" name="caught at review threshold" radius={[0, 4, 4, 0]} barSize={16}>
                {attackData.map((d, i) => (
                  <Cell key={i} fill={d.recall >= 0.85 ? "#2dd4a7" : d.recall >= 0.6 ? "#f5b53d" : "#fb5e6d"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Confusion matrix" hint={`at the reject threshold (${cm.threshold})`}>
          <div className="grid grid-cols-2 gap-2 text-center">
            {[
              { k: "tp", label: "True positive", v: cm.tp, tone: "text-pass", note: "fraud caught" },
              { k: "fn", label: "False negative", v: cm.fn, tone: "text-reject", note: "fraud missed" },
              { k: "fp", label: "False positive", v: cm.fp, tone: "text-review", note: "genuine blocked" },
              { k: "tn", label: "True negative", v: cm.tn, tone: "text-slate-300", note: "genuine passed" },
            ].map((c) => (
              <div key={c.k} className="rounded-lg border border-edge bg-ink-850 p-3">
                <div className={clsx("font-mono text-2xl font-semibold tabular-nums", c.tone)}>{c.v ?? "—"}</div>
                <div className="mt-0.5 text-[10px] uppercase tracking-wider text-slate-500">{c.label}</div>
                <div className="text-[10px] text-slate-600">{c.note}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 space-y-1.5 border-t border-edge pt-3 font-mono text-[11px] text-slate-400">
            <div className="flex justify-between"><span className="text-slate-500">precision</span><span>{fmtPct(cm.precision, 1)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">recall</span><span>{fmtPct(cm.recall, 1)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">F1</span><span>{cm.f1?.toFixed(3)}</span></div>
            <div className="flex justify-between border-t border-edge pt-1.5"><span className="text-slate-500">recall @ review ({cmReview.threshold})</span><span>{fmtPct(cmReview.recall, 1)}</span></div>
            <div className="flex justify-between"><span className="text-slate-500">FRR @ review</span><span>{fmtPct(cmReview.false_reject_rate, 1)}</span></div>
          </div>
        </Section>
      </div>

      {/* ---------------- cost of friction ---------------- */}
      <Section
        title="Cost of friction — fraud prevented vs merchants lost"
        hint="Four of these inputs are business assumptions, not measurements. They are sliders precisely so you can move them: a single ₹ figure would be unfalsifiable."
      >
        {costError && <div className="mb-3 text-xs text-review">{costError}</div>}
        <div className="grid gap-6 lg:grid-cols-[1fr_300px]">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={cost?.curve ?? []} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="threshold" tick={AXIS} tickFormatter={(v) => v.toFixed(2)}
                label={{ value: "reject threshold", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={AXIS} tickFormatter={(v) => fmtInr(v)} width={62} />
              <Tooltip {...tooltipStyle} formatter={(v: any, n: any) => [fmtInr(Number(v)), n]}
                labelFormatter={(l) => `threshold ${Number(l).toFixed(2)}`} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Line type="monotone" dataKey="fraud_prevented_inr" name="fraud prevented" stroke="#2dd4a7" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="friction_cost_inr" name="lost to false rejects" stroke="#fb5e6d" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="net_benefit_inr" name="net benefit" stroke="#5b8cff" strokeWidth={2.5} dot={false} />
              {cost?.optimal && (
                <ReferenceLine x={cost.optimal.threshold} stroke="#5b8cff" strokeDasharray="4 4"
                  label={{ value: `optimum ${cost.optimal.threshold}`, fill: "#5b8cff", fontSize: 10, position: "top" }} />
              )}
            </LineChart>
          </ResponsiveContainer>

          <div className="space-y-4">
            {[
              { label: "Avg fraud loss per fake merchant", value: fraudLoss, set: setFraudLoss, min: 5000, max: 500000, step: 5000, fmt: fmtInr },
              { label: "Legitimate merchant LTV", value: ltv, set: setLtv, min: 5000, max: 300000, step: 1000, fmt: fmtInr },
              { label: "Abandon probability after false reject", value: abandon, set: setAbandon, min: 0, max: 1, step: 0.05, fmt: (v: number) => fmtPct(v, 0) },
              { label: "Fraud base rate in traffic", value: baseRate, set: setBaseRate, min: 0.001, max: 0.2, step: 0.001, fmt: (v: number) => fmtPct(v, 1) },
            ].map((s) => (
              <div key={s.label}>
                <div className="flex items-baseline justify-between">
                  <label className="text-[11px] text-slate-400">{s.label}</label>
                  <span className="font-mono text-xs text-slate-200">{s.fmt(s.value)}</span>
                </div>
                <input type="range" min={s.min} max={s.max} step={s.step} value={s.value}
                  onChange={(e) => s.set(Number(e.target.value))}
                  className="mt-1.5 w-full accent-[#5b8cff]" />
              </div>
            ))}

            {cost?.optimal && (
              <div className="rounded-lg border border-accent/40 bg-accent/8 p-3">
                <div className="label text-accent">Optimal threshold</div>
                <div className="mt-1 font-mono text-2xl font-semibold text-accent">{cost.optimal.threshold}</div>
                <div className="mt-1.5 space-y-0.5 text-[11px] leading-relaxed text-slate-400">
                  <div>net {fmtInr(cost.optimal.net_benefit_inr)} per {cost.assumptions.per_onboardings.toLocaleString()} onboardings</div>
                  <div>FAR {fmtPct(cost.optimal.false_accept_rate, 1)} · FRR {fmtPct(cost.optimal.false_reject_rate, 1)}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </Section>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section title="Bias audit — accuracy by skin-tone proxy"
          hint="Deepfake detectors are known to degrade on darker skin. Measuring it is the minimum bar.">
          <div className="scroll-x">
            <table className="w-full min-w-[420px] text-left text-xs">
              <thead className="text-[10px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="pb-2 font-medium">ITA° bucket</th>
                  <th className="pb-2 text-right font-medium">n</th>
                  <th className="pb-2 text-right font-medium">AUC</th>
                  <th className="pb-2 text-right font-medium">FRR</th>
                  <th className="pb-2 text-right font-medium">FAR</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {Object.entries(ev.bias_audit?.buckets ?? {}).map(([k, b]: any) => (
                  <tr key={k} className="border-t border-edge/60">
                    <td className="py-2 font-sans text-slate-300">{k.replace(/_/g, " ")}</td>
                    <td className="py-2 text-right tabular-nums text-slate-400">{b.n}</td>
                    <td className="py-2 text-right tabular-nums text-slate-300">{b.auc?.toFixed(3) ?? "—"}</td>
                    <td className="py-2 text-right tabular-nums text-slate-400">{b.false_reject_rate !== undefined ? fmtPct(b.false_reject_rate, 1) : "—"}</td>
                    <td className="py-2 text-right tabular-nums text-slate-400">{b.false_accept_rate !== undefined ? fmtPct(b.false_accept_rate, 1) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 border-t border-edge pt-3 text-[11px] leading-relaxed text-slate-500">
            {ev.bias_audit?.caveat}
          </p>
        </Section>

        <Section title="Latency" hint="Live API traffic on this instance; detectors run concurrently in the request path.">
          {latencyData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={latencyData} margin={{ top: 4, right: 8, bottom: 4, left: -20 }}>
                <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
                <XAxis dataKey="bucket" tick={{ ...AXIS, fontSize: 10 }}
                  label={{ value: "ms", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 10 }} />
                <YAxis tick={AXIS} allowDecimals={false} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="count" fill="#5b8cff" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">No live traffic yet — run the gauntlet.</div>
          )}
          {ev.per_capture_mode && Object.keys(ev.per_capture_mode).length > 0 && (
            <div className="mt-4 border-t border-edge pt-3">
              <div className="label">Tamper detection by upload type</div>
              <div className="mt-2 space-y-1 font-mono text-[11px]">
                {Object.entries(ev.per_capture_mode).map(([mode, m]: any) => (
                  <div key={mode} className="flex justify-between gap-3">
                    <span className="text-slate-500">{mode === "scan" ? "digital scan / direct upload" : "phone photo of card"} (n={m.n})</span>
                    <span className="text-slate-300">ID-forensics AUC {m.id_forensics_auc?.toFixed(3) ?? "—"}</span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-500">
                Compression-based tamper analysis reads an image&apos;s edit history. Photographing a card re-encodes
                the whole frame and largely erases it, so the two populations are reported separately rather than averaged.
              </p>
            </div>
          )}
        </Section>
      </div>

      <Section title="Known limitations" hint="Stated up front, because a reviewer will find them anyway.">
        <ul className="space-y-2">
          {(ev.limitations ?? []).map((l: string, i: number) => (
            <li key={i} className="flex gap-3 text-xs leading-relaxed text-slate-400">
              <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-review" />
              {l}
            </li>
          ))}
        </ul>
      </Section>
    </div>
  );
}

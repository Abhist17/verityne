"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import clsx from "clsx";
import { api, fmtInr, fmtPct, type Ablation, type BehavioralReport } from "@/lib/api";
import { Empty, ErrorBox, PageHeader, Spinner, StatTile } from "@/components/ui";

// Charts read the same palette the rest of the app does. The per-detector
// series are deliberately dimmer and thinner than fusion: the point of the ROC
// panel is whether fusion dominates them, so fusion has to be the figure and
// they have to be the ground.
import { ACCENT, AXIS_STROKE, GRID_STROKE, INK, EDGE_STRONG, PASS, REJECT, REVIEW, SERIES } from "@/lib/palette";

const SERIES_COLORS = SERIES;
const AXIS = { stroke: AXIS_STROKE, fontSize: 10 };
const GRID = GRID_STROKE;

const tooltipStyle = {
  contentStyle: {
    background: INK[900],
    border: `1px solid ${EDGE_STRONG}`,
    borderRadius: 6,
    fontSize: 11,
    boxShadow: "0 8px 24px rgb(0 0 0 / 0.5)",
  },
  labelStyle: { color: "#94a3b8", marginBottom: 2 },
  itemStyle: { padding: "1px 0" },
};

function Section({ title, hint, children }: { title: string; hint?: React.ReactNode; children: React.ReactNode }) {
  return (
    <section className="card-pad">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h2 className="text-sm font-semibold text-slate-200">{title}</h2>
        {hint && <p className="max-w-[62ch] text-2xs leading-relaxed text-slate-500">{hint}</p>}
      </div>
      <div className="mt-3.5">{children}</div>
    </section>
  );
}

/**
 * Detector 6, measured.
 *
 * The layout puts the weakest number in the largest type. Every other panel on
 * this page leads with what works; this one leads with the strategy that beats
 * the model, because a reviewer who finds that on slide twelve stops believing
 * slides one through eleven.
 */
function BehavioralSection({ r }: { r: BehavioralReport }) {
  const shipped = r.ablation[r.shipped_feature_set];
  const strategies = Object.entries(shipped?.leave_one_strategy_out ?? {});
  const worstAuc = r.headline.worst_unseen_strategy_auc;

  return (
    <Section
      title="Detector 6 — keystroke rhythm, measured on real data"
      hint={
        <>
          Genuine sessions from {r.corpus.human_aalto_sessions.toLocaleString()} Aalto and{" "}
          {r.corpus.human_cmu_sessions.toLocaleString()} CMU typing sessions; automated sessions from{" "}
          {r.corpus.bot_sessions.toLocaleString()} runs of a real headless Chromium, captured by the
          collector this app ships. Neither half was written to look like what we expected.
        </>
      }
    >
      <div className="grid gap-4 wide:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]">
        <div className="space-y-3">
          <div className="grid grid-cols-3 gap-3">
            <StatTile label="Held-out AUC" value={r.headline.held_out_auc?.toFixed(3) ?? "—"} />
            <StatTile label="Recall" value={fmtPct(r.headline.held_out_recall, 0)} />
            <StatTile label="Human FPR" value={fmtPct(r.headline.human_false_positive_rate, 1)} />
          </div>
          <p className="text-2xs leading-relaxed text-slate-500">
            Subject-disjoint: no participant appears in both train and test. Keystroke dynamics
            identifies people, so a row-wise split would measure that instead. The threshold
            ({shipped?.threshold.toFixed(3)}) is read off held-out humans for a{" "}
            {fmtPct(r.human_fpr_budget, 0)} false-positive budget, not assumed at 0.5.
          </p>
          {r.sweep?.ran && (
            <p className="text-2xs leading-relaxed text-slate-500">
              {r.sweep.configurations} configurations searched, ranked by transfer to an unseen
              human population rather than by held-out AUC — every configuration reaches 1.000 there,
              so ranking on it would have picked one at random and called it tuned.
            </p>
          )}
        </div>

        <div>
          <div className="label mb-2">Against automation it has never seen</div>
          <div className="space-y-1">
            {strategies.map(([name, v]) => {
              const beaten = (v.auc ?? 1) < 0.65;
              return (
                <div key={name} className="flex items-center gap-3 border-t border-edge/60 py-1.5">
                  <span className="num min-w-0 flex-1 truncate text-xs text-slate-400">{name}</span>
                  <span className="num w-24 text-right text-xs text-slate-500">
                    recall {fmtPct(v.recall_at_threshold, 0)}
                  </span>
                  <span
                    className={clsx("num w-14 text-right text-xs", beaten ? "text-reject" : "text-slate-300")}
                  >
                    {v.auc?.toFixed(3) ?? "—"}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="mt-2 text-2xs leading-relaxed text-slate-500">
            Each row trains without that strategy and tests on it. This is the number that describes
            deployment: the kit in production next month is not in this corpus.
          </p>
        </div>
      </div>

      {worstAuc !== null && worstAuc < 0.65 && (
        <div className="mt-4 border-t border-edge pt-3.5">
          <div className="label text-reject">The attack that defeats it</div>
          <p className="mt-1.5 max-w-[78ch] text-xs leading-relaxed text-slate-400">
            <span className="num text-slate-200">{r.headline.worst_unseen_strategy}</span> replays a
            real person&apos;s dwell and flight timings through the devtools protocol, rollover
            included, and scores{" "}
            <span className="num text-reject">{worstAuc.toFixed(3)}</span> — chance. It is not a bug
            in the model: the rhythm genuinely is human, so no rhythm model can separate it. What
            still catches it is that a replayed recording is a <em>reused</em> one, which is a
            linkage problem rather than a timing one.
          </p>
        </div>
      )}
    </Section>
  );
}

export default function MetricsPage() {
  const [data, setData] = useState<any>(null);
  const [cost, setCost] = useState<any>(null);
  const [ablation, setAblation] = useState<Ablation | null>(null);
  const [behav, setBehav] = useState<BehavioralReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [costError, setCostError] = useState<string | null>(null);

  const [fraudLoss, setFraudLoss] = useState(85000);
  const [ltv, setLtv] = useState(42000);
  const [abandon, setAbandon] = useState(0.35);
  const [baseRate, setBaseRate] = useState(0.03);

  useEffect(() => { api.metrics().then(setData).catch((e) => setError(e.message)); }, []);
  // Absent until the corpus is built and the model fitted; the section simply
  // does not render rather than showing a measurement nobody made.
  useEffect(() => { api.behavioralMetrics().then(setBehav).catch(() => setBehav(null)); }, []);
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
      <PageHeader title="Metrics">
        Everything below is computed on an{" "}
        <strong className="font-medium text-slate-200">identity-disjoint held-out split</strong> — no face that
        trained the fusion layer appears in these numbers. The uncomfortable numbers are here too.
      </PageHeader>

      {/* auto-fit rather than a fixed column count: five tiles into a 2- or
          4-column grid leaves a hole, and the hole reads as a missing metric. */}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(178px,1fr))] gap-3">
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

      <div className="grid gap-5 wide:grid-cols-2">
        <Section title="ROC — fusion vs each detector"
          hint="The fusion curve should dominate. Where a single detector beats it, the fusion weights are wrong.">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={rocData} margin={{ top: 4, right: 8, bottom: 4, left: -18 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={AXIS} tickFormatter={(v) => v.toFixed(1)}
                label={{ value: "false positive rate", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 10 }} />
              <YAxis type="number" domain={[0, 1]} tick={AXIS} tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip {...tooltipStyle} formatter={(v: any) => (typeof v === "number" ? v.toFixed(3) : v)} />
              <Legend iconType="plainline" iconSize={8} wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
              <Line type="monotone" dataKey="fusion" stroke={ACCENT} strokeWidth={2.5} dot={false} name="fusion" />
              {Object.keys(ev.per_detector ?? {}).map((k, i) => (
                <Line key={k} type="monotone" dataKey={k} stroke={SERIES_COLORS[(i + 1) % SERIES_COLORS.length]}
                  strokeWidth={1.3} dot={false} strokeOpacity={0.75} name={k.replace(/_/g, " ")} />
              ))}
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#333c4c" strokeDasharray="4 4" />
            </LineChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Per-detector performance"
          hint="“On target attacks” grades each detector only against the fraud it is designed to catch — the all-rows AUC is diluted by attacks it cannot see.">
          <div className="scroll-x">
            <table className="w-full min-w-[440px] text-left text-xs">
              <thead className="text-2xs uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="pb-2 font-medium">Detector</th>
                  <th className="pb-2 text-right font-medium">AUC (all)</th>
                  <th className="pb-2 text-right font-medium">AUC (target)</th>
                  <th className="pb-2 text-right font-medium">Genuine</th>
                  <th className="pb-2 text-right font-medium">Fraud</th>
                  <th className="pb-2 text-right font-medium">Coverage</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {Object.entries(ev.per_detector).map(([k, d]: any) => (
                  <tr key={k} className="border-t border-edge/60">
                    <td className="py-2 pr-3 font-sans text-slate-300">{d.label}</td>
                    <td className={clsx("py-2 text-right ",
                      d.auc_all_rows >= 0.7 ? "text-pass" : d.auc_all_rows >= 0.55 ? "text-review" : "text-reject")}>
                      {d.auc_all_rows?.toFixed(3) ?? "—"}
                    </td>
                    <td className={clsx("py-2 text-right ",
                      d.auc_on_target_attacks >= 0.7 ? "text-pass" : d.auc_on_target_attacks >= 0.55 ? "text-review" : "text-slate-500")}>
                      {d.auc_on_target_attacks?.toFixed(3) ?? "—"}
                    </td>
                    <td className="py-2 text-right text-slate-400">{d.mean_genuine?.toFixed(2) ?? "—"}</td>
                    <td className="py-2 text-right text-slate-400">{d.mean_fraud?.toFixed(2) ?? "—"}</td>
                    <td className="py-2 text-right text-slate-500">{fmtPct(d.coverage, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.fusion_weights && Object.keys(data.fusion_weights).length > 0 && (
            <div className="mt-4 border-t border-edge pt-3">
              <div className="label">Fusion weights (positive raises risk)</div>
              <div className="num mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
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
              <thead className="text-2xs uppercase tracking-wider text-slate-500">
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
                  <td className="py-2 text-right text-slate-200">{ablation.full_model.roc_auc.toFixed(3)}</td>
                  <td className="py-2 text-right text-slate-600">—</td>
                  <td className="pl-6" />
                </tr>
                {ablation.ranked.map((r) => {
                  const share = r.share_of_headline_auc_above_chance;
                  const helps = r.auc_drop > 0;
                  return (
                    <tr key={r.detector} className="border-t border-edge/60">
                      <td className="py-2 pr-3 font-sans text-slate-300">{r.detector.replace(/_/g, " ")}</td>
                      <td className="py-2 text-right text-slate-300">{r.roc_auc.toFixed(3)}</td>
                      <td className={clsx("py-2 text-right", helps ? "text-slate-300" : "text-reject")}>
                        {r.auc_drop >= 0 ? "−" : "+"}{Math.abs(r.auc_drop).toFixed(3)}
                      </td>
                      <td className="py-2 pl-6">
                        <div className="flex items-center gap-2">
                          <div className="h-1.5 w-32 overflow-hidden rounded-full bg-ink-800">
                            <div
                              className={clsx("h-full rounded-full", helps ? "bg-accent" : "bg-reject")}
                              style={{ width: `${Math.min(100, Math.abs(share) * 100)}%` }}
                            />
                          </div>
                          <span className={clsx("text-xs", helps ? "text-slate-300" : "text-reject")}>
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

          <p className="mt-3 text-xs leading-relaxed text-slate-500">
            <span className="text-accent">Blue</span> is the share of the above-chance AUC that detector
            carries. <span className="text-reject">Red</span> is a negative share — muting it made the model{" "}
            <em>better</em>, so it was contributing noise rather than signal.
          </p>

          <div className="mt-4 border-t border-edge pt-3">
            <div className="label">Corpus leak audit</div>
            {ablation.corpus_clean ? (
              <p className="mt-2 text-xs leading-relaxed text-pass">
                No generator setting lands on one class only without a physical reason to. Every
                detector above is reading the packet rather than a label written into the file.
              </p>
            ) : (
              <>
                <p className="mt-2 text-xs leading-relaxed text-reject">
                  {ablation.leaks.length} corpus value{ablation.leaks.length === 1 ? "" : "s"} appear
                  on one class only. A detector reading one of these is reading the answer key, and
                  every number above it is inflated by an unknown amount.
                </p>
                <div className="mt-2 grid gap-1 font-mono text-xs sm:grid-cols-2">
                  {ablation.leaks.map((l) => (
                    <div key={`${l.field}-${l.value}`} className="flex justify-between gap-2 rounded border border-reject/30 bg-reject/[0.07] px-2 py-1">
                      <span className="truncate text-slate-400">{l.field}={l.value}</span>
                      <span className="text-reject">{l.n} packets, all {l.class}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
            {ablation.one_sided_but_expected?.length > 0 && (
              <div className="mt-3">
                <p className="text-xs leading-relaxed text-slate-500">
                  One-sided by construction, and allowed to be:
                </p>
                <div className="mt-1 grid gap-1 font-mono text-xs sm:grid-cols-2">
                  {ablation.one_sided_but_expected.map((l) => (
                    <div key={`${l.field}-${l.value}`} className="rounded border border-edge bg-ink-850 px-2 py-1">
                      <div className="flex justify-between gap-2">
                        <span className="truncate text-slate-400">{l.field}={l.value}</span>
                        <span className="text-slate-500">{l.n} packets</span>
                      </div>
                      {l.justification && (
                        <div className="mt-0.5 font-sans text-2xs leading-snug text-slate-600">{l.justification}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Section>
      )}

      <div className="grid gap-5 wide:grid-cols-[1fr_340px]">
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
                  <Cell key={i} fill={d.recall >= 0.85 ? PASS : d.recall >= 0.6 ? REVIEW : REJECT} />
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
              <div key={c.k} className="rounded border border-edge bg-ink-850 p-3">
                <div className={clsx("num text-2xl font-medium", c.tone)}>{c.v ?? "—"}</div>
                <div className="mt-0.5 text-2xs uppercase tracking-wider text-slate-500">{c.label}</div>
                <div className="text-2xs text-slate-600">{c.note}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 space-y-1.5 border-t border-edge pt-3 font-mono text-xs text-slate-400">
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
        <div className="grid gap-5 wide:grid-cols-[1fr_290px]">
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={cost?.curve ?? []} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
              <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
              <XAxis dataKey="threshold" tick={AXIS} tickFormatter={(v) => v.toFixed(2)}
                label={{ value: "reject threshold", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 10 }} />
              <YAxis tick={AXIS} tickFormatter={(v) => fmtInr(v)} width={62} />
              <Tooltip {...tooltipStyle} formatter={(v: any, n: any) => [fmtInr(Number(v)), n]}
                labelFormatter={(l) => `threshold ${Number(l).toFixed(2)}`} />
              <Legend iconType="plainline" iconSize={8} wrapperStyle={{ fontSize: 10, paddingTop: 6 }} />
              <Line type="monotone" dataKey="fraud_prevented_inr" name="fraud prevented" stroke={PASS} strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="friction_cost_inr" name="lost to false rejects" stroke={REJECT} strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="net_benefit_inr" name="net benefit" stroke={ACCENT} strokeWidth={2.5} dot={false} />
              {cost?.optimal && (
                <ReferenceLine x={cost.optimal.threshold} stroke={ACCENT} strokeDasharray="4 4"
                  label={{ value: `optimum ${cost.optimal.threshold}`, fill: ACCENT, fontSize: 10, position: "top" }} />
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
                  <label className="text-xs text-slate-400">{s.label}</label>
                  <span className="font-mono text-xs text-slate-200">{s.fmt(s.value)}</span>
                </div>
                <input type="range" min={s.min} max={s.max} step={s.step} value={s.value}
                  onChange={(e) => s.set(Number(e.target.value))}
                  className="mt-1.5 w-full accent-accent" />
              </div>
            ))}

            {cost?.optimal && (
              <div className="rounded border border-accent/40 bg-accent/[0.07] p-3">
                <div className="label text-accent">Optimal threshold</div>
                <div className="num mt-1 text-2xl font-medium text-accent">{cost.optimal.threshold}</div>
                <div className="mt-1.5 space-y-0.5 text-xs leading-relaxed text-slate-400">
                  <div>net {fmtInr(cost.optimal.net_benefit_inr)} per {cost.assumptions.per_onboardings.toLocaleString()} onboardings</div>
                  <div>FAR {fmtPct(cost.optimal.false_accept_rate, 1)} · FRR {fmtPct(cost.optimal.false_reject_rate, 1)}</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </Section>

      <div className="grid gap-5 wide:grid-cols-2">
        <Section title="Bias audit — accuracy by skin-tone proxy"
          hint="Deepfake detectors are known to degrade on darker skin. Measuring it is the minimum bar.">
          <div className="scroll-x">
            <table className="w-full min-w-[420px] text-left text-xs">
              <thead className="text-2xs uppercase tracking-wider text-slate-500">
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
                    <td className="py-2 text-right text-slate-400">{b.n}</td>
                    <td className="py-2 text-right text-slate-300">{b.auc?.toFixed(3) ?? "—"}</td>
                    <td className="py-2 text-right text-slate-400">{b.false_reject_rate !== undefined ? fmtPct(b.false_reject_rate, 1) : "—"}</td>
                    <td className="py-2 text-right text-slate-400">{b.false_accept_rate !== undefined ? fmtPct(b.false_accept_rate, 1) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 border-t border-edge pt-3 text-xs leading-relaxed text-slate-500">
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
                <Bar dataKey="count" fill={ACCENT} radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">No live traffic yet — run the gauntlet.</div>
          )}
          {ev.per_capture_mode && Object.keys(ev.per_capture_mode).length > 0 && (
            <div className="mt-4 border-t border-edge pt-3">
              <div className="label">Tamper detection by upload type</div>
              <div className="mt-2 space-y-1 font-mono text-xs">
                {Object.entries(ev.per_capture_mode).map(([mode, m]: any) => (
                  <div key={mode} className="flex justify-between gap-3">
                    <span className="text-slate-500">{mode === "scan" ? "digital scan / direct upload" : "phone photo of card"} (n={m.n})</span>
                    <span className="text-slate-300">ID-forensics AUC {m.id_forensics_auc?.toFixed(3) ?? "—"}</span>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-xs leading-relaxed text-slate-500">
                Compression-based tamper analysis reads an image&apos;s edit history. Photographing a card re-encodes
                the whole frame and largely erases it, so the two populations are reported separately rather than averaged.
              </p>
            </div>
          )}
        </Section>
      </div>

      {behav && <BehavioralSection r={behav} />}

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

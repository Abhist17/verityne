"use client";

import { useCallback, useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import clsx from "clsx";
import { api, fmtPct } from "@/lib/api";
import { Empty, ErrorBox, PageHeader, ScoreBar, Spinner, VerdictBadge } from "@/components/ui";

export default function ReviewQueuePage() {
  const [queue, setQueue] = useState<any>(null);
  const [agreement, setAgreement] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState("");

  const load = useCallback(() => {
    api.reviewQueue().then((q) => { setQueue(q); setActive((a) => a ?? q.items?.[0]?.submission_id ?? null); })
      .catch((e) => setError(e.message));
    api.reviewAgreement().then(setAgreement).catch(() => {});
  }, []);
  useEffect(load, [load]);

  const decide = async (id: string, decision: string) => {
    setBusy(id);
    try {
      await api.decide(id, decision, note || undefined);
      setNote("");
      setActive(null);
      load();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  };

  const items: any[] = queue?.items ?? [];
  const current = items.find((i) => i.submission_id === active) ?? items[0];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Review Queue"
        actions={
          <div className="flex gap-5">
            <div className="text-right">
              <div className="label">Pending</div>
              <div className="stat mt-0.5 text-xl">{queue?.pending ?? "—"}</div>
            </div>
            {agreement?.decisions > 0 && (
              <div className="text-right">
                <div className="label">Analyst agreement</div>
                <div className="stat mt-0.5 text-xl">{fmtPct(agreement.agreement_rate, 0)}</div>
              </div>
            )}
          </div>
        }
      >
        Cases Verityne declined to decide. The evidence is already surfaced — heatmaps, reasons, detector
        scores — so a reviewer confirms a judgement rather than starting an investigation.
      </PageHeader>

      {error && <ErrorBox error={error} />}
      {!queue && !error && <div className="py-24"><Spinner label="Loading queue…" /></div>}

      {queue && items.length === 0 && (
        <Empty
          title="Queue is empty"
          hint="Nothing is currently sitting in REVIEW. Run the Gauntlet to generate borderline cases, or lower the review threshold in backend/verityne/policy.yaml."
        />
      )}

      {items.length > 0 && (
        <div className="grid gap-5 wide:grid-cols-[290px_minmax(0,1fr)]">
          <div className="space-y-2">
            {items.map((it) => (
              <button
                key={it.submission_id}
                onClick={() => setActive(it.submission_id)}
                className={clsx(
                  "w-full rounded border p-2.5 text-left transition-colors duration-150",
                  it.submission_id === current?.submission_id
                    ? "border-accent/50 bg-accent/[0.07]"
                    : "border-edge bg-ink-900 hover:border-edge-strong hover:bg-ink-850"
                )}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate text-sm text-slate-200">{it.claimed_name ?? "unnamed"}</span>
                  <span className="num text-sm text-review">{it.score.toFixed(2)}</span>
                </div>
                <div className="mt-1.5"><ScoreBar score={it.score} /></div>
                <div className="mt-1.5 flex items-center gap-2 text-2xs text-slate-500">
                  <span>{it.pattern_label ?? it.attack_pattern ?? "unclassified"}</span>
                  {it.abstained && <span className="chip bg-accent/10 px-1.5 py-0 text-accent">abstained</span>}
                </div>
              </button>
            ))}
          </div>

          <AnimatePresence mode="wait">
            {current && (
              <motion.div
                key={current.submission_id}
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.18 }}
                className="space-y-4"
              >
                <div className="card-pad">
                  <div className="flex flex-wrap items-center gap-3">
                    <VerdictBadge verdict="REVIEW" size="lg" />
                    <span className="num text-lg text-slate-100">{current.score.toFixed(3)}</span>
                    <span className="chip bg-ink-800 text-slate-300 ring-1 ring-edge">
                      {current.pattern_label ?? "unclassified"}
                    </span>
                    <span className="ml-auto num text-xs text-slate-600">{current.submission_id}</span>
                  </div>
                  <p className="mt-3 text-sm leading-relaxed text-slate-300">{current.explanation}</p>
                </div>

                <div className="grid gap-4 md:grid-cols-2">
                  {["selfie", "id_document"].map((k) => {
                    const asset = current.assets?.[k];
                    const heat = current.heatmaps?.[k === "selfie" ? "selfie_deepfake" : "id_forensics"];
                    if (!asset && !heat) return null;
                    return (
                      <div key={k} className="card overflow-hidden">
                        <div className="border-b border-edge px-3 py-2 text-xs uppercase tracking-wider text-slate-500">
                          {k.replace("_", " ")}
                        </div>
                        <div className="grid grid-cols-2">
                          {asset && <img src={asset} alt={k} className="aspect-square w-full object-cover" />}
                          {heat && <img src={heat} alt={`${k} heatmap`} className="aspect-square w-full object-cover" />}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="card-pad">
                  <div className="label">Why it is here</div>
                  <ol className="mt-2 space-y-2">
                    {current.top_reasons?.map((r: string, i: number) => (
                      <li key={i} className="flex gap-3 text-sm leading-relaxed text-slate-300">
                        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-ink-800 num text-xs text-slate-400 ring-1 ring-edge">
                          {i + 1}
                        </span>
                        {r}
                      </li>
                    ))}
                  </ol>
                  <div className="mt-4 grid gap-2 border-t border-edge pt-3 sm:grid-cols-2 xl:grid-cols-3">
                    {Object.entries(current.detector_summary ?? {}).map(([k, d]: any) => (
                      <div key={k} className="flex items-center gap-2 text-xs">
                        <span className="w-32 truncate text-slate-500">{d.label}</span>
                        <span className="flex-1"><ScoreBar score={d.score} status={d.status} /></span>
                        <span className="w-8 text-right num text-slate-400">{d.score.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="card-pad">
                  <div className="label">Decision</div>
                  <textarea
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="Optional note for the audit log…"
                    rows={2}
                    className="input mt-2 resize-none bg-ink-950"
                  />
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button onClick={() => decide(current.submission_id, "approve")} disabled={!!busy}
                      className="btn border-pass/50 bg-pass/10 text-pass hover:bg-pass/20">
                      Approve merchant
                    </button>
                    <button onClick={() => decide(current.submission_id, "reject")} disabled={!!busy}
                      className="btn border-reject/50 bg-reject/10 text-reject hover:bg-reject/20">
                      Reject
                    </button>
                    <button onClick={() => decide(current.submission_id, "escalate")} disabled={!!busy} className="btn">
                      Escalate
                    </button>
                    {busy === current.submission_id && <Spinner />}
                  </div>
                  <p className="mt-2.5 text-xs leading-relaxed text-slate-600">
                    Decisions are written to the audit log with the model&apos;s score attached — that pairing is the
                    label source for the next retrain, and it is how analyst-vs-model agreement is measured.
                  </p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

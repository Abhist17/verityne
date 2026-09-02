"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import clsx from "clsx";
import { api, fmtPct } from "@/lib/api";
import { Empty, ErrorBox, PageHeader, ScoreBar, Section, Spinner, VerdictBadge } from "@/components/ui";

/**
 * `/review?case=<submission_id>` opens straight onto that case.
 *
 * The Attacks page shows a flagged submission and its verdict, and until now
 * the only way to act on one was to read its id, come here, and find it in a
 * list of 27 - so the two pages described the same case and could not hand it
 * over. A REVIEW card there is now a link to here, and this reads the id back.
 *
 * The parameter is a preference, not a command: if the case has since been
 * decided it is no longer in the queue, and the queue's own first item is used
 * rather than showing an empty panel.
 */
export default function ReviewQueuePage() {
  return (
    <Suspense fallback={<div className="py-24"><Spinner label="Loading queue…" /></div>}>
      <ReviewQueue />
    </Suspense>
  );
}

function ReviewQueue() {
  const requested = useSearchParams().get("case");
  const panel = useRef<HTMLDivElement>(null);
  const scrolled = useRef(false);
  const [queue, setQueue] = useState<any>(null);
  const [agreement, setAgreement] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState("");

  const load = useCallback(() => {
    api.reviewQueue().then((q) => {
      setQueue(q);
      const ids: string[] = (q.items ?? []).map((i: any) => i.submission_id);
      setActive((a) => a ?? (requested && ids.includes(requested) ? requested : ids[0] ?? null));
    }).catch((e) => setError(e.message));
    api.reviewAgreement().then(setAgreement).catch(() => {});
  }, [requested]);
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

  /* Arriving from an Attacks card, the case is selected but the detail panel
     sits below a 27-row queue - over 2,000px down on a laptop. The link looked
     like it had done nothing. Scroll to it once, and only when a case was asked
     for by name: a normal visit to /review should still open at the top. */
  useEffect(() => {
    if (scrolled.current || !requested || !panel.current) return;
    if (current?.submission_id !== requested) return;
    scrolled.current = true;
    panel.current.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [requested, current?.submission_id]);

  return (
    <div className="space-y-12">
      <PageHeader
        title="Review Queue"
        eyebrow="Human adjudication"
        actions={
          <div className="flex gap-5">
            <div className="text-right">
              <div className="label">Pending</div>
              <div className="stat mt-0.5 text-xl">{queue?.pending ?? "-"}</div>
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
        Cases Verityne declined to decide. The evidence is already surfaced - heatmaps, reasons, detector
        scores - so a reviewer confirms a judgement rather than starting an investigation.
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
        <div className="grid gap-x-12 gap-y-10 wide:grid-cols-[290px_minmax(0,1fr)]">
          <div className="space-y-0.5">
            {items.map((it) => (
              <button
                key={it.submission_id}
                onClick={() => setActive(it.submission_id)}
                className={clsx(
                  "w-full border-l-2 py-2.5 pl-3 pr-2 text-left transition-colors duration-150",
                  it.submission_id === current?.submission_id
                    ? "border-accent bg-accent/[0.06]"
                    : "border-transparent hover:border-edge-strong hover:bg-ink-900"
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
                ref={panel}
                key={current.submission_id}
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.18 }}
                className="space-y-12"
              >
                <div>
                  <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                    <VerdictBadge verdict="REVIEW" size="lg" />
                    <span className="num text-lg text-slate-100">{current.score.toFixed(3)}</span>
                    <span className="text-xs text-slate-500">
                      {current.pattern_label ?? "unclassified"}
                    </span>
                    <span className="num ml-auto text-2xs text-slate-700">{current.submission_id}</span>
                  </div>
                  <p className="mt-4 max-w-[68ch] text-sm leading-relaxed text-slate-300">{current.explanation}</p>
                </div>

                <div className="grid gap-5 md:grid-cols-2">
                  {["selfie", "id_document"].map((k) => {
                    const asset = current.assets?.[k];
                    const heat = current.heatmaps?.[k === "selfie" ? "selfie_deepfake" : "id_forensics"];
                    if (!asset && !heat) return null;
                    return (
                      <div key={k} className="surface overflow-hidden">
                        <div className="label border-b border-edge px-3 py-2">{k.replace("_", " ")}</div>
                        <div className="grid grid-cols-2">
                          {asset && <img src={asset} alt={k} className="aspect-square w-full object-cover" />}
                          {heat && <img src={heat} alt={`${k} heatmap`} className="aspect-square w-full object-cover" />}
                        </div>
                      </div>
                    );
                  })}
                </div>

                <Section title="Why it is here">
                  <ol className="space-y-3">
                    {current.top_reasons?.map((r: string, i: number) => (
                      <li key={i} className="flex gap-4 text-sm leading-relaxed text-slate-300">
                        <span className="num shrink-0 text-2xs text-slate-700">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span className="max-w-[62ch]">{r}</span>
                      </li>
                    ))}
                  </ol>
                  <div className="mt-6 grid gap-x-8 gap-y-2 border-t border-edge/60 pt-4 sm:grid-cols-2 xl:grid-cols-3">
                    {Object.entries(current.detector_summary ?? {}).map(([k, d]: any) => (
                      <div key={k} className="flex items-center gap-2 text-xs">
                        <span className="w-32 truncate text-slate-500">{d.label}</span>
                        <span className="flex-1"><ScoreBar score={d.score} status={d.status} /></span>
                        <span className="w-8 text-right num text-slate-400">{d.score.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </Section>

                <Section title="Decision">
                  <textarea
                    value={note}
                    onChange={(e) => setNote(e.target.value)}
                    placeholder="Optional note for the audit log…"
                    rows={2}
                    className="input resize-none bg-ink-950"
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
                    Decisions are written to the audit log with the model&apos;s score attached - that pairing is the
                    label source for the next retrain, and it is how analyst-vs-model agreement is measured.
                  </p>
                </Section>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

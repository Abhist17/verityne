"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import clsx from "clsx";
import { api, fmtPct, type GauntletResult, type GauntletSummary } from "@/lib/api";
import { Empty, ErrorBox, PageHeader, Section, StatTile, VerdictBadge } from "@/components/ui";

type Running = Omit<GauntletSummary, "results">;

/** What each number will mean, before there is one.
 *
 *  Deliberately not zeros: a zero is a measurement and there has not been one.
 *  An em dash says "not yet", and the sub-line says what the number has to be
 *  for the run to have gone well — which is the part a reader cannot infer and
 *  the part that makes watching it fill in worth doing. */
function ScoreboardAtRest({ total }: { total: number }) {
  const tiles: [string, string][] = [
    ["Detection rate", "share of fraudulent fixtures that reach a human — REVIEW counts"],
    ["False accepts", "fraud that would have been approved outright"],
    ["Genuine not passed", "rejected and reviewed are counted apart; they are not the same failure"],
    ["Wrongly rejected", "genuine applicants turned away with no human in the loop"],
    ["Wall clock", "the full API path per packet, linkage included"],
  ];
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(178px,1fr))] gap-x-8 gap-y-8">
      {tiles.map(([label, sub]) => (
        <div key={label}>
          <div className="label">{label}</div>
          <div className="stat mt-2 text-slate-700">—</div>
          <p className="mt-1.5 max-w-[26ch] text-2xs leading-relaxed text-slate-600">{sub}</p>
        </div>
      ))}
    </div>
  );
}

export default function GauntletPage() {
  const [manifest, setManifest] = useState<any>(null);
  const [results, setResults] = useState<GauntletResult[]>([]);
  const [running, setRunning] = useState<Running | null>(null);
  const [live, setLive] = useState(false);
  const [done, setDone] = useState<GauntletSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const esRef = useRef<EventSource | null>(null);
  const timerRef = useRef<any>(null);

  const load = useCallback(() => {
    api.gauntletManifest().then(setManifest).catch((e) => setError(e.message));
  }, []);
  useEffect(() => {
    load();
    return () => {
      esRef.current?.close();
      clearInterval(timerRef.current);
    };
  }, [load]);

  const start = useCallback(() => {
    setResults([]);
    setDone(null);
    setRunning(null);
    setError(null);
    setLive(true);
    const t0 = Date.now();
    setElapsed(0);
    timerRef.current = setInterval(() => setElapsed(Date.now() - t0), 90);

    const es = new EventSource("/api/gauntlet/stream");
    esRef.current = es;

    es.addEventListener("result", (ev: MessageEvent) => {
      const payload = JSON.parse(ev.data);
      setResults((prev) => [...prev, payload.result]);
      setRunning(payload.running);
    });
    es.addEventListener("error", (ev: any) => {
      // The stream also emits a named "error" event per failed fixture.
      try {
        const d = JSON.parse(ev.data);
        setError(`fixture ${d.index}: ${d.error}`);
      } catch {
        /* transport-level error; `done` or onerror will handle it */
      }
    });
    es.addEventListener("done", (ev: MessageEvent) => {
      setDone(JSON.parse(ev.data));
      setLive(false);
      clearInterval(timerRef.current);
      es.close();
      load();
    });
    es.onerror = () => {
      if (!done) setError("Stream disconnected — is the API running?");
      setLive(false);
      clearInterval(timerRef.current);
      es.close();
    };
  }, [done, load]);

  const summary: Running | null = done ?? running;
  const total = manifest?.count ?? 0;
  const items: any[] = manifest?.items ?? [];
  const progress = total ? results.length / total : 0;
  const scored = new Set(results.map((r) => r.submission_id));

  return (
    <div className="space-y-12">
      <PageHeader
        title="The Gauntlet"
        eyebrow="Fixture run"
        actions={
          <>
            {manifest && (
              <span className="num text-xs text-slate-500">
                {manifest.real} genuine · {manifest.fake} fraudulent
              </span>
            )}
            <button onClick={start} disabled={live || !total} className="btn-primary px-4 py-1.5">
              {live ? `Running ${results.length}/${total}…` : "Run the gauntlet"}
            </button>
          </>
        }
      >
        Every loaded fixture — genuine and fraudulent — scored end to end in one pass. A fake counts as caught
        if it lands in <span className="text-review">REVIEW</span> or <span className="text-reject">REJECT</span>:
        both put a human in the loop before money moves.
      </PageHeader>

      {error && <ErrorBox error={error} />}

      {!total && !error && (
        <Empty
          title="No gauntlet fixtures loaded"
          hint={
            <>
              Run <code className="num text-accent">make gauntlet</code> to load the 10 genuine + 10 fraudulent
              demo packets.
            </>
          }
        />
      )}

      {live && (
        <div className="h-0.5 w-full overflow-hidden bg-ink-800">
          <motion.div
            className="h-full bg-accent"
            animate={{ width: `${progress * 100}%` }}
            transition={{ duration: 0.25 }}
          />
        </div>
      )}

      {/* The board. Before a run this is the only thing on the page, so it has to
          show what is about to be scored rather than leaving the viewport empty;
          during a run each tile resolves as its result streams in. */}
      {total > 0 && (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(88px,1fr))] gap-2">
          {items.map((it) => {
            const r = results.find((x) => x.submission_id === it.submission_id);
            const pending = live && !scored.has(it.submission_id);
            return (
              <motion.div
                key={it.submission_id}
                animate={{ opacity: pending ? 0.35 : 1 }}
                transition={{ duration: 0.3 }}
                title={it.attack_type?.replace(/_/g, " ") ?? "genuine"}
                className={clsx(
                  "relative overflow-hidden rounded border bg-ink-900",
                  r ? (r.correct ? "border-pass/40" : "border-reject/60") : "border-edge"
                )}
              >
                {it.thumb_url ? (
                  <img src={it.thumb_url} alt="" className="aspect-square w-full object-cover" />
                ) : (
                  <div className="aspect-square w-full bg-ink-850" />
                )}
                <div className="absolute inset-x-0 bottom-0 flex items-center justify-between gap-1 bg-gradient-to-t from-ink-950 via-ink-950/85 to-transparent px-1.5 pb-1 pt-4">
                  <span className="num truncate text-2xs text-slate-400">{it.name}</span>
                  {r && <span className="num text-2xs text-slate-300">{r.score.toFixed(2)}</span>}
                </div>
                <span
                  className={clsx(
                    "absolute left-1 top-1 h-1.5 w-1.5 rounded-full ring-2 ring-ink-950",
                    it.truth === "fake" ? "bg-reject" : "bg-pass"
                  )}
                />
                {r && (
                  <span className="absolute right-1 top-1">
                    <VerdictBadge verdict={r.verdict} size="sm" />
                  </span>
                )}
              </motion.div>
            );
          })}
        </div>
      )}

      {/* The scoreboard exists before the run, not after it.
          Rendering it only once results arrive left the page ending at the
          fixture grid with most of the viewport blank, which reads as a page
          that has not loaded rather than one waiting for you to press a button.
          At rest the tiles carry their labels and what each number would have to
          be to count as good, so the page is legible before anything happens and
          the run fills it in rather than constructing it. */}
      {summary ? (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(178px,1fr))] gap-x-8 gap-y-8">
          <StatTile
            label="Detection rate"
            value={fmtPct(summary.detection_rate, 0)}
            sub={`${summary.fakes_caught}/${summary.fakes_total} fakes flagged`}
            tone={summary.detection_rate >= 0.9 ? "pass" : summary.detection_rate >= 0.7 ? "review" : "reject"}
          />
          <StatTile
            label="False accepts"
            value={fmtPct(summary.false_accept_rate, 0)}
            sub="fraud that would have been approved"
            tone={summary.false_accept_rate <= 0.1 ? "pass" : "reject"}
          />
          <StatTile
            label="Genuine not passed"
            value={fmtPct(summary.false_reject_rate, 0)}
            sub={`${summary.reals_rejected} rejected, ${summary.reals_reviewed} to review, of ${summary.reals_total}`}
            tone={summary.reals_rejected > 0 ? "reject" : summary.false_reject_rate <= 0.1 ? "pass" : "review"}
          />
          {/* Not accuracy. This page's own copy says a fake in REVIEW counts as
              caught, and then an accuracy figure counted every genuine merchant
              in REVIEW as a miss — so the tile contradicted the paragraph above
              it and read ~50% on a run where nothing had gone wrong. What a
              merchant actually cares about is whether an honest applicant was
              turned away with no human involved, and that number is zero or it
              is not. */}
          <StatTile
            label="Wrongly rejected"
            value={summary.reals_rejected}
            sub={`of ${summary.reals_total} genuine — auto-rejected with no human in the loop`}
            tone={summary.reals_rejected === 0 ? "pass" : "reject"}
          />
          <StatTile
            label={live ? "Elapsed" : "Wall clock"}
            value={`${((done?.wall_clock_ms ?? elapsed) / 1000).toFixed(1)}s`}
            sub={`${summary.mean_latency_ms.toFixed(0)} ms mean per packet`}
          />
        </div>
      ) : (
        <ScoreboardAtRest total={total} />
      )}

      {results.length > 0 && (
        <Section title="Results" right={<span className="num">{results.length} scored</span>}>
          <div className="scroll-x">
            <table className="w-full min-w-[860px] text-left text-sm">
              <thead className="border-b border-edge text-2xs uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="py-2 pr-4 font-medium">Fixture</th>
                  <th className="py-2 pr-4 font-medium">Truth</th>
                  <th className="py-2 pr-4 font-medium">Attack</th>
                  <th className="py-2 pr-4 font-medium">Verdict</th>
                  <th className="py-2 pr-4 text-right font-medium">Score</th>
                  <th className="py-2 pr-4 font-medium">Top reason</th>
                  <th className="py-2 pr-4 text-right font-medium">ms</th>
                  <th className="py-2 text-center font-medium">✓</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence initial={false}>
                  {results.map((r) => (
                    <motion.tr
                      key={r.submission_id}
                      initial={{ opacity: 0, backgroundColor: "rgba(109,140,255,0.10)" }}
                      animate={{ opacity: 1, backgroundColor: "rgba(0,0,0,0)" }}
                      transition={{ duration: 0.7 }}
                      className="border-b border-edge/60 last:border-0"
                    >
                      <td className="py-2 pr-4">
                        <div className="flex items-center gap-2">
                          {r.thumb_url && (
                            <img src={r.thumb_url} alt="" className="h-7 w-7 rounded object-cover ring-1 ring-edge" />
                          )}
                          <span className="num text-xs text-slate-300">{r.name}</span>
                        </div>
                      </td>
                      <td className="py-2 pr-4">
                        <span
                          className={clsx(
                            "chip ring-1",
                            r.truth === "fake"
                              ? "bg-reject/10 text-reject ring-reject/30"
                              : "bg-pass/10 text-pass ring-pass/30"
                          )}
                        >
                          {r.truth}
                        </span>
                      </td>
                      <td className="py-2 pr-4 text-xs text-slate-500">{r.attack_type?.replace(/_/g, " ") ?? "—"}</td>
                      <td className="py-2 pr-4">
                        <VerdictBadge verdict={r.verdict} size="sm" />
                      </td>
                      <td className="num py-2 pr-4 text-right text-slate-300">{r.score.toFixed(3)}</td>
                      <td className="max-w-md py-2 pr-4 text-xs leading-relaxed text-slate-400">
                        <span className="line-clamp-2">{r.top_reasons?.[0] ?? "—"}</span>
                      </td>
                      <td className="num py-2 pr-4 text-right text-xs text-slate-500">{r.latency_ms.toFixed(0)}</td>
                      <td className="py-2 text-center">
                        {r.correct ? <span className="text-pass">✓</span> : <span className="text-reject">✗</span>}
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {done && (
        <Section title="What this number is and is not">
          <p className="max-w-[90ch] text-xs leading-relaxed text-slate-400">
            {done.total} packets is a demonstration, not a measurement — the confidence interval on a rate
            estimated from {done.fakes_total} fakes is very wide. The statistically meaningful numbers live on
            the{" "}
            <a href="/metrics" className="text-accent hover:underline">
              Metrics
            </a>{" "}
            page, computed on an identity-disjoint held-out split that the fusion model never saw during
            training.
          </p>
        </Section>
      )}
    </div>
  );
}

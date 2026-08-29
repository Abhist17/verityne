"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import clsx from "clsx";
import { api, fmtPct, type GauntletResult, type GauntletSummary } from "@/lib/api";
import { Empty, ErrorBox, StatTile, VerdictBadge } from "@/components/ui";

type Running = Omit<GauntletSummary, "results">;

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
  useEffect(() => { load(); return () => { esRef.current?.close(); clearInterval(timerRef.current); }; }, [load]);

  const start = useCallback(() => {
    setResults([]); setDone(null); setRunning(null); setError(null); setLive(true);
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
      } catch { /* transport-level error; `done` or onerror will handle it */ }
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
  const progress = total ? results.length / total : 0;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">The Gauntlet</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-400">
            Every loaded fixture — genuine and fraudulent — scored end to end in one pass.
            A fake counts as caught if it lands in <span className="text-review">REVIEW</span> or{" "}
            <span className="text-reject">REJECT</span>: both put a human in the loop before money moves.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {manifest && (
            <span className="font-mono text-xs text-slate-500">
              {manifest.real} genuine · {manifest.fake} fraudulent
            </span>
          )}
          <button onClick={start} disabled={live || !total} className="btn-primary px-5 py-2.5">
            {live ? `Running ${results.length}/${total}…` : "Run the gauntlet"}
          </button>
        </div>
      </header>

      {error && <ErrorBox error={error} />}

      {!total && !error && (
        <Empty
          title="No gauntlet fixtures loaded"
          hint={<>Run <code className="font-mono text-accent">python backend/scripts/seed_gauntlet.py</code> to load the 10 genuine + 10 fraudulent demo packets.</>}
        />
      )}

      {live && (
        <div className="h-1 w-full overflow-hidden rounded-full bg-ink-800">
          <motion.div className="h-full rounded-full bg-accent" animate={{ width: `${progress * 100}%` }} transition={{ duration: 0.25 }} />
        </div>
      )}

      {summary && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
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
            label="False rejects"
            value={fmtPct(summary.false_reject_rate, 0)}
            sub={`${summary.reals_total - summary.reals_passed}/${summary.reals_total} genuine merchants blocked`}
            tone={summary.false_reject_rate <= 0.1 ? "pass" : "review"}
          />
          <StatTile label="Accuracy" value={fmtPct(summary.accuracy, 0)} sub={`${summary.total} packets scored`} />
          <StatTile
            label={live ? "Elapsed" : "Wall clock"}
            value={`${(((done?.wall_clock_ms ?? elapsed)) / 1000).toFixed(1)}s`}
            sub={`${summary.mean_latency_ms.toFixed(0)} ms mean per packet`}
          />
        </div>
      )}

      {results.length > 0 && (
        <div className="card overflow-hidden">
          <div className="scroll-x">
            <table className="w-full min-w-[880px] text-left text-sm">
              <thead className="border-b border-edge bg-ink-850/60 text-[11px] uppercase tracking-wider text-slate-500">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Fixture</th>
                  <th className="px-4 py-2.5 font-medium">Truth</th>
                  <th className="px-4 py-2.5 font-medium">Attack</th>
                  <th className="px-4 py-2.5 font-medium">Verdict</th>
                  <th className="px-4 py-2.5 text-right font-medium">Score</th>
                  <th className="px-4 py-2.5 font-medium">Top reason</th>
                  <th className="px-4 py-2.5 text-right font-medium">ms</th>
                  <th className="px-4 py-2.5 text-center font-medium">✓</th>
                </tr>
              </thead>
              <tbody>
                <AnimatePresence initial={false}>
                  {results.map((r) => (
                    <motion.tr
                      key={r.submission_id}
                      initial={{ opacity: 0, backgroundColor: "rgba(91,140,255,0.10)" }}
                      animate={{ opacity: 1, backgroundColor: "rgba(0,0,0,0)" }}
                      transition={{ duration: 0.7 }}
                      className="border-b border-edge/60 last:border-0"
                    >
                      <td className="px-4 py-2.5">
                        <div className="flex items-center gap-2.5">
                          {r.thumb_url && <img src={r.thumb_url} alt="" className="h-8 w-8 rounded object-cover ring-1 ring-edge" />}
                          <span className="font-mono text-xs text-slate-300">{r.name}</span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className={clsx("chip ring-1", r.truth === "fake" ? "bg-reject/10 text-reject ring-reject/30" : "bg-pass/10 text-pass ring-pass/30")}>
                          {r.truth}
                        </span>
                      </td>
                      <td className="px-4 py-2.5 text-xs text-slate-500">{r.attack_type?.replace(/_/g, " ") ?? "—"}</td>
                      <td className="px-4 py-2.5"><VerdictBadge verdict={r.verdict} size="sm" /></td>
                      <td className="px-4 py-2.5 text-right font-mono tabular-nums text-slate-300">{r.score.toFixed(3)}</td>
                      <td className="max-w-md px-4 py-2.5 text-xs leading-relaxed text-slate-400">
                        <span className="line-clamp-2">{r.top_reasons?.[0] ?? "—"}</span>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs text-slate-500">{r.latency_ms.toFixed(0)}</td>
                      <td className="px-4 py-2.5 text-center">
                        {r.correct ? <span className="text-pass">✓</span> : <span className="text-reject">✗</span>}
                      </td>
                    </motion.tr>
                  ))}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {done && (
        <div className="card-pad">
          <div className="label">What this number is and is not</div>
          <p className="mt-2 max-w-4xl text-xs leading-relaxed text-slate-400">
            {done.total} packets is a demonstration, not a measurement — the confidence interval on a rate
            estimated from {done.fakes_total} fakes is very wide. The statistically meaningful numbers live on the{" "}
            <a href="/metrics" className="text-accent hover:underline">Metrics</a> page, computed on an
            identity-disjoint held-out split that the fusion model never saw during training.
          </p>
        </div>
      )}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import { motion } from "framer-motion";
import { api, fmtPct } from "@/lib/api";
import { Empty, ErrorBox, Spinner, VerdictBadge } from "@/components/ui";

const WINDOWS = [
  { hours: 24, label: "24h" },
  { hours: 72, label: "3d" },
  { hours: 168, label: "7d" },
  { hours: 8760, label: "all" },
];

export default function AttackGalleryPage() {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [redteam, setRedteam] = useState<any[] | null>(null);

  const load = useCallback(() => {
    setError(null);
    api.attacks(hours).then(setData).catch((e) => setError(e.message));
  }, [hours]);

  useEffect(load, [load]);

  const runRedteam = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await api.redteam(3);
      setRedteam(res);
      load();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">Attack Gallery</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-400">
            Flagged submissions grouped by attack pattern. Fraud arrives in waves — one tampered PAN is noise,
            forty in an afternoon is a campaign, and only the grouping makes that visible.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex rounded-lg border border-edge bg-ink-900 p-0.5">
            {WINDOWS.map((w) => (
              <button
                key={w.hours}
                onClick={() => setHours(w.hours)}
                className={clsx("rounded-md px-3 py-1 text-xs transition",
                  hours === w.hours ? "bg-ink-700 text-slate-100" : "text-slate-500 hover:text-slate-300")}
              >
                {w.label}
              </button>
            ))}
          </div>
          <button onClick={runRedteam} disabled={busy} className="btn-primary">
            {busy ? <Spinner /> : "↯ Red-team us"}
          </button>
        </div>
      </header>

      {error && <ErrorBox error={error} />}

      {redteam && redteam.length > 0 && (
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="card-pad border-accent/40">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="label text-accent">Live red-team result</div>
            <div className="text-[11px] text-slate-500">
              generated after the model was frozen — never seen in training or evaluation
            </div>
          </div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            {redteam.map((r) => (
              <div key={r.submission_id} className="rounded-lg border border-edge bg-ink-850 p-3">
                <div className="flex items-center justify-between gap-2">
                  <VerdictBadge verdict={r.verdict} size="sm" />
                  <span className="font-mono text-sm text-slate-200">{r.final_score.toFixed(2)}</span>
                </div>
                <div className="mt-2 text-[11px] text-slate-500">{r.attack_pattern?.replace(/_/g, " ")}</div>
                <div className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-slate-400">{r.top_reasons?.[0]}</div>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {!data && !error && <div className="py-24"><Spinner label="Loading attack patterns…" /></div>}

      {data && data.groups?.length === 0 && (
        <Empty
          title="Nothing flagged in this window"
          hint="Run the Gauntlet or the red-team button to populate the gallery, then widen the time window."
        />
      )}

      {data && data.groups?.length > 0 && (
        <>
          <div className="flex flex-wrap items-center gap-3 text-sm text-slate-400">
            <span className="font-mono text-lg text-slate-100">{data.total_flagged}</span>
            <span>submissions flagged across</span>
            <span className="font-mono text-lg text-slate-100">{data.groups.length}</span>
            <span>attack patterns in the last {hours >= 8760 ? "all time" : `${hours}h`}</span>
          </div>

          <div className="space-y-4">
            {data.groups.map((g: any) => (
              <section key={g.pattern} className="card overflow-hidden">
                <div className="flex flex-wrap items-center gap-4 border-b border-edge bg-ink-850/60 px-5 py-3">
                  <div className="flex items-baseline gap-3">
                    <span className="font-mono text-2xl font-semibold tabular-nums text-slate-100">{g.count}</span>
                    <h2 className="text-sm font-medium text-slate-200">{g.label}</h2>
                  </div>
                  <div className="ml-auto flex flex-wrap items-center gap-4 text-[11px] text-slate-500">
                    <span>mean risk <span className="font-mono text-slate-300">{g.mean_score.toFixed(2)}</span></span>
                    <span>{g.merchants.length} merchant{g.merchants.length === 1 ? "" : "s"}</span>
                    {Object.keys(g.generators ?? {}).length > 0 && (
                      <span className="chip bg-ink-800 text-slate-400 ring-1 ring-edge">
                        {Object.entries(g.generators).map(([k, v]: any) => `${k} ×${v}`).join(" · ")}
                      </span>
                    )}
                  </div>
                </div>

                <div className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-4">
                  {g.samples.map((s: any) => (
                    <div key={s.submission_id} className="group overflow-hidden rounded-lg border border-edge bg-ink-850">
                      {s.thumb_url ? (
                        <img src={s.thumb_url} alt="" className="h-32 w-full object-cover" />
                      ) : (
                        <div className="flex h-32 items-center justify-center bg-ink-900 text-[11px] text-slate-600">
                          no preview
                        </div>
                      )}
                      <div className="space-y-1.5 p-2.5">
                        <div className="flex items-center justify-between gap-2">
                          <VerdictBadge verdict={s.verdict} size="sm" />
                          <span className="font-mono text-xs text-slate-300">{s.score.toFixed(2)}</span>
                        </div>
                        <div className="line-clamp-3 text-[11px] leading-relaxed text-slate-500">{s.top_reason}</div>
                        <div className="font-mono text-[10px] text-slate-600">{s.merchant_id}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

"use client";

import { useCallback, useEffect, useState } from "react";
import clsx from "clsx";
import { motion } from "framer-motion";
import { api, fmtPct } from "@/lib/api";
import { Empty, ErrorBox, PageHeader, Section, Spinner, VerdictBadge } from "@/components/ui";

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
    <div className="space-y-12">
      <PageHeader
        title="Attack Gallery"
        eyebrow="Observed traffic"
        actions={
          <>
            <div className="segment" role="group" aria-label="Time window">
              {WINDOWS.map((w) => (
                <button key={w.hours} data-active={hours === w.hours} onClick={() => setHours(w.hours)}>
                  {w.label}
                </button>
              ))}
            </div>
            <button onClick={runRedteam} disabled={busy} className="btn-primary">
              {busy ? <Spinner label="Generating…" /> : "↯ Red-team us"}
            </button>
          </>
        }
      >
        Flagged submissions grouped by attack pattern. Fraud arrives in waves — one tampered PAN is noise,
        forty in an afternoon is a campaign, and only the grouping makes that visible.
      </PageHeader>

      {error && <ErrorBox error={error} />}

      {redteam && redteam.length > 0 && (
        <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="border-l-2 border-accent/60 pl-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <div className="label text-accent">Live red-team result</div>
            <div className="text-xs text-slate-500">
              generated after the model was frozen — never seen in training or evaluation
            </div>
          </div>
          <div className="mt-4 grid gap-x-8 gap-y-5 md:grid-cols-3">
            {redteam.map((r) => (
              <div key={r.submission_id} className="border-t border-edge/60 pt-2.5">
                <div className="flex items-center justify-between gap-2">
                  <VerdictBadge verdict={r.verdict} size="sm" />
                  <span className="num text-sm text-slate-200">{r.final_score.toFixed(2)}</span>
                </div>
                <div className="mt-2 text-xs text-slate-500">{r.attack_pattern?.replace(/_/g, " ")}</div>
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
            <span className="num text-lg text-slate-100">{data.total_flagged}</span>
            <span>submissions flagged across</span>
            <span className="num text-lg text-slate-100">{data.groups.length}</span>
            <span>attack patterns in the last {hours >= 8760 ? "all time" : `${hours}h`}</span>
          </div>

          <div className="space-y-14">
            {data.groups.map((g: any) => (
              <section key={g.pattern}>
                <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                  <span className="num text-2xl font-light text-slate-100">{g.count}</span>
                  <h2 className="text-sm font-medium text-slate-200">{g.label}</h2>
                  <span className="rule-soft mx-1 hidden min-w-0 flex-1 sm:block" />
                  <div className="flex flex-wrap items-baseline gap-4 text-xs text-slate-500">
                    <span>mean risk <span className="num text-slate-300">{g.mean_score.toFixed(2)}</span></span>
                    <span>{g.merchants.length} merchant{g.merchants.length === 1 ? "" : "s"}</span>
                    {Object.keys(g.generators ?? {}).length > 0 && (
                      <span className="num text-2xs text-slate-600">
                        {Object.entries(g.generators).map(([k, v]: any) => `${k} ×${v}`).join(" · ")}
                      </span>
                    )}
                  </div>
                </div>

                {/* The thumbnail is the evidence, so it keeps its frame — this is
                    the image itself, not a box drawn around a group of text. */}
                <div className="mt-4 grid gap-x-5 gap-y-6 sm:grid-cols-2 wide:grid-cols-3 xl:grid-cols-4">
                  {g.samples.map((s: any) => (
                    <div key={s.submission_id} className="group">
                      {s.thumb_url ? (
                        <img src={s.thumb_url} alt="" className="h-32 w-full rounded object-cover" />
                      ) : (
                        <div className="flex h-32 items-center justify-center rounded bg-ink-900 text-xs text-slate-600">
                          no preview
                        </div>
                      )}
                      <div className="mt-2 space-y-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <VerdictBadge verdict={s.verdict} size="sm" />
                          <span className="num text-xs text-slate-300">{s.score.toFixed(2)}</span>
                        </div>
                        <div className="line-clamp-3 text-xs leading-relaxed text-slate-500">{s.top_reason}</div>
                        <div className="num text-2xs text-slate-600">{s.merchant_id}</div>
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

"use client";

import { useEffect, useMemo, useState } from "react";
import clsx from "clsx";
import { api, type Correction, type CorrectionsReport } from "@/lib/api";
import { Empty, ErrorBox, PageHeader, Spinner } from "@/components/ui";

/**
 * The audit trail: every belief this project held, measured, and lost.
 *
 * This page is the argument. Every other page reports what the system scores;
 * this one reports what it used to score and why that was wrong — which is the
 * only evidence a fraud team can actually use, because a vendor's headline
 * number is unfalsifiable and a vendor's list of its own defects is not.
 *
 * Two rules it holds itself to, both visible on screen:
 *
 *   * **Nothing here is prose about a number.** Each entry names the evidence
 *     file and the path its figures came from, and `build_corrections.py`
 *     resolves every one at build time — an entry citing a number no report
 *     contains fails the build. The citations are rendered rather than hidden
 *     in a tooltip, because being able to check them is the point.
 *   * **Open findings are not sorted to the bottom.** Three of these are
 *     unresolved, one of them severe, and they read in the same type as the
 *     fixed ones. A corrections page that buried its open rows would be doing
 *     the thing it exists to document.
 */

const STATUS: Record<Correction["status"], { label: string; tone: string; dot: string }> = {
  fixed: { label: "fixed", tone: "text-pass", dot: "bg-pass" },
  open: { label: "open", tone: "text-reject", dot: "bg-reject" },
  designed_around: { label: "designed around", tone: "text-review", dot: "bg-review" },
};

/** A number is "good" relative to what the finding is about, not to its sign.
 *  Held-out AUC falling from 0.913 to 0.753 is the honest direction. */
function deltaTone(c: Correction): string {
  if (c.status === "open") return "text-reject";
  return "text-pass";
}

function fmt(v: number): string {
  if (v === 0) return "0";
  if (Math.abs(v) < 0.001) return v.toExponential(1);
  if (Math.abs(v) >= 1000) return v.toLocaleString();
  return v.toFixed(v < 1 ? 4 : 3).replace(/0+$/, "").replace(/\.$/, "");
}

/** before → after, with the arrow doing the work rather than a colour alone. */
function Delta({ c }: { c: Correction }) {
  if (c.before === undefined || c.after === undefined) return null;
  return (
    <div className="flex items-baseline gap-3">
      <span className="num text-2xl font-light text-slate-500 line-through decoration-slate-700 decoration-1">
        {fmt(c.before)}
      </span>
      <span className="text-slate-600" aria-hidden>→</span>
      <span className={clsx("num text-2xl font-light", deltaTone(c))}>{fmt(c.after)}</span>
    </div>
  );
}

/** Several values against one reference line — used where the finding is not a
 *  before/after but a spread, like four generator families all at chance. */
function Series({ c }: { c: Correction }) {
  if (!c.series?.length) return null;
  const ref = c.reference?.value ?? 0.5;
  const max = Math.max(ref, ...c.series.map((s) => s.value)) * 1.15;
  return (
    <div className="space-y-1.5">
      {c.series.map((s) => {
        const bad = s.value < ref - 0.02 || s.value > ref + 0.02;
        return (
          <div key={s.label} className="grid grid-cols-[10rem_1fr_3.5rem] items-center gap-3">
            <span className="truncate text-xs text-slate-500" title={s.label}>{s.label}</span>
            <span className="relative h-3">
              <span className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-ink-750" />
              <span
                className={clsx("absolute top-1/2 h-[3px] -translate-y-1/2",
                  c.status === "open" ? "bg-reject/70" : "bg-slate-600")}
                style={{ width: `${Math.max(1, (s.value / max) * 100)}%` }}
              />
              {/* the reference: 0.5 is chance, and on several of these it is the
                  only defensible value rather than a floor to beat */}
              <span
                className="absolute top-0 h-3 w-px bg-slate-500"
                style={{ left: `${(ref / max) * 100}%` }}
                title={c.reference?.label}
              />
            </span>
            <span className={clsx("num text-right text-xs",
              bad && c.status === "open" ? "text-reject" : "text-slate-400")}>
              {fmt(s.value)}
            </span>
          </div>
        );
      })}
      {c.reference && (
        <p className="pl-[10.75rem] text-2xs text-slate-600">
          vertical rule — {c.reference.label} ({fmt(c.reference.value)})
        </p>
      )}
    </div>
  );
}

function Evidence({ c }: { c: Correction }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
      {c.evidence.map((e, i) => {
        const path = Array.isArray(e.path) ? e.path.join(" → ") : e.path;
        return (
          <span key={i} className="num text-2xs text-slate-600">
            {e.source === "code" ? (
              <>
                <span className="text-slate-500">{e.file}</span>
                <span className="text-slate-700">::</span>
                <span className="text-slate-500">{e.symbol}</span>
              </>
            ) : (
              <>
                <span className="text-slate-500">eval/{e.file}</span>
                {path && <span className="text-slate-700"> · {path}</span>}
              </>
            )}
          </span>
        );
      })}
    </div>
  );
}

function Entry({ c, legend }: { c: Correction; legend: Record<string, string> }) {
  const st = STATUS[c.status];
  const codeNote = c.evidence.find((e) => e.source === "code")?.note;
  return (
    <article className="grid gap-x-8 gap-y-4 border-t border-edge py-7 wide:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="min-w-0">
        <div className="flex items-center gap-2.5">
          <span className="num text-2xs text-slate-700">{String(c.order).padStart(2, "0")}</span>
          <span className={clsx("h-1.5 w-1.5 rounded-full", st.dot)} />
          <span className={clsx("text-2xs uppercase tracking-[0.14em]", st.tone)}>{st.label}</span>
        </div>

        <h2 className="mt-2 text-base font-medium leading-snug text-slate-100">{c.title}</h2>

        <dl className="mt-4 space-y-3">
          <div>
            <dt className="label">We believed</dt>
            <dd className="mt-1 max-w-[68ch] text-xs leading-relaxed text-slate-500">{c.believed}</dd>
          </div>
          <div>
            <dt className="label">Measuring it showed</dt>
            <dd className="mt-1 max-w-[68ch] text-xs leading-relaxed text-slate-300">{c.measured}</dd>
          </div>
          <div>
            <dt className="label">{c.status === "open" ? "Why it is still open" : "What changed"}</dt>
            <dd className="mt-1 max-w-[68ch] text-xs leading-relaxed text-slate-500">{c.outcome}</dd>
          </div>
        </dl>
      </div>

      <div className="space-y-4 wide:pt-6">
        <div>
          <div className="label">{c.metric}</div>
          <div className="mt-2">
            {c.series ? <Series c={c} /> : <Delta c={c} />}
          </div>
        </div>

        <div>
          <div className="label mb-1.5">How it was found</div>
          <p className="text-xs leading-relaxed text-slate-500">{legend[c.how_found] ?? c.how_found}</p>
        </div>

        <div>
          <div className="label mb-1.5">Evidence</div>
          <Evidence c={c} />
          {codeNote && (
            <p className="mt-1.5 max-w-[46ch] text-2xs leading-relaxed text-slate-600">{codeNote}</p>
          )}
        </div>
      </div>
    </article>
  );
}

export default function CorrectionsPage() {
  const [data, setData] = useState<CorrectionsReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.corrections().then(setData).catch((e) => setError(e.message));
  }, []);

  const counts = data?.counts ?? {};
  const open = counts.open ?? 0;
  const fixed = counts.fixed ?? 0;

  const legend = data?.how_found_legend ?? {};
  const methods = useMemo(() => {
    if (!data) return [];
    const seen = new Map<string, number>();
    for (const c of data.corrections) seen.set(c.how_found, (seen.get(c.how_found) ?? 0) + 1);
    return [...seen.entries()].sort((a, b) => b[1] - a[1]);
  }, [data]);

  return (
    <div className="space-y-8">
      <PageHeader title="Corrections" eyebrow="Audit trail">
        Every belief this project held, measured, and lost. Generated from the evidence files each
        entry cites — an entry claiming a number no report contains fails the build.
      </PageHeader>

      {error && <ErrorBox error={error} />}
      {!data && !error && <div className="py-24"><Spinner label="Loading the audit trail…" /></div>}

      {data && data.corrections.length === 0 && (
        <Empty title="No corrections recorded" hint="Run `make corrections` to assemble the file." />
      )}

      {data && data.corrections.length > 0 && (
        <>
          <div className="grid gap-6 wide:grid-cols-[minmax(0,1fr)_22rem]">
            <div className="flex flex-wrap items-baseline gap-x-8 gap-y-3">
              <span className="flex items-baseline gap-2">
                <span className="num text-3xl font-light text-slate-100">{data.n}</span>
                <span className="text-xs text-slate-500">beliefs measured and lost</span>
              </span>
              <span className="flex items-baseline gap-2">
                <span className="num text-3xl font-light text-pass">{fixed}</span>
                <span className="text-xs text-slate-500">fixed</span>
              </span>
              <span className="flex items-baseline gap-2">
                <span className="num text-3xl font-light text-reject">{open}</span>
                <span className="text-xs text-slate-500">still open</span>
              </span>
            </div>

            <div>
              <div className="label mb-1.5">How they were found</div>
              <ul className="space-y-1">
                {methods.map(([k, n]) => (
                  <li key={k} className="flex gap-3 text-2xs leading-relaxed">
                    <span className="num w-4 shrink-0 text-right text-slate-600">{n}</span>
                    <span className="text-slate-500">{legend[k] ?? k}</span>
                  </li>
                ))}
              </ul>
              {/* The distribution is the point: one of these came from reading a
                  metric. The rest came from running the thing, or from data we
                  did not make. */}
              <p className="mt-2 max-w-[40ch] text-2xs leading-relaxed text-slate-600">
                Only one of these was visible in an aggregate metric. The rest needed third-party
                data, the running product, or an attack built against ourselves.
              </p>
            </div>
          </div>

          <div>
            {data.corrections.map((c) => (
              <Entry key={c.id} c={c} legend={legend} />
            ))}
          </div>

          <p className="border-t border-edge pt-4 text-2xs leading-relaxed text-slate-600">
            {data.what_this_is} Regenerate with{" "}
            <span className="num text-slate-500">make corrections</span>.
          </p>
        </>
      )}
    </div>
  );
}

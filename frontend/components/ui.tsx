"use client";

import clsx from "clsx";
import { fmtPct, type Verdict } from "@/lib/api";

export function VerdictBadge({ verdict, size = "md" }: { verdict: Verdict; size?: "sm" | "md" | "lg" }) {
  const tone =
    verdict === "PASS"
      ? "bg-pass/12 text-pass ring-pass/40"
      : verdict === "REVIEW"
      ? "bg-review/12 text-review ring-review/40"
      : "bg-reject/12 text-reject ring-reject/40";
  const sizing = size === "lg" ? "px-4 py-1.5 text-sm" : size === "sm" ? "px-2 py-0.5 text-[11px]" : "px-3 py-1 text-xs";
  return (
    <span className={clsx("chip font-semibold uppercase tracking-wider ring-1", tone, sizing)}>
      {verdict}
    </span>
  );
}

/** Radial risk gauge. The coloured arc is the score; the ticks are the policy thresholds. */
export function ScoreDial({
  score,
  verdict,
  reviewAt = 0.4,
  rejectAt = 0.75,
  size = 168,
}: {
  score: number;
  verdict: Verdict;
  reviewAt?: number;
  rejectAt?: number;
  size?: number;
}) {
  const stroke = 12;
  const r = (size - stroke) / 2 - 6;
  const c = size / 2;
  const circ = 2 * Math.PI * r;
  const arc = 0.75; // three-quarter dial
  const color = verdict === "PASS" ? "#2dd4a7" : verdict === "REVIEW" ? "#f5b53d" : "#fb5e6d";

  const tick = (v: number) => {
    const a = (-225 + v * 270) * (Math.PI / 180);
    return { x1: c + Math.cos(a) * (r - 9), y1: c + Math.sin(a) * (r - 9), x2: c + Math.cos(a) * (r + 9), y2: c + Math.sin(a) * (r + 9) };
  };
  const t1 = tick(reviewAt);
  const t2 = tick(rejectAt);

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`Risk score ${score.toFixed(2)}`}>
      <g transform={`rotate(135 ${c} ${c})`}>
        <circle cx={c} cy={c} r={r} fill="none" stroke="#1e2432" strokeWidth={stroke}
          strokeDasharray={`${circ * arc} ${circ}`} strokeLinecap="round" />
        <circle cx={c} cy={c} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeDasharray={`${circ * arc * Math.min(1, Math.max(0, score))} ${circ}`}
          strokeLinecap="round" style={{ transition: "stroke-dasharray 700ms cubic-bezier(.2,.8,.2,1), stroke 300ms" }} />
      </g>
      <line {...t1} stroke="#f5b53d" strokeWidth="2" opacity="0.75" />
      <line {...t2} stroke="#fb5e6d" strokeWidth="2" opacity="0.75" />
      <text x={c} y={c - 2} textAnchor="middle" className="fill-slate-100 font-mono text-[30px] font-semibold">
        {score.toFixed(2)}
      </text>
      <text x={c} y={c + 20} textAnchor="middle" className="fill-slate-500 text-[10px] uppercase tracking-[0.18em]">
        risk score
      </text>
    </svg>
  );
}

export function StatTile({
  label, value, sub, tone = "default",
}: { label: string; value: React.ReactNode; sub?: React.ReactNode; tone?: "default" | "pass" | "review" | "reject" }) {
  const toneCls =
    tone === "pass" ? "text-pass" : tone === "review" ? "text-review" : tone === "reject" ? "text-reject" : "text-slate-100";
  return (
    <div className="card-pad">
      <div className="label">{label}</div>
      <div className={clsx("stat mt-1.5", toneCls)}>{value}</div>
      {sub && <div className="mt-1 text-xs leading-relaxed text-slate-500">{sub}</div>}
    </div>
  );
}

/** Horizontal 0..1 meter with the policy thresholds marked. */
export function ScoreBar({ score, confidence, status }: { score: number; confidence?: number; status?: string }) {
  const dim = status !== "ok";
  const color = score >= 0.75 ? "bg-reject" : score >= 0.4 ? "bg-review" : "bg-pass";
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-ink-700">
      <div
        className={clsx("h-full rounded-full transition-all duration-700", color, dim && "opacity-30")}
        style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
      />
      {confidence !== undefined && (
        <div className="absolute inset-y-0 right-0 flex items-center pr-1 text-[9px] text-slate-500" />
      )}
      <div className="pointer-events-none absolute inset-y-0 left-[40%] w-px bg-review/50" />
      <div className="pointer-events-none absolute inset-y-0 left-[75%] w-px bg-reject/50" />
    </div>
  );
}

export function Empty({ title, hint }: { title: string; hint?: React.ReactNode }) {
  return (
    <div className="card flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
      <div className="text-sm font-medium text-slate-300">{title}</div>
      {hint && <div className="max-w-lg text-xs leading-relaxed text-slate-500">{hint}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2.5 text-sm text-slate-400">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent/30 border-t-accent" />
      {label}
    </div>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="card border-reject/40 bg-reject/5 p-4 text-sm text-reject">
      <div className="font-medium">Something went wrong</div>
      <div className="mt-1 font-mono text-xs opacity-80">{error}</div>
    </div>
  );
}

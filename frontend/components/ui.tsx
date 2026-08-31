"use client";

import clsx from "clsx";
import type { Verdict } from "@/lib/api";

/** Every page opens the same way: title, one line of what you are looking at,
 *  and its controls pinned right. Defined once so the pages cannot drift. */
export function PageHeader({
  title,
  children,
  actions,
}: {
  title: string;
  children?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-start justify-between gap-x-6 gap-y-3 pb-1">
      {/* The basis is what makes this wrap instead of overflow: with `min-w-0`
          alone the title shrinks toward nothing and keeps the actions on the
          same line until they run off the right edge. Claiming a real minimum
          width forces the actions onto their own row first. */}
      <div className="min-w-0 flex-1 basis-[min(100%,26rem)]">
        <h1 className="text-xl font-semibold text-slate-100">{title}</h1>
        {children && <p className="prose-note mt-1 max-w-[68ch]">{children}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

const VERDICT_TONE: Record<Verdict, string> = {
  PASS: "bg-pass/10 text-pass ring-pass/30",
  REVIEW: "bg-review/10 text-review ring-review/30",
  REJECT: "bg-reject/10 text-reject ring-reject/30",
};

export function VerdictBadge({ verdict, size = "md" }: { verdict: Verdict; size?: "sm" | "md" | "lg" }) {
  const sizing =
    size === "lg" ? "px-2.5 py-1 text-sm" : size === "sm" ? "px-1.5 py-0.5 text-2xs" : "px-2 py-0.5 text-xs";
  return (
    <span className={clsx("chip font-semibold uppercase tracking-wider ring-1", VERDICT_TONE[verdict], sizing)}>
      {verdict}
    </span>
  );
}

export const scoreTone = (score: number, reviewAt = 0.4, rejectAt = 0.75) =>
  score >= rejectAt ? "reject" : score >= reviewAt ? "review" : "pass";

const HEX = { pass: "#3ddc97", review: "#e8b04b", reject: "#f4626f" } as const;

/**
 * Risk gauge. A three-quarter arc, with the merchant's two policy thresholds cut
 * into the track as gaps rather than drawn on top of it.
 *
 * The gaps matter: the number alone does not say whether 0.61 is a comfortable
 * pass or one point below a rejection, and that depends entirely on a policy
 * that differs per merchant. Cutting the track shows the bands the score is
 * being read against, so the same dial stays honest under a strict policy and a
 * lenient one without any legend.
 */
export function ScoreDial({
  score,
  verdict,
  reviewAt = 0.4,
  rejectAt = 0.75,
  size = 150,
}: {
  score: number;
  verdict: Verdict;
  reviewAt?: number;
  rejectAt?: number;
  size?: number;
}) {
  const stroke = 8;
  const r = size / 2 - stroke - 8;
  const c = size / 2;
  const SWEEP = 270; // degrees of live track
  const circ = 2 * Math.PI * r;
  const color = HEX[verdict === "PASS" ? "pass" : verdict === "REVIEW" ? "review" : "reject"];
  const clamped = Math.min(1, Math.max(0, score));

  // Track split into three band segments with a 2.5deg gap at each threshold.
  const GAP = 2.5;
  const bands = [
    { from: 0, to: reviewAt, tone: HEX.pass },
    { from: reviewAt, to: rejectAt, tone: HEX.review },
    { from: rejectAt, to: 1, tone: HEX.reject },
  ];

  const seg = (from: number, to: number) => {
    const a0 = from * SWEEP + (from > 0 ? GAP / 2 : 0);
    const a1 = to * SWEEP - (to < 1 ? GAP / 2 : 0);
    const len = Math.max(0, ((a1 - a0) / 360) * circ);
    return { offset: (a0 / 360) * circ, len };
  };

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="img"
      aria-label={`Risk score ${score.toFixed(2)}, verdict ${verdict}`}
      className="shrink-0"
    >
      <g transform={`rotate(135 ${c} ${c})`} fill="none" strokeLinecap="butt">
        {bands.map((b) => {
          const { offset, len } = seg(b.from, b.to);
          return (
            <circle
              key={b.from}
              cx={c}
              cy={c}
              r={r}
              stroke={b.tone}
              strokeOpacity={0.16}
              strokeWidth={stroke}
              strokeDasharray={`${len} ${circ}`}
              strokeDashoffset={-offset}
            />
          );
        })}
        <circle
          cx={c}
          cy={c}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${(clamped * SWEEP * circ) / 360} ${circ}`}
          style={{ transition: "stroke-dasharray 650ms cubic-bezier(.2,.8,.2,1), stroke 250ms" }}
        />
      </g>
      <text x={c} y={c + 2} textAnchor="middle" fill="#f1f5f9" className="font-mono text-[26px] font-medium">
        {score.toFixed(2)}
      </text>
      <text x={c} y={c + 19} textAnchor="middle" fill="#64748b" className="text-[9px] uppercase tracking-[0.18em]">
        risk
      </text>
    </svg>
  );
}

export function StatTile({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "default" | "pass" | "review" | "reject";
}) {
  const toneCls =
    tone === "pass" ? "text-pass" : tone === "review" ? "text-review" : tone === "reject" ? "text-reject" : "text-slate-100";
  return (
    <div className="card-pad">
      <div className="label">{label}</div>
      <div className={clsx("stat mt-2", toneCls)}>{value}</div>
      {sub && <div className="mt-1.5 text-xs leading-relaxed text-slate-500">{sub}</div>}
    </div>
  );
}

/** Horizontal 0..1 meter. Thresholds are notches cut through the bar, for the
 *  same reason the dial cuts its track: a score means nothing without them. */
export function ScoreBar({
  score,
  status,
  reviewAt = 0.4,
  rejectAt = 0.75,
}: {
  score: number;
  confidence?: number;
  status?: string;
  reviewAt?: number;
  rejectAt?: number;
}) {
  const dim = status !== undefined && status !== "ok";
  const tone = scoreTone(score, reviewAt, rejectAt);
  const fill = tone === "reject" ? "bg-reject" : tone === "review" ? "bg-review" : "bg-pass";
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-ink-750">
      <div
        className={clsx("h-full rounded-full transition-all duration-700", fill, dim && "opacity-30")}
        style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
      />
      {[reviewAt, rejectAt].map((t) => (
        <span
          key={t}
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-ink-950"
          style={{ left: `${t * 100}%` }}
        />
      ))}
    </div>
  );
}

export function Empty({ title, hint, icon }: { title: string; hint?: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="card flex flex-col items-center justify-center gap-2 px-6 py-14 text-center">
      {icon && <div className="mb-1 text-slate-700">{icon}</div>}
      <div className="text-sm font-medium text-slate-300">{title}</div>
      {hint && <div className="max-w-[52ch] text-xs leading-relaxed text-slate-500">{hint}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm">
      <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-[1.5px] border-current/25 border-t-current" />
      {label}
    </span>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="card border-reject/40 bg-reject/[0.04] p-3.5">
      <div className="text-sm font-medium text-reject">Something went wrong</div>
      <div className="num mt-1 break-words text-xs text-reject/70">{error}</div>
    </div>
  );
}

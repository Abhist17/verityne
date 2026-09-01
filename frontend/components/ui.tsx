"use client";

import clsx from "clsx";
import type { Verdict } from "@/lib/api";

/** Title, optional controls, nothing else. The explanatory paragraph that used
 *  to live here is gone on purpose: it was the same three lines on every page
 *  and it pushed the actual content below the fold. */
export function PageHeader({
  title,
  actions,
  children,
}: {
  title: string;
  actions?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <header className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-3">
      <div className="min-w-0 flex-1 basis-[min(100%,20rem)]">
        <h1 className="text-xl font-medium tracking-tight text-slate-100">{title}</h1>
        {children && <p className="mt-1 max-w-[62ch] text-sm text-slate-500">{children}</p>}
      </div>
      {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
    </header>
  );
}

/** Section heading: a tiny label over a fading hairline. This is what replaced
 *  the card border — it opens a section without boxing it. */
export function SectionLabel({ children, right }: { children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-baseline gap-3">
      <span className="label shrink-0">{children}</span>
      <span className="rule-soft min-w-0 flex-1" />
      {right && <span className="shrink-0 text-2xs text-slate-600">{right}</span>}
    </div>
  );
}

/** A section on a dense page: a sentence-case title inline with a fading
 *  hairline, an optional hint beneath it, and the content in open space.
 *
 *  This is SectionLabel's shape at a larger scale, and it is what replaced the
 *  card on the analysis pages. SectionLabel's uppercase micro-label is right for
 *  a one-word heading ("Why", "Detectors"); these headings are sentences, and
 *  0.14em of tracking across forty characters stops being readable. Same rule,
 *  same rhythm, different type role. */
export function Section({
  title,
  hint,
  right,
  children,
}: {
  title: React.ReactNode;
  hint?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section>
      <div className="flex items-baseline gap-3">
        <h2 className="shrink-0 text-sm font-medium text-slate-200">{title}</h2>
        <span className="rule-soft min-w-0 flex-1" />
        {right && <span className="shrink-0 text-2xs text-slate-600">{right}</span>}
      </div>
      {hint && <p className="mt-1.5 max-w-[74ch] text-2xs leading-relaxed text-slate-500">{hint}</p>}
      <div className="mt-4">{children}</div>
    </section>
  );
}

const VERDICT_TEXT: Record<Verdict, string> = {
  PASS: "text-pass",
  REVIEW: "text-review",
  REJECT: "text-reject",
};

/** No pill, no ring — just the word, in its colour, at weight. At this size the
 *  word is the badge; a chip around it only added a rectangle to the page. */
export function VerdictBadge({ verdict, size = "md" }: { verdict: Verdict; size?: "sm" | "md" | "lg" }) {
  const sizing =
    size === "lg"
      ? "text-2xl tracking-[-0.01em]"
      : size === "sm"
      ? "text-2xs tracking-[0.14em]"
      : "text-xs tracking-[0.12em]";
  return (
    <span className={clsx("font-medium uppercase", VERDICT_TEXT[verdict], sizing)}>{verdict}</span>
  );
}

export const scoreTone = (score: number, reviewAt = 0.4, rejectAt = 0.75) =>
  score >= rejectAt ? "reject" : score >= reviewAt ? "review" : "pass";

import { PASS, REJECT, REVIEW } from "@/lib/palette";

const HEX = { pass: PASS, review: REVIEW, reject: REJECT } as const;

/**
 * The risk readout: an oversized numeral, the verdict beside it, and one full
 * width meter with the policy bands named underneath.
 *
 * This replaced a radial dial. The dial looked like a gauge and read like an
 * ornament: an arc cannot show you *where the thresholds are* without a legend,
 * and the thresholds are the entire reason a 0.61 means something. A straight
 * line can — the bands are laid out along it, labelled, in the merchant's own
 * policy positions, so a strict policy and a lenient one visibly differ.
 */
export function ScoreMeter({
  score,
  verdict,
  reviewAt = 0.4,
  rejectAt = 0.75,
}: {
  score: number;
  verdict: Verdict;
  reviewAt?: number;
  rejectAt?: number;
}) {
  const pct = Math.min(100, Math.max(0, score * 100));
  const color = HEX[verdict === "PASS" ? "pass" : verdict === "REVIEW" ? "review" : "reject"];

  return (
    <div>
      <div className="label">Risk</div>
      <div className="mt-2 flex flex-wrap items-baseline gap-x-6 gap-y-1">
        <span className="hero" style={{ color }}>
          {score.toFixed(2)}
        </span>
        <VerdictBadge verdict={verdict} size="lg" />
      </div>

      <div className="relative mt-5 h-[3px] w-full rounded-full bg-ink-800">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
        {[reviewAt, rejectAt].map((t) => (
          <span
            key={t}
            className="absolute -top-1 h-[11px] w-px bg-ink-600"
            style={{ left: `${t * 100}%` }}
            aria-hidden
          />
        ))}
      </div>
      {/* Band names sit at their own thresholds, so the scale is self-describing. */}
      <div className="relative mt-2 h-4 text-2xs text-slate-600">
        <span className="absolute left-0">pass</span>
        <span className="absolute" style={{ left: `${reviewAt * 100}%` }}>
          review {reviewAt}
        </span>
        <span className="absolute" style={{ left: `${rejectAt * 100}%` }}>
          reject {rejectAt}
        </span>
      </div>
    </div>
  );
}

/** A number and its name. No border, no background — the scale does the work. */
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
    tone === "pass"
      ? "text-pass"
      : tone === "review"
      ? "text-review"
      : tone === "reject"
      ? "text-reject"
      : "text-slate-100";
  return (
    <div>
      <div className="label">{label}</div>
      <div className={clsx("stat mt-2", toneCls)}>{value}</div>
      {sub && <div className="mt-1.5 text-xs leading-snug text-slate-600">{sub}</div>}
    </div>
  );
}

/** Hairline meter for a list row. Thin on purpose: at this weight a column of
 *  them reads as a distribution rather than as a stack of progress bars. */
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
    <div className="relative h-px w-full bg-ink-750">
      <div
        className={clsx("absolute inset-y-0 left-0 transition-all duration-700", fill, dim && "opacity-25")}
        style={{ width: `${Math.min(100, Math.max(0, score * 100))}%` }}
      />
      {[reviewAt, rejectAt].map((t) => (
        <span
          key={t}
          className="absolute -top-[3px] h-[7px] w-px bg-ink-700"
          style={{ left: `${t * 100}%` }}
          aria-hidden
        />
      ))}
    </div>
  );
}

export function Empty({ title, hint, icon }: { title: string; hint?: React.ReactNode; icon?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-20 text-center">
      {icon && <div className="mb-1 text-ink-600">{icon}</div>}
      <div className="text-sm text-slate-400">{title}</div>
      {hint && <div className="max-w-[46ch] text-xs leading-relaxed text-slate-600">{hint}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm">
      <span className="h-3 w-3 shrink-0 animate-spin rounded-full border-[1.5px] border-current/20 border-t-current" />
      {label}
    </span>
  );
}

export function ErrorBox({ error }: { error: string }) {
  return (
    <div className="border-l-2 border-reject/60 pl-3">
      <div className="text-sm text-reject">Something went wrong</div>
      <div className="num mt-1 break-words text-xs text-reject/60">{error}</div>
    </div>
  );
}

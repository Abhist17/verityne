"use client";

import clsx from "clsx";
import type { Verdict } from "@/lib/api";

/** Title, optional controls, nothing else. The explanatory paragraph that used
 *  to live here is gone on purpose: it was the same three lines on every page
 *  and it pushed the actual content below the fold.
 *
 *  The title is set at 34px rather than the 19px it used to be. At 19px an <h1>
 *  is the same size as the row labels under it, so a page opened with no visible
 *  hierarchy at all — every page looked like the middle of a page. The scale
 *  already carried a `3xl` step with -0.025em on it and nothing was using it.
 *
 *  `eyebrow` is the small caps line above the title: it names the section a page
 *  belongs to, which is what lets the title itself stay one word.
 */
export function PageHeader({
  title,
  eyebrow,
  actions,
  children,
}: {
  title: string;
  eyebrow?: React.ReactNode;
  actions?: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <header>
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-4">
        <div className="min-w-0 flex-1 basis-[min(100%,20rem)]">
          {eyebrow && <div className="eyebrow mb-2.5">{eyebrow}</div>}
          <h1 className="page-title">{title}</h1>
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {/* The rule anchors the title. A large heading on a dark page has no
          baseline to sit on and floats away from the column beneath it. */}
      <div className="title-rule" />
      {children && (
        <p className="mt-4 max-w-[68ch] text-sm leading-relaxed text-slate-500">{children}</p>
      )}
    </header>
  );
}

/**
 * One member of a numbered set: an ordinal, a name, and a line about it.
 *
 * The set this exists for is the six detectors, which were a column of hairline
 * rows — correct as data, wrong as an introduction, because a reader meeting
 * this system for the first time needs to see that there are *six separate
 * things* before reading what any one of them does. Six boxes say that at a
 * glance; six rules do not.
 *
 * Colour stays out of it. The ordinal is ink, the name is slate, the sentence is
 * slate — the palette's rule is that hue means a verdict, and a detector that
 * has not run yet has no verdict to report.
 */
export function NumberedCard({
  index,
  title,
  children,
}: {
  index: number;
  title: React.ReactNode;
  children?: React.ReactNode;
}) {
  return (
    <div className="card card-hover">
      <div className="ordinal">{String(index).padStart(2, "0")}</div>
      <div className="mt-3 text-sm font-medium text-slate-100">{title}</div>
      {children && <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{children}</p>}
    </div>
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
/** The one sentence on a page that *is* the argument.
 *
 *  Used sparingly and never for a detail — if every paragraph is a callout then
 *  none of them is. The label is what makes it work: it says what kind of claim
 *  is about to be made before the claim arrives. */
export function Callout({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <aside className="callout">
      <div className="label text-accent">{label}</div>
      <div className="mt-2 max-w-[76ch] text-sm leading-relaxed text-slate-300">{children}</div>
    </aside>
  );
}

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
    // `min-w-0` is load-bearing, not tidiness. A grid or flex item defaults to
    // `min-width: auto`, so a section refuses to shrink below the intrinsic
    // width of its widest child — and a Recharts container or an SVG graph has a
    // large intrinsic width. On a phone that pushed the whole document 394px
    // wider than the viewport and put a horizontal scrollbar under every chart
    // page. The charts already scale once the box is allowed to.
    <section className="min-w-0">
      <div className="flex min-w-0 items-baseline gap-3">
        {/* Not `shrink-0`. Monospace runs materially wider than the sans it
            replaced, so a title like "What the headline number is made of" is
            422px — wider than a phone — and a heading that cannot shrink took
            the document with it. It wraps; the rule takes whatever is left. */}
        <h2 className="num min-w-0 text-sm font-normal tracking-tight text-slate-200">{title}</h2>
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

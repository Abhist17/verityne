"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Mark } from "@/components/Mark";
import { Hint } from "@/components/Hint";

/**
 * The seven tabs, each with the sentence it needs to not be a guess.
 *
 * Six of these names are internal vocabulary. "Gauntlet" and "Corrections" in
 * particular mean nothing to someone meeting this product for the first time,
 * and a row of seven equal words gives them no way in except clicking each one
 * to find out. `blurb` is what a tab would say if you asked it what it was for;
 * `meta` is the one concrete fact that makes it real rather than a slogan.
 */
const LINKS: { href: string; label: string; blurb: string; meta?: string }[] = [
  {
    href: "/verify",
    label: "Verify",
    blurb:
      "Score one applicant. Drop in a selfie, an ID document and a liveness clip, and get a verdict back with the reasons and heatmaps behind it.",
    meta: "6 detectors · ~2.5s",
  },
  {
    href: "/gauntlet",
    label: "Gauntlet",
    blurb:
      "A gauntlet is a run of trials you put something through to see if it survives. This one is 20 prepared cases - 10 genuine applicants and 10 known frauds - scored live through the real API so you can watch the system get them right or wrong.",
    meta: "10 genuine · 10 fraudulent",
  },
  {
    href: "/metrics",
    label: "Metrics",
    blurb:
      "The held-out evaluation: how well this system actually separates fraud from honest applicants, on packets it was never trained on. ROC, recall per attack type, and a cost curve you can drag.",
    meta: "105 held-out packets",
  },
  {
    href: "/corrections",
    label: "Corrections",
    blurb:
      "Every belief this project held, measured, and turned out to be wrong about - each one linked to the evidence file that disproved it. Published rather than quietly fixed.",
    meta: "9 findings",
  },
  {
    href: "/attacks",
    label: "Attacks",
    blurb:
      "Flagged submissions grouped by the attack they used, so a wave of the same fraud looks like a wave instead of ten unrelated cases.",
  },
  {
    href: "/threat",
    label: "Threat",
    blurb:
      "Fraud rings. One fake merchant is an incident; forty sharing a face or a file is organised, and only linking them makes that visible.",
  },
  {
    href: "/review",
    label: "Review",
    blurb:
      "The human queue. Cases the policy would not decide alone land here for a person to approve, reject or escalate.",
  },
];

export function Nav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  const up = health?.status === "ok";
  const device = health?.device;

  return (
    <header className="sticky top-0 z-40 border-b border-edge bg-ink-950/90 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-[1180px] items-center gap-6 px-6 py-3">
        <Link href="/" className="group flex shrink-0 items-center gap-2" aria-label="Verityne home">
          <Mark className="h-[17px] w-[20px] text-accent" />
          <span className="font-display text-base tracking-tight text-slate-50">Verityne</span>
        </Link>

        {/* The active tab is marked by an accent underline flush with the
            header's own border, so the panel below reads as belonging to that
            tab. The rule is drawn on the header and the marker on the link, and
            the link's `-bottom-3` is the `py-3` on the row - the two have to
            stay in step or the marker floats above the border it sits in. */}
        {/* Scrolls rather than overflows. Seven tabs do not fit a 390px
            viewport, and a flex row that cannot shrink pushed the document
            233px wide - which put a horizontal scrollbar under every page on a
            phone, not just this one. `no-bar` hides the bar itself: the swipe
            still works where it is needed, and on a desktop where all seven fit
            there is no longer a stray rule sitting under the tabs. */}
        <nav className="scroll-x no-bar -mx-1 flex min-w-0 flex-1 items-center gap-5 px-1">
          {LINKS.map((l) => {
            const active = pathname === l.href || (l.href !== "/" && pathname.startsWith(l.href));
            return (
              <Hint key={l.href} title={l.label} body={l.blurb} meta={l.meta}>
                <Link
                  href={l.href}
                  aria-current={active ? "page" : undefined}
                  className={clsx(
                    "relative whitespace-nowrap text-sm transition-colors duration-150",
                    active ? "text-slate-100" : "text-slate-500 hover:text-slate-200"
                  )}
                >
                  {l.label}
                  {active && (
                    <span
                      className="absolute inset-x-0 -bottom-3 h-px bg-accent"
                      aria-hidden
                    />
                  )}
                </Link>
              </Hint>
            );
          })}
        </nav>

        <div className="ml-auto flex shrink-0 items-center gap-3">
          {/* One dot. If it is green the system is up; that is the whole status
              bar's job, and the device string was chrome nobody read. */}
          <Hint
            align="right"
            title={up ? "API online" : "API offline"}
            body={
              up
                ? "The scoring API is reachable and its models are loaded. Every number on this dashboard is being read from it live, not from a fixture."
                : "The dashboard cannot reach the scoring API on :8000. Pages that read live data will show an error until it is back."
            }
            meta={up && device ? `running on ${device}` : undefined}
          >
            <span
              className={clsx("h-1.5 w-1.5", up ? "bg-pass" : "bg-reject")}
              aria-hidden
            />
          </Hint>
          {/* The dot alone carries the status visually, which was the point. It
              carries nothing at all to a screen reader, and `title` is not an
              accessible name - so the state is also announced in text. */}
          <span role="status" className="sr-only">
            {up ? `API online${device ? `, running on ${device}` : ""}` : "API offline"}
          </span>
          {/* The header's one action. A dashboard whose nav is seven equal tabs
              never tells a first-time visitor which door to open; this is the
              door. Hidden on narrow viewports, where the tab strip already owns
              every pixel of the row. */}
          <Link href="/verify" className="btn-accent hidden py-1 text-xs wide:inline-flex">
            Score a packet
          </Link>
        </div>
      </div>
    </header>
  );
}

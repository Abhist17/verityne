"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Mark } from "@/components/Mark";

const LINKS = [
  { href: "/verify", label: "Verify" },
  { href: "/gauntlet", label: "Gauntlet" },
  { href: "/metrics", label: "Metrics" },
  { href: "/corrections", label: "Corrections" },
  { href: "/attacks", label: "Attacks" },
  { href: "/threat", label: "Threat" },
  { href: "/review", label: "Review" },
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
            phone, not just this one. */}
        <nav className="scroll-x -mx-1 flex min-w-0 flex-1 items-center gap-5 px-1">
          {LINKS.map((l) => {
            const active = pathname === l.href || (l.href !== "/" && pathname.startsWith(l.href));
            return (
              <Link
                key={l.href}
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
            );
          })}
        </nav>

        <div className="ml-auto flex shrink-0 items-center gap-3">
          {/* One dot. If it is green the system is up; that is the whole status
              bar's job, and the device string was chrome nobody read. */}
          <span
            className={clsx("h-1.5 w-1.5", up ? "bg-pass" : "bg-reject")}
            title={up ? `online · ${device ?? ""}` : "offline"}
            aria-hidden
          />
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

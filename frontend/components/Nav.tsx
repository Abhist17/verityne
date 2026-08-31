"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Verify" },
  { href: "/gauntlet", label: "Gauntlet" },
  { href: "/metrics", label: "Metrics" },
  { href: "/attacks", label: "Attacks" },
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
    <header className="sticky top-0 z-40 border-b border-edge bg-ink-950/80 backdrop-blur-md">
      <div className="mx-auto flex w-full max-w-[1440px] items-center gap-5 px-5 py-2.5">
        <Link href="/" className="group flex items-center gap-2" aria-label="Verityne home">
          <svg viewBox="0 0 24 24" className="h-[18px] w-[18px] text-accent" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6l7-3z" strokeLinejoin="round" />
            <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="text-base font-semibold tracking-tight text-slate-100">Verityne</span>
        </Link>

        {/* The active tab is marked by an underline flush with the header's own
            border, so the panel below reads as belonging to that tab. */}
        <nav className="-mb-2.5 flex items-center gap-0.5 self-stretch">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={clsx(
                  "relative px-2.5 pb-2.5 pt-1 text-sm transition-colors duration-150",
                  active ? "text-slate-100" : "text-slate-500 hover:text-slate-300"
                )}
              >
                {l.label}
                {active && <span className="absolute inset-x-2.5 -bottom-px h-px bg-accent" />}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-2.5 text-2xs">
          {device && (
            <span className="num rounded border border-edge px-1.5 py-0.5 uppercase text-slate-500">{device}</span>
          )}
          <span className="flex items-center gap-1.5 text-slate-500">
            <span className={clsx("h-1.5 w-1.5 rounded-full", up ? "bg-pass" : "bg-reject")} />
            {up ? "online" : "offline"}
          </span>
        </div>
      </div>
    </header>
  );
}

"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import clsx from "clsx";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";

const LINKS = [
  { href: "/", label: "Live Verify" },
  { href: "/gauntlet", label: "Gauntlet" },
  { href: "/metrics", label: "Metrics" },
  { href: "/attacks", label: "Attack Gallery" },
  { href: "/review", label: "Review Queue" },
];

export function Nav() {
  const pathname = usePathname();
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth({ status: "down" }));
  }, []);

  const up = health?.status === "ok";
  const device = health?.device;
  const model = health?.models?.deepfake_classifier;

  return (
    <header className="sticky top-0 z-40 border-b border-edge bg-ink-950/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-[1500px] items-center gap-6 px-6 py-3">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="relative flex h-7 w-7 items-center justify-center">
            <span className="absolute inset-0 rounded-md bg-accent/20" />
            <span className="absolute inset-0 rounded-md ring-1 ring-accent/50" />
            <svg viewBox="0 0 24 24" className="relative h-4 w-4 text-accent" fill="none" stroke="currentColor" strokeWidth="2.2">
              <path d="M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6l7-3z" strokeLinejoin="round" />
              <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-slate-100">Verityne</span>
        </Link>

        <nav className="flex items-center gap-1">
          {LINKS.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={clsx(
                  "rounded-lg px-3 py-1.5 text-sm transition",
                  active ? "bg-ink-800 text-slate-100 ring-1 ring-edge" : "text-slate-400 hover:text-slate-200"
                )}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto flex items-center gap-3 text-xs text-slate-500">
          {model && <span className="hidden font-mono lg:inline">{String(model).split("/").pop()}</span>}
          {device && (
            <span className="rounded border border-edge px-1.5 py-0.5 font-mono uppercase">{device}</span>
          )}
          <span className="flex items-center gap-1.5">
            <span className={clsx("h-1.5 w-1.5 rounded-full", up ? "bg-pass" : "bg-reject")} />
            {up ? "API up" : "API down"}
          </span>
        </div>
      </div>
    </header>
  );
}

import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import localFont from "next/font/local";
import Link from "next/link";
import "./globals.css";
import { Footer } from "@/components/Footer";
import { Nav } from "@/components/Nav";
import { headline } from "@/lib/evidence";

// Loaded through next/font so the files are self-hosted at build time and the
// browser makes no third-party request at runtime.
const sans = Inter({ subsets: ["latin"], display: "swap", variable: "--font-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], display: "swap", variable: "--font-mono" });

// The display face. A bitmap monospace drawn at 11px, which is why it is used
// at exactly two scales: title sizes, where the staircase edges are the point,
// and the 10.5px micro-label, which is its design size. Everything numeric
// stays in JetBrains Mono - see app/fonts/README.md.
const display = localFont({
  src: "./fonts/DepartureMono-Regular.woff2",
  display: "swap",
  variable: "--font-display",
  weight: "400",
});

export const metadata: Metadata = {
  title: {
    default: "Verityne - Deepfake-aware KYC verification",
    template: "%s · Verityne",
  },
  description:
    "Six independent detectors, a calibrated fusion layer, an explanation behind every verdict - "
    + "and a published record of every belief this project measured and lost.",
};

/**
 * The strip above the nav.
 *
 * Not a banner for its own sake. This project's one claim about itself is that
 * it publishes what it got wrong, and that claim was living inside the landing
 * hero - which meant it was invisible from the six pages a reviewer actually
 * spends their time on. It is a count read from `corrections.json` at build
 * time, so it cannot drift from the file behind it.
 */
function AnnouncementBar() {
  const h = headline();
  if (!h.corrections) return null;
  return (
    <div className="border-b border-edge bg-ink-950">
      <div className="mx-auto flex w-full max-w-[1180px] items-center justify-center gap-3 px-6 py-2">
        <span className="tag shrink-0">Audit</span>
        <Link
          href="/corrections"
          className="group truncate text-xs text-slate-500 transition-colors hover:text-slate-200"
        >
          <span className="text-slate-300">{h.corrections.n} things</span> this project believed
          and measured wrong · {h.corrections.open} still open{" "}
          <span className="inline-block text-accent transition-transform group-hover:translate-x-0.5">
            ↗
          </span>
        </Link>
      </div>
    </div>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${sans.variable} ${mono.variable} ${display.variable}`}
    >
      {/* A column layout so the footer sits at the bottom of the viewport on a
          short page instead of floating halfway up it. */}
      <body className="flex min-h-screen flex-col">
        <AnnouncementBar />
        <Nav />
        <main className="mx-auto w-full max-w-[1180px] flex-1 px-6 pb-24 pt-8">{children}</main>
        <Footer />
      </body>
    </html>
  );
}

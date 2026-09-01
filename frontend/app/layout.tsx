import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Footer } from "@/components/Footer";
import { Nav } from "@/components/Nav";

// Loaded through next/font so the files are self-hosted at build time and the
// browser makes no third-party request at runtime.
const sans = Inter({ subsets: ["latin"], display: "swap", variable: "--font-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], display: "swap", variable: "--font-mono" });

export const metadata: Metadata = {
  title: {
    default: "Verityne — Deepfake-aware KYC verification",
    template: "%s · Verityne",
  },
  description:
    "Six independent detectors, a calibrated fusion layer, an explanation behind every verdict — "
    + "and a published record of every belief this project measured and lost.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      {/* A column layout so the footer sits at the bottom of the viewport on a
          short page instead of floating halfway up it. */}
      <body className="flex min-h-screen flex-col">
        <Nav />
        <main className="mx-auto w-full max-w-[1180px] flex-1 px-6 pb-24 pt-8">{children}</main>
        <Footer />
      </body>
    </html>
  );
}

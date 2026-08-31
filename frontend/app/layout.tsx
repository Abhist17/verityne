import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Nav } from "@/components/Nav";

// Loaded through next/font so the files are self-hosted at build time and the
// browser makes no third-party request at runtime.
const sans = Inter({ subsets: ["latin"], display: "swap", variable: "--font-sans" });
const mono = JetBrains_Mono({ subsets: ["latin"], display: "swap", variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Verityne — Deepfake-aware KYC verification",
  description:
    "Five independent detectors, a calibrated fusion layer, and an explanation for every verdict.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto w-full max-w-[1440px] px-5 pb-20 pt-5">{children}</main>
      </body>
    </html>
  );
}

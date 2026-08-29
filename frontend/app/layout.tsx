import type { Metadata } from "next";
import "./globals.css";
import { Nav } from "@/components/Nav";

export const metadata: Metadata = {
  title: "Verityne — Deepfake-aware KYC verification",
  description:
    "Five independent detectors, a calibrated fusion layer, and an explanation for every verdict.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <Nav />
        <main className="mx-auto w-full max-w-[1500px] px-6 pb-24 pt-6">{children}</main>
      </body>
    </html>
  );
}

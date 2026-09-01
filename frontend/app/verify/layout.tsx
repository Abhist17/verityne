import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Verify",
  description:
    "Score one KYC packet: six detectors in parallel, a verdict, the three reasons behind it, "
    + "and the evidence each one rests on.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}

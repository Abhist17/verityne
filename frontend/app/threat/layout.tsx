import type { Metadata } from "next";

/** Tab title for this route.
 *
 *  The page itself is a client component and cannot export metadata, so the
 *  title lives in a server layout beside it. Without this every tab in the
 *  dashboard read the same default, which is the kind of detail that makes a
 *  working product feel like a prototype.
 */
export const metadata: Metadata = {
  title: "Threat Intelligence",
  description: "Fraud rings as a node-link graph, generator fingerprint mix, and a live feed of flagged submissions.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}

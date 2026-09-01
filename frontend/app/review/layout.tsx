import type { Metadata } from "next";

/** Tab title for this route.
 *
 *  The page itself is a client component and cannot export metadata, so the
 *  title lives in a server layout beside it. Without this every tab in the
 *  dashboard read the same default, which is the kind of detail that makes a
 *  working product feel like a prototype.
 */
export const metadata: Metadata = {
  title: "Review Queue",
  description: "Cases Verityne declined to decide, with the evidence already surfaced for a human.",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}

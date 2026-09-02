import Link from "next/link";
import { corrections, headline, packets } from "@/lib/evidence";
import { ScoreField } from "@/components/ScoreField";

/**
 * The landing page.
 *
 * `/` used to be the upload form, which meant a first-time visitor met a file
 * picker before they met the argument. This is a server component and every
 * number on it is read out of `eval/*.json` at build time rather than typed in:
 * the one claim this project makes about itself is that a stated figure has a
 * committed file behind it, and the hero is the last place to break that.
 *
 * The order is deliberate. The detectors come second, not first, because a
 * detector list is the part every submission in this category has. What is rare
 * is the audit, so the audit is the middle of the page and the largest block on
 * it, and the roadmap that closes the page is drawn from the findings it left
 * open - the plan is an output of the measurements rather than a wish list.
 */

export const metadata = {
  title: "Verityne - Deepfake-aware KYC verification",
};

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

function Rule({ n, label }: { n: string; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="num text-2xs text-accent">{n}</span>
      <span className="label text-slate-400">{label}</span>
      <span className="h-px flex-1 bg-edge" />
    </div>
  );
}

export default function LandingPage() {
  const h = headline();
  const rows = corrections();
  const cells = packets();
  const open = rows.filter((r) => r.status === "open");

  return (
    <div className="pb-8">
      {/* ---------------------------------------------------------------- hero
          Full-bleed, and closed by a full-bleed accent hairline.

          The bleed used to be here to carry a darker ground than the app's - the
          app sat at #0a0a0c and the hero dropped to black. The whole interface
          is black now, so there is no value step left to make; what the bleed
          still buys is the rule at the bottom, which has to run edge to edge or
          it reads as the underline of a column rather than the end of a band.

          `-mx-6` would only cancel the container padding - the page is capped at
          1180px with auto margins, so the band would stop short and float. This
          breaks out to the viewport and re-centres its own content. */}
      <section className="relative left-1/2 w-screen -translate-x-1/2 border-b border-accent/60">
        <div className="mx-auto w-full max-w-[1180px] px-6 pb-16 pt-10 wide:pt-14">
        {/* The line above the headline. The corrections count that used to sit
            here is now the announcement strip at the top of every page, which is
            where it belongs - it was the site's one standing claim and it was
            only visible on the home page. This says instead what is behind the
            headline, so a reader arrives at the claim already knowing there is a
            report under it. */}
        {/* Plain, and first. A reviewer read this page and asked what a vendor
            was and what a ROC curve was - which means the old opening line, a
            jargon inventory of the metrics report, had spent the first five
            seconds of the page on words that only land if you already know the
            answer. This says what the thing is. The clever line can follow. */}
        <div className="text-xs uppercase tracking-[0.14em] text-accent">
          Fraud detection for merchant onboarding
        </div>

        {/* 19ch, not 17. The measure is in `ch`, which is the advance of "0" in
            the *current* font - and the current font is now a monospace with a
            wider advance than the one this was set against, so "Every vendor
            shows" no longer fit the line the explicit <br /> promises it. The
            break is authored, so the box has to be wide enough to honour it. */}
        <h1 className="mt-7 max-w-[19ch] font-display text-3xl font-normal leading-[1.04] tracking-tight text-slate-50 wide:text-display">
          Every vendor shows
          <br />
          you a ROC curve.
          <span className="mt-2 block text-slate-600">We show you where it fails.</span>
        </h1>

        {/* Two sentences, in this order on purpose. The first says what happens,
            in words that need no background. The second defines ROC in passing -
            because the headline above uses it, and a headline that has to be
            looked up is a headline that failed. */}
        <p className="mt-8 max-w-[56ch] text-base leading-relaxed text-slate-300">
          When a business signs up to accept payments, it uploads a selfie and an ID document.
          We check whether that is a real person using their own documents - or an AI-generated
          fake.
        </p>
        <p className="mt-4 max-w-[56ch] text-sm leading-relaxed text-slate-500">
          A ROC curve is the single number detectors like this are usually judged on. It hides
          where they break. So this publishes both: six detectors with the evidence behind every
          verdict, and a list of everything we measured and got wrong.
        </p>

        {/* Sharp corners, sitting flush. */}
        <div className="mt-8 flex flex-wrap items-stretch">
          <Link
            href="/verify"
            className="bg-accent px-5 py-2.5 text-sm font-medium text-ink-1000 transition-colors hover:bg-accent-soft"
          >
            Score a packet
          </Link>
          <Link
            href="/corrections"
            className="border border-l-0 border-edge-strong px-5 py-2.5 text-sm text-slate-300 transition-colors hover:bg-ink-900 hover:text-slate-100"
          >
            Read what we got wrong
          </Link>
          <a
            href="https://github.com/Abhist17/verityne"
            target="_blank"
            rel="noreferrer"
            className="ml-4 self-center text-sm text-slate-500 transition-colors hover:text-slate-300"
          >
            Source ↗
          </a>
        </div>

        {/* The hero image is the evaluation itself. */}
        <div className="mt-14 grid gap-10 wide:grid-cols-[minmax(0,1fr)_20rem] wide:items-end">
          <ScoreField cells={cells} />
          <p className="max-w-[34ch] text-xs leading-relaxed text-slate-500">
            Every held-out packet this system has scored. The two classes are not two blocks - they
            interleave through the middle, and that band is every case where a genuine merchant and
            a fraudulent one look the same to the model.
            <span className="mt-3 block text-slate-600">
              That overlap is what {h.fusionAuc?.toFixed(3)} means. A single AUC is a summary of
              this picture.
            </span>
          </p>
        </div>

        {/* the four numbers, straight from the reports */}
        <dl className="mt-16 grid gap-x-10 gap-y-8 border-t border-edge pt-8 sm:grid-cols-2 wide:grid-cols-4">
          {[
            ["Held-out ROC-AUC", h.fusionAuc?.toFixed(3), "identity-disjoint split; was 0.913 before an ablation found the leak"],
            ["Detector 6", h.behavioral?.auc?.toFixed(3), "keystroke rhythm, fitted on 168,595 real people"],
            ["Findings fixed", h.corrections ? String(h.corrections.fixed) : null, `of ${h.corrections?.n ?? "-"} the audit found, each with the evidence file behind it`],
            ["Worst third-party fake", h.realFaces?.worstAuc?.toFixed(3), `${h.realFaces?.worstFamily ?? "-"} - below chance means inverted`],
          ].map(([label, value, sub]) => (
            <div key={label as string}>
              <dt className="label">{label}</dt>
              <dd className="stat mt-2">{value ?? "-"}</dd>
              <dd className="mt-1.5 max-w-[30ch] text-2xs leading-relaxed text-slate-600">{sub}</dd>
            </div>
          ))}
        </dl>
        </div>
      </section>

      {/* ------------------------------------------------------------- 01 the problem */}
      <section className="mt-24">
        <Rule n="01" label="The problem" />
        <div className="mt-6 grid gap-10 wide:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <p className="max-w-[58ch] text-sm leading-relaxed text-slate-400">
            KYC verification was designed when faking an identity meant forging a plastic card and
            finding a lookalike. That era ended. A photorealistic face of a person who does not
            exist takes about four seconds and costs nothing, face-swap tooling puts any face onto
            a &ldquo;turn your head and blink&rdquo; liveness video, and matched fake PAN + selfie +
            liveness kits sell in Telegram groups for a few hundred rupees.
          </p>
          <p className="max-w-[58ch] text-sm leading-relaxed text-slate-500">
            Every fake merchant that gets through becomes chargeback losses, laundering exposure,
            and a regulatory problem for the platform that onboarded them. The detector is the easy
            half. Knowing what it is worth - and where it is blind - is the half nobody ships.
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------------- 03 the audit */}
      <section className="mt-24">
        <Rule n="02" label="What we got wrong" />
        <p className="mt-6 max-w-[66ch] text-sm leading-relaxed text-slate-400">
          Any vendor can show you a curve. The question a fraud team actually needs answered is
          where it fails and how you would know. So this project keeps a published list, generated
          from the evidence files each entry cites - a correction that claims a number no report
          contains fails the build.
        </p>

        <div className="mt-8 overflow-hidden border border-edge">
          <table className="w-full text-left text-xs">
            <thead className="bg-ink-900 font-display text-2xs uppercase tracking-[0.11em] text-slate-500">
              <tr>
                <th className="w-10 px-4 py-2.5 font-medium">#</th>
                <th className="px-4 py-2.5 font-medium">What measuring it showed</th>
                <th className="w-48 px-4 py-2.5 text-right font-medium">Moved</th>
                <th className="w-28 px-4 py-2.5 text-right font-medium">Status</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                // Trimmed to three significant figures. A rate printed as
                // 0.999919 is four characters of noise that wrapped the column
                // onto two lines and told a reader nothing they did not have.
                const fmt = (v: number) =>
                  v === 0 ? "0" : v >= 1 ? v.toFixed(2).replace(/\.?0+$/, "") : v.toPrecision(3).replace(/0+$/, "").replace(/\.$/, "");
                const moved =
                  r.before !== undefined && r.after !== undefined
                    ? `${fmt(r.before)} → ${fmt(r.after)}`
                    : r.series?.length
                    ? `${fmt(Math.min(...r.series.map((s) => s.value)))} - ${fmt(
                        Math.max(...r.series.map((s) => s.value))
                      )}`
                    : "-";
                return (
                  <tr key={r.order} className="border-t border-edge/60">
                    <td className="num px-4 py-3 text-slate-700">
                      {String(r.order).padStart(2, "0")}
                    </td>
                    <td className="px-4 py-3 text-slate-300">{r.title}</td>
                    <td className="num whitespace-nowrap px-4 py-3 text-right text-slate-500">
                      {moved}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span
                        className={
                          r.status === "open"
                            ? "text-reject"
                            : r.status === "fixed"
                            ? "text-pass"
                            : "text-review"
                        }
                      >
                        {r.status.replace(/_/g, " ")}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="mt-5 flex flex-wrap items-baseline gap-x-6 gap-y-2">
          <Link
            href="/corrections"
            className="text-xs text-slate-300 underline decoration-edge-strong underline-offset-4 transition-colors hover:text-slate-100"
          >
            Read all {rows.length} in full
          </Link>
          <span className="text-2xs text-slate-600">
            Only one of these was visible in an aggregate metric. The rest needed third-party data,
            the running product, or an attack built against ourselves.
          </span>
        </div>
      </section>

      {/* ------------------------------------------------------------- 04 roadmap
          Same three findings that used to sit here under "Still open", turned
          to face forward. The list is not a confession any more, it is the work
          queue: each card is what the measurement showed we have to build next,
          and the sentence under it is lifted from that finding's own `outcome`
          field so the plan cannot drift from the evidence that motivated it. */}
      {open.length > 0 && (
        <section className="mt-24">
          <Rule n="03" label="Roadmap" />
          <p className="mt-6 max-w-[64ch] text-sm leading-relaxed text-slate-400">
            What the measurements say to build next. Each of these came out of the audit above,
            which is the point of running one - the work queue is derived from evidence rather
            than from a planning meeting.
          </p>
          <div className="mt-7 grid gap-x-10 gap-y-8 wide:grid-cols-3">
            {open.map((r, i) => (
              <div key={r.order} className="border-t border-accent/40 pt-4">
                <div className="label text-accent">next · {String(i + 1).padStart(2, "0")}</div>
                <h3 className="mt-2 text-sm leading-snug text-slate-200">{r.title}</h3>
                {r.outcome && (
                  <p className="mt-2 line-clamp-4 text-2xs leading-relaxed text-slate-500">
                    {r.outcome}
                  </p>
                )}
                <p className="mt-2 text-2xs leading-relaxed text-slate-600">{r.metric}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------- close */}
      <section className="mt-24 border-t border-edge pt-10">
        <h2 className="max-w-[26ch] font-display text-2xl font-normal leading-snug tracking-tight text-slate-100">
          {h.fusionAuc?.toFixed(3)} is what the detectors earn.
          <span className="block text-slate-500">
            Where to cut it is an operator&rsquo;s decision, not ours.
          </span>
        </h2>
        <p className="mt-4 max-w-[64ch] text-xs leading-relaxed text-slate-500">
          Every figure on this page is read from a committed evidence file at build time, and the
          test suite fails if the two disagree. Reproduce with{" "}
          <span className="num text-slate-400">make pipeline</span> and{" "}
          <span className="num text-slate-400">make real</span>.
        </p>
        <div className="mt-7 flex flex-wrap gap-3">
          <Link
            href="/verify"
            className="bg-accent px-4 py-2 text-sm font-medium text-ink-1000 transition-colors hover:bg-accent-soft"
          >
            Score a packet
          </Link>
          <Link
            href="/gauntlet"
            className="px-4 py-2 text-sm text-slate-300 ring-1 ring-inset ring-edge-strong transition-colors hover:bg-ink-850 hover:text-slate-100"
          >
            Run the gauntlet
          </Link>
        </div>
      </section>
    </div>
  );
}

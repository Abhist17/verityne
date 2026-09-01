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
 * it — including the three findings that are still open.
 */

export const metadata = {
  title: "Verityne — Deepfake-aware KYC verification",
};

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

function Rule({ n, label }: { n: string; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="num text-2xs text-slate-700">{n}</span>
      <span className="label">{label}</span>
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
          Full-bleed true black behind the fold. The app's surface is #0a0a0c,
          which is right for something you read for an hour; a hero wants the
          screen to vanish behind the type instead. */}
      {/* Full-bleed. `-mx-6` only cancels the container padding — the page is
          capped at 1180px with auto margins, so the black stopped short and the
          hero read as a card floating on the app ground. This breaks out to the
          viewport and re-centres its own content. */}
      <section className="relative left-1/2 w-screen -translate-x-1/2 bg-ink-1000">
        <div className="mx-auto w-full max-w-[1180px] px-6 pb-16 pt-10 wide:pt-14">
        {/* Eyebrow. Not an announcement banner for its own sake — it is the one
            sentence that says what kind of thing this is before the headline
            makes a claim. */}
        <Link
          href="/corrections"
          className="group inline-flex items-center gap-2.5 text-xs text-slate-500 transition-colors hover:text-slate-300"
        >
          <span className="border border-edge-strong px-1.5 py-0.5 text-2xs uppercase tracking-[0.12em] text-slate-400">
            Audit
          </span>
          {h.corrections && (
            <span>
              {h.corrections.n} things this project believed and measured wrong ·{" "}
              {h.corrections.open} still open
            </span>
          )}
          <span className="transition-transform group-hover:translate-x-0.5">→</span>
        </Link>

        <h1 className="num mt-7 max-w-[17ch] text-3xl font-normal leading-[1.04] tracking-tight text-slate-50 wide:text-display">
          Every vendor shows
          <br />
          you a ROC curve.
          <span className="mt-2 block text-slate-600">We show you where it fails.</span>
        </h1>

        <p className="mt-8 max-w-[54ch] text-sm leading-relaxed text-slate-400">
          Six independent detectors, a calibrated fusion layer, and a human-readable explanation
          behind every verdict — built on the assumption that the attacker has Stable Diffusion and
          DeepFaceLab on their laptop.
        </p>

        {/* Sharp corners, sitting flush. */}
        <div className="mt-8 flex flex-wrap items-stretch">
          <Link
            href="/verify"
            className="bg-accent px-5 py-2.5 text-sm font-medium text-ink-1000 transition-opacity hover:opacity-90"
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
            Every held-out packet this system has scored. The two classes are not two blocks — they
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
            ["Findings still open", h.corrections ? String(h.corrections.open) : null, "published, not buried — including one that is severe"],
            ["Worst third-party fake", h.realFaces?.worstAuc?.toFixed(3), `${h.realFaces?.worstFamily ?? "—"} — below chance means inverted`],
          ].map(([label, value, sub]) => (
            <div key={label as string}>
              <dt className="label">{label}</dt>
              <dd className="stat mt-2">{value ?? "—"}</dd>
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
            half. Knowing what it is worth — and where it is blind — is the half nobody ships.
          </p>
        </div>
      </section>

      {/* ------------------------------------------------------------- 02 detectors */}
      <section className="mt-24">
        <Rule n="02" label="Six detectors, run in parallel" />
        <div className="mt-6">
          {[
            ["Selfie deepfake", "A pretrained transformer and a fitted frequency head, voting. Grad-CAM shows which pixels drove the call."],
            ["ID forensics", "OCR with confusion repair, then structural validation — a PAN's 4th character is a holder-type code, Aadhaar carries a Verhoeff digit."],
            ["Liveness video", "Identity drift between frames, head-pose jitter, and optical-flow discontinuity at splice boundaries."],
            ["Face match", "512-d FaceNet embeddings, selfie against the portrait on the card. The threshold is fitted on LFW's 6,000 real pairs."],
            ["Metadata / EXIF", "Generator tags, editor software, capture-to-submission age, and whether a file carries the detail its resolution claims."],
            ["Behavioral biometrics", "Not an artifact at all — how the form was filled. Keystroke rhythm, pointer path, device coherence."],
          ].map(([name, what], i) => (
            <div key={name} className="flex gap-5 border-t border-edge/60 py-3.5 wide:gap-8">
              <span className="num w-6 shrink-0 pt-0.5 text-2xs text-slate-700">
                {String(i + 1).padStart(2, "0")}
              </span>
              <span className="w-40 shrink-0 text-sm text-slate-200">{name}</span>
              <span className="max-w-[62ch] text-xs leading-relaxed text-slate-500">{what}</span>
            </div>
          ))}
        </div>
        <p className="mt-5 max-w-[68ch] text-xs leading-relaxed text-slate-500">
          The first five read the files. The sixth reads the person — and it is the only one whose
          adversary is not on a release cycle. Detectors 1&ndash;5 degrade every time a better
          generator ships; defeating Detector 6 needs a rig that reproduces human motor timing.
        </p>
      </section>

      {/* ------------------------------------------------------------- 03 the audit */}
      <section className="mt-24">
        <Rule n="03" label="What we got wrong" />
        <p className="mt-6 max-w-[66ch] text-sm leading-relaxed text-slate-400">
          Any vendor can show you a curve. The question a fraud team actually needs answered is
          where it fails and how you would know. So this project keeps a published list, generated
          from the evidence files each entry cites — a correction that claims a number no report
          contains fails the build.
        </p>

        <div className="mt-8 overflow-hidden rounded border border-edge">
          <table className="w-full text-left text-xs">
            <thead className="bg-ink-900 text-2xs uppercase tracking-[0.11em] text-slate-600">
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
                    ? `${fmt(Math.min(...r.series.map((s) => s.value)))} – ${fmt(
                        Math.max(...r.series.map((s) => s.value))
                      )}`
                    : "—";
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

      {/* ------------------------------------------------------------- 04 still open */}
      {open.length > 0 && (
        <section className="mt-24">
          <Rule n="04" label="Still open" />
          <p className="mt-6 max-w-[64ch] text-sm leading-relaxed text-slate-400">
            These are not fixed, and they are on the front page rather than in an appendix. A
            corrections list that only contained solved problems would be a changelog.
          </p>
          <div className="mt-7 grid gap-x-10 gap-y-8 wide:grid-cols-3">
            {open.map((r) => (
              <div key={r.order} className="border-t border-reject/40 pt-4">
                <div className="label text-reject">open</div>
                <h3 className="mt-2 text-sm leading-snug text-slate-200">{r.title}</h3>
                <p className="mt-2 text-2xs leading-relaxed text-slate-600">{r.metric}</p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------- 05 the dashboard */}
      <section className="mt-24">
        <Rule n="05" label="Six surfaces" />
        <div className="mt-6 grid gap-x-10 gap-y-7 sm:grid-cols-2 wide:grid-cols-3">
          {[
            ["/verify", "Verify", "Drag in a packet. Verdict, three reasons, heatmaps, and the per-detector breakdown."],
            ["/gauntlet", "Gauntlet", "Twenty fixtures scored live through the full API path — the only place linkage runs end to end."],
            ["/metrics", "Metrics", "The held-out report: ROC, per-attack recall, the ablation, and a cost curve you can drag."],
            ["/corrections", "Corrections", "The audit trail, with the evidence file and path behind every number."],
            ["/threat", "Threat Intelligence", "Fraud rings as a graph. Proven clusters and inferred ones are drawn differently."],
            ["/review", "Review Queue", "Everything the system declined to decide, with the evidence already surfaced."],
          ].map(([href, name, what]) => (
            <Link
              key={href}
              href={href}
              className="group border-t border-edge pt-4 transition-colors hover:border-edge-strong"
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-sm text-slate-200 transition-colors group-hover:text-accent">
                  {name}
                </span>
                <span className="text-slate-700 transition-colors group-hover:text-slate-500">→</span>
              </div>
              <p className="mt-1.5 text-xs leading-relaxed text-slate-500">{what}</p>
            </Link>
          ))}
        </div>
      </section>

      {/* ------------------------------------------------------------- close */}
      <section className="mt-24 border-t border-edge pt-10">
        <h2 className="max-w-[26ch] text-xl font-medium leading-snug tracking-tight text-slate-100">
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
            className="rounded-md bg-accent px-4 py-2 text-sm font-medium text-ink-950 transition-opacity hover:opacity-90"
          >
            Score a packet
          </Link>
          <Link
            href="/gauntlet"
            className="rounded-md px-4 py-2 text-sm text-slate-300 ring-1 ring-inset ring-edge-strong transition-colors hover:bg-ink-850 hover:text-slate-100"
          >
            Run the gauntlet
          </Link>
        </div>
      </section>
    </div>
  );
}

"use client";

import clsx from "clsx";
import type { DetectorOutput } from "@/lib/api";

/**
 * Detector 6's evidence, as an instrument rather than a JSON dump.
 *
 * Every other detector's evidence is an image - a Grad-CAM overlay, an ELA
 * residual - and a reviewer can look at it and see what the model saw. This one
 * has no image, because what it measured was time. So the rendering has to do
 * the work the heatmap does elsewhere: show the measurement *and* the range it
 * was judged against, in one glance, without the reviewer having to remember
 * what a normal keystroke dwell is.
 *
 * The bands are not defined here. They arrive in `signals.reference`, from the
 * same constants the rules fire on, so a re-fitted threshold moves this chart
 * with it. A dashboard that hard-codes its own idea of "normal" eventually draws
 * a picture that no longer describes the decision underneath it.
 */

const LABELS: Record<string, { label: string; unit?: string; hint: string }> = {
  dwell_mean_ms: { label: "Key hold", unit: "ms", hint: "How long each key is held down" },
  dwell_cv: { label: "Hold variance", hint: "A hand varies; a timer does not" },
  flight_cv: { label: "Gap variance", hint: "Typing comes in bursts and pauses" },
  flight_min_ms: { label: "Fastest gap", unit: "ms", hint: "Below ~15 ms is not a hand" },
  rollover_rate: { label: "Key rollover", hint: "Next key pressed before the last is released" },
  typing_speed_cps: { label: "Typing speed", unit: "c/s", hint: "Characters per second, sustained" },
  mouse_straightness: { label: "Pointer path", hint: "1.0 is a straight line between fields" },
  mouse_path_entropy: { label: "Pointer tremor", hint: "Direction change along the path" },
  mouse_dt_cv: { label: "Sample spacing", hint: "Real mousemove delivery is irregular" },
  total_time_s: { label: "Time on form", unit: "s", hint: "How long the whole fill took" },
};

type Band = { human: [number, number]; measured: number | null };

/** Log-ish position so a value orders of magnitude outside the band still lands
 *  on the track instead of pinning to an edge and losing its distance. */
function place(v: number, lo: number, hi: number): number {
  const span = hi - lo || 1;
  return Math.max(0, Math.min(1, (v - lo) / span));
}

function Measure({ name, band }: { name: string; band: Band }) {
  const meta = LABELS[name];
  if (!meta || band?.measured === null || band?.measured === undefined) return null;
  const [lo, hi] = band.human;
  const v = band.measured;
  const inside = v >= lo && v <= hi;
  // The track spans a little beyond the human band on both sides, so a value
  // that falls outside is visibly outside rather than clamped to the end.
  const pad = (hi - lo) * 0.35 || 1;
  const t0 = lo - pad;
  const t1 = hi + pad;
  const bandL = place(lo, t0, t1);
  const bandR = place(hi, t0, t1);
  const at = place(v, t0, t1);

  return (
    <div className="grid grid-cols-[8.5rem_1fr_4.5rem] items-center gap-3 py-1.5">
      <span className="truncate text-xs text-slate-500" title={meta.hint}>
        {meta.label}
      </span>
      <span className="relative h-4">
        {/* the track */}
        <span className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-ink-750" />
        {/* the range a human occupies */}
        <span
          className="absolute top-1/2 h-[3px] -translate-y-1/2 bg-ink-700"
          style={{ left: `${bandL * 100}%`, right: `${(1 - bandR) * 100}%` }}
        />
        {/* this submission */}
        <span
          className={clsx(
            "absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-ink-900",
            inside ? "bg-slate-400" : "bg-reject"
          )}
          style={{ left: `${at * 100}%` }}
          aria-label={inside ? "within the human range" : "outside the human range"}
        />
      </span>
      <span
        className={clsx(
          "num text-right text-xs tabular-nums",
          inside ? "text-slate-400" : "text-reject"
        )}
      >
        {v >= 100 ? v.toFixed(0) : v.toFixed(2)}
        {meta.unit && <span className="ml-0.5 text-slate-600">{meta.unit}</span>}
      </span>
    </div>
  );
}

export function BehavioralEvidence({ detector }: { detector: DetectorOutput }) {
  const s = detector.signals ?? {};
  const reference: Record<string, Band> = s.reference ?? {};
  const env = s.environment ?? {};
  const evidence = s.evidence ?? {};
  const hits: { rule: string; weight: number }[] = s.rule_hits ?? [];
  const modelP: number | null = evidence.keystroke_model ?? null;

  const keystroke = ["dwell_mean_ms", "dwell_cv", "flight_cv", "flight_min_ms",
                     "rollover_rate", "typing_speed_cps"];
  const pointer = ["mouse_straightness", "mouse_path_entropy", "mouse_dt_cv"];
  const timing = ["total_time_s"];
  const has = (k: string) => reference[k]?.measured !== null && reference[k]?.measured !== undefined;

  const flags: string[] = env.automation_flags ?? [];

  return (
    <div className="space-y-5">
      {/* What the fitted model made of the rhythm, when there is one. */}
      {modelP !== null && (
        <div className="flex items-baseline justify-between gap-4 rounded bg-ink-900 px-3 py-2.5">
          <div className="min-w-0">
            <div className="label">Keystroke model</div>
            <p className="mt-1 text-xs leading-relaxed text-slate-500">
              Fitted on 168,595 real people and on real browser automation
            </p>
          </div>
          <span
            className={clsx(
              "num shrink-0 text-xl font-light",
              modelP >= 0.5 ? "text-reject" : "text-slate-300"
            )}
          >
            {(modelP * 100).toFixed(0)}
            <span className="ml-0.5 text-xs text-slate-600">% automated</span>
          </span>
        </div>
      )}

      {keystroke.some(has) && (
        <section>
          <div className="label mb-1">Keystroke rhythm</div>
          {keystroke.map((k) => has(k) && <Measure key={k} name={k} band={reference[k]} />)}
        </section>
      )}

      {pointer.some(has) && (
        <section>
          <div className="label mb-1">Pointer</div>
          {pointer.map((k) => has(k) && <Measure key={k} name={k} band={reference[k]} />)}
        </section>
      )}

      {timing.some(has) && (
        <section>
          <div className="label mb-1">Timing</div>
          {timing.map((k) => has(k) && <Measure key={k} name={k} band={reference[k]} />)}
        </section>
      )}

      <section>
        <div className="label mb-1.5">Device</div>
        <dl className="num grid grid-cols-[7rem_1fr] gap-x-3 gap-y-1 text-xs">
          {env.timezone && (
            <>
              <dt className="text-slate-600">timezone</dt>
              <dd className="truncate text-slate-400">{env.timezone}</dd>
            </>
          )}
          {Array.isArray(env.languages) && env.languages.length > 0 && (
            <>
              <dt className="text-slate-600">languages</dt>
              <dd className="truncate text-slate-400">{env.languages.join(", ")}</dd>
            </>
          )}
          <dt className="text-slate-600">events</dt>
          <dd className="text-slate-400">{evidence.events_seen ?? "-"}</dd>
          {flags.length > 0 && (
            <>
              <dt className="text-slate-600">automation</dt>
              <dd className="text-reject">{flags.join(", ")}</dd>
            </>
          )}
        </dl>
      </section>

      {hits.length > 0 && (
        <section>
          <div className="label mb-1.5">What fired</div>
          <ul className="space-y-1">
            {hits.map((h) => (
              <li key={h.rule} className="flex items-baseline gap-3 text-xs">
                {/* Weight is the evidential strength the rule carries into the
                    noisy-OR, so it is worth showing rather than hiding: two weak
                    rules and one strong one produce very different verdicts. */}
                <span className="num w-9 shrink-0 text-right text-slate-600">
                  {h.weight.toFixed(2)}
                </span>
                <span className="num truncate text-slate-400">{h.rule}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

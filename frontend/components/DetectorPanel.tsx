"use client";

import { useState } from "react";
import clsx from "clsx";
import { motion, AnimatePresence } from "framer-motion";
import type { DetectorOutput } from "@/lib/api";
import { BehavioralEvidence } from "./BehavioralEvidence";
import { ScoreBar, scoreTone } from "./ui";

const TONE_TEXT = { pass: "text-pass", review: "text-review", reject: "text-reject" } as const;

/**
 * One detector, as a row rather than a card.
 *
 * Six bordered panels in a grid was six rectangles competing for attention when
 * the only thing worth comparing across them is the number. As a column of rows
 * the scores line up in one place and the shape of the packet reads at a glance:
 * one high number among five low ones is now visible without reading any labels.
 */
export function DetectorPanel({
  detector,
  heatmap,
  reviewAt = 0.4,
  rejectAt = 0.75,
}: {
  detector: DetectorOutput;
  heatmap?: string | null;
  reviewAt?: number;
  rejectAt?: number;
}) {
  const [open, setOpen] = useState(false);
  const d = detector;
  const inactive = d.status !== "ok";
  const tone = inactive ? "text-slate-600" : TONE_TEXT[scoreTone(d.score, reviewAt, rejectAt)];

  return (
    <div className="border-t border-edge/60">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="group w-full py-3 text-left transition-colors duration-150"
      >
        <div className="flex items-baseline gap-4">
          <span
            className={clsx(
              "min-w-0 flex-1 truncate text-sm transition-colors",
              inactive ? "text-slate-600" : "text-slate-300 group-hover:text-slate-100"
            )}
          >
            {d.label}
          </span>
          <span className={clsx("num shrink-0 text-base font-light", tone)}>
            {inactive ? <span className="text-xs">{d.status}</span> : d.score.toFixed(2)}
          </span>
        </div>

        <div className="mt-2.5">
          <ScoreBar score={d.score} status={d.status} reviewAt={reviewAt} rejectAt={rejectAt} />
        </div>

        {(d.reasons?.[0] || (inactive && d.detail)) && (
          <p className="mt-2.5 line-clamp-1 text-xs text-slate-600 group-hover:text-slate-500">
            {inactive ? d.detail : d.reasons[0]}
          </p>
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-4 pb-5 pt-1">
              <div className="num flex flex-wrap gap-x-5 gap-y-1 text-2xs text-slate-600">
                <span>confidence {(d.confidence * 100).toFixed(0)}%</span>
                <span>{d.latency_ms.toFixed(0)} ms</span>
              </div>

              {d.reasons?.length > 0 && (
                <ul className="space-y-1.5">
                  {d.reasons.map((r, i) => (
                    <li key={i} className="text-xs leading-relaxed text-slate-400">
                      {r}
                    </li>
                  ))}
                </ul>
              )}

              {heatmap && (
                <img
                  src={heatmap}
                  alt={`${d.label} saliency overlay`}
                  className="w-full max-w-sm rounded"
                />
              )}

              {/* Detector 6 measured time, not pixels, so it gets a reading
                  rather than the raw signal dump the image detectors fall back
                  to. See BehavioralEvidence for why the bands come from the
                  server. */}
              {d.name === "behavioral" && d.status === "ok" ? (
                <BehavioralEvidence detector={d} />
              ) : (
                <pre className="scroll-x num max-h-56 rounded bg-ink-900 p-3 text-2xs leading-relaxed text-slate-500">
                  {JSON.stringify(d.signals, null, 2)}
                </pre>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

const STAGES = [
  "Selfie deepfake",
  "ID forensics",
  "Liveness video",
  "Metadata / EXIF",
  "Face match",
  "Behavioral",
];

/** The detectors in flight. They genuinely run concurrently, so all six pulse
 *  at once rather than faking a sequential progress bar. */
export function PipelineRunning({ elapsed }: { elapsed: number }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="label">Running</span>
        <span className="num text-xs text-slate-600">{(elapsed / 1000).toFixed(1)}s</span>
      </div>
      <div className="mt-4 space-y-3">
        {STAGES.map((s, i) => (
          <div key={s} className="flex items-center gap-3">
            <span
              className="h-1 w-1 shrink-0 rounded-full bg-accent animate-sweep"
              style={{ animationDelay: `${i * 0.11}s` }}
            />
            <span className="text-sm text-slate-600">{s}</span>
            <span className="relative ml-auto h-px w-24 overflow-hidden bg-ink-800">
              <span
                className="absolute inset-y-0 w-1/3 bg-accent animate-shimmer"
                style={{ animationDelay: `${i * 0.11}s` }}
              />
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

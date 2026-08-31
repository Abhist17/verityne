"use client";

import { useState } from "react";
import clsx from "clsx";
import { motion, AnimatePresence } from "framer-motion";
import type { DetectorOutput } from "@/lib/api";
import { ScoreBar, scoreTone } from "./ui";

const ICONS: Record<string, React.ReactNode> = {
  selfie_deepfake: <path d="M12 3a4 4 0 110 8 4 4 0 010-8zM4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5" />,
  liveness_video: <path d="M3 6h13v12H3zM16 10l5-3v10l-5-3" />,
  id_forensics: <path d="M3 5h18v14H3zM7 9h4v5H7zM14 10h4M14 13h4" />,
  face_match: (
    <path d="M8 4a3 3 0 110 6 3 3 0 010-6zM2 19c0-3 2.7-5 6-5M16 4a3 3 0 110 6 3 3 0 010-6zM12 19c0-3 2.7-5 6-5M11 11l2 2" />
  ),
  metadata_exif: <path d="M4 4h16v16H4zM8 8h8M8 12h8M8 16h5" />,
};

function DetectorIcon({ name }: { name: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {ICONS[name] ?? <circle cx="12" cy="12" r="8" />}
    </svg>
  );
}

const TONE_TEXT = { pass: "text-pass", review: "text-review", reject: "text-reject" } as const;

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
    <div className={clsx("card overflow-hidden", inactive && "opacity-60")}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-2.5 p-3.5 text-left transition-colors duration-150 hover:bg-ink-850"
        aria-expanded={open}
      >
        <span className={clsx("mt-px rounded bg-ink-800 p-1.5 ring-1 ring-edge", tone)}>
          <DetectorIcon name={d.name} />
        </span>

        <span className="min-w-0 flex-1">
          <span className="flex items-baseline justify-between gap-3">
            <span className="truncate text-sm font-medium text-slate-200">{d.label}</span>
            <span className={clsx("num text-sm font-medium", tone)}>{inactive ? d.status : d.score.toFixed(2)}</span>
          </span>

          <span className="mt-2 block">
            <ScoreBar score={d.score} status={d.status} reviewAt={reviewAt} rejectAt={rejectAt} />
          </span>

          <span className="num mt-2 flex items-center gap-2 text-2xs text-slate-600">
            <span>conf {(d.confidence * 100).toFixed(0)}%</span>
            <span>·</span>
            <span>{d.latency_ms.toFixed(0)} ms</span>
            {heatmap && (
              <>
                <span>·</span>
                <span className="text-accent">heatmap</span>
              </>
            )}
            <span className="ml-auto">{open ? "hide" : "details"}</span>
          </span>

          {d.reasons?.[0] && <span className="mt-2 block text-xs leading-relaxed text-slate-400">{d.reasons[0]}</span>}
          {inactive && d.detail && <span className="num mt-1 block text-2xs text-slate-600">{d.detail}</span>}
        </span>
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }}
            className="overflow-hidden border-t border-edge bg-ink-950/50"
          >
            <div className="space-y-3 p-3.5">
              {d.reasons?.length > 1 && (
                <ul className="space-y-1.5">
                  {d.reasons.slice(1).map((r, i) => (
                    <li key={i} className="flex gap-2 text-xs leading-relaxed text-slate-400">
                      <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-slate-700" />
                      {r}
                    </li>
                  ))}
                </ul>
              )}
              {heatmap && (
                <div>
                  <div className="label mb-1.5">Saliency overlay</div>
                  <img src={heatmap} alt={`${d.label} heatmap`} className="w-full rounded border border-edge" />
                </div>
              )}
              <div>
                <div className="label mb-1.5">Raw signals</div>
                <pre className="scroll-x num max-h-64 rounded border border-edge bg-ink-950 p-2.5 text-2xs leading-relaxed text-slate-400">
                  {JSON.stringify(d.signals, null, 2)}
                </pre>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

const STAGES = [
  { name: "selfie_deepfake", label: "Selfie deepfake" },
  { name: "id_forensics", label: "ID forensics" },
  { name: "liveness_video", label: "Liveness video" },
  { name: "metadata_exif", label: "Metadata / EXIF" },
  { name: "face_match", label: "Face match" },
];

/** The five detectors in flight. They genuinely run concurrently, so this shows
 *  them all active at once rather than faking a sequential progress bar. */
export function PipelineRunning({ elapsed }: { elapsed: number }) {
  return (
    <div className="card-pad">
      <div className="flex items-center justify-between">
        <div className="label">Pipeline running</div>
        <div className="num text-xs text-slate-500">{(elapsed / 1000).toFixed(1)}s</div>
      </div>
      <div className="mt-3.5 space-y-2.5">
        {STAGES.map((s, i) => (
          <div key={s.name} className="flex items-center gap-2.5">
            <span
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent animate-sweep"
              style={{ animationDelay: `${i * 0.12}s` }}
            />
            <span className="text-xs text-slate-400">{s.label}</span>
            <span className="relative ml-auto h-0.5 w-28 overflow-hidden rounded-full bg-ink-750">
              <span
                className="absolute inset-y-0 w-1/3 rounded-full bg-accent animate-shimmer"
                style={{ animationDelay: `${i * 0.12}s` }}
              />
            </span>
          </div>
        ))}
        <div className="flex items-center gap-2.5 border-t border-edge pt-2.5">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-ink-700" />
          <span className="text-xs text-slate-600">Fusion + explanation</span>
        </div>
      </div>
    </div>
  );
}

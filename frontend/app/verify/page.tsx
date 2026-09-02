"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { motion } from "framer-motion";
import { api, type VerifyResponse } from "@/lib/api";
import { DropZone } from "@/components/DropZone";
import { LiveFaceMatch } from "@/components/LiveFaceMatch";
import { DetectorPanel, PipelineRunning } from "@/components/DetectorPanel";
import { ErrorBox, NumberedCard, PageHeader, ScoreMeter, SectionLabel, Spinner } from "@/components/ui";
import { TelemetryCollector } from "@/lib/telemetry";
import { Hint } from "@/components/Hint";

const DETECTOR_ORDER = [
  "selfie_deepfake",
  "liveness_video",
  "id_forensics",
  "face_match",
  "metadata_exif",
  "behavioral",
];

/**
 * The three shipped merchant policies, and what choosing one actually changes.
 *
 * These were three bare lowercase words in a segmented control with no heading,
 * which made them look like a debug toggle rather than the product's central
 * claim: the same six detectors, the same scores, and a different verdict
 * because a crypto exchange and a gig marketplace do not price fraud the same
 * way. The thresholds below are read from `backend/verityne/policy.yaml`, so
 * this list has to move when that file does.
 */
const POLICIES = [
  {
    id: "default",
    label: "Default",
    blurb:
      "The balanced baseline, taken from this system's own cost curve rather than picked by hand. Everything on the Metrics page is reported at this setting.",
    meta: "reject 0.80 · review 0.45",
  },
  {
    id: "crypto_exchange_01",
    label: "Strict",
    blurb:
      "A high-risk vertical, where one fraudulent account costs far more than turning away an honest applicant. Both thresholds tighten, so more packets are rejected and far more go to a human.",
    meta: "reject 0.60 · review 0.25 · fraud loss ₹2.5L",
  },
  {
    id: "gig_marketplace_02",
    label: "Lenient",
    blurb:
      "A low-value, high-volume marketplace, where friction costs more than the fraud does. The reject line moves up and the liveness clip stops being required at all.",
    meta: "reject 0.88 · review 0.45 · no liveness",
  },
];

/** What the six detectors read, shown while the right column is otherwise empty.
 *
 *  The idle state used to be one sentence in slate-700 on a near-black ground -
 *  invisible in practice, and it left two thirds of the page blank at exactly
 *  the moment someone is deciding whether this thing is serious. The detectors
 *  are the answer to that, and listing what each one actually looks at is more
 *  honest than a hero graphic.
 */
function WhatRuns() {
  const rows: [string, string][] = [
    ["Selfie deepfake", "A pretrained transformer and a fitted frequency head, voting"],
    ["ID forensics", "OCR, structural check digits, and compression-level analysis"],
    ["Liveness video", "Identity drift, pose jitter and splice discontinuity across frames"],
    ["Face match", "512-d embeddings, selfie against the portrait on the card"],
    ["Metadata / EXIF", "Generator tags, capture age, and whether detail matches resolution"],
    ["Behavioral", "How the form was filled - typing rhythm, pointer, device coherence"],
  ];
  return (
    <div>
      {/* The opening line carries the page, so it is set as a sentence rather
          than as body copy - one step up the scale, and short enough to be read
          before the eye moves to the cards. */}
      <p className="max-w-[46ch] text-lg leading-snug text-slate-200">
        Attach a packet, or score a fixture. Every verdict comes back with the
        evidence behind it.
      </p>

      <div className="mt-9 flex items-baseline gap-3">
        <span className="label shrink-0">Six detectors run concurrently</span>
        <span className="rule-soft min-w-0 flex-1" />
      </div>

      {/* Two columns, not six rows. The grid is what makes "six" legible as a
          quantity - a list of six is something you count, a 3x2 block is
          something you see. It collapses to one column under `wide` because at
          300px per card the second column starts hyphenating. */}
      <div className="mt-4 grid gap-2.5 sm:grid-cols-2">
        {rows.map(([name, what], i) => (
          <NumberedCard key={name} index={i + 1} title={name}>
            {what}
          </NumberedCard>
        ))}
      </div>

      <p className="mt-5 max-w-[62ch] text-xs leading-relaxed text-slate-500">
        The first five read the files. The sixth reads the person - and it is the only one
        whose adversary is not on a release cycle.
      </p>
    </div>
  );
}

export default function LiveVerifyPage() {
  const [selfie, setSelfie] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [idDoc, setIdDoc] = useState<File | null>(null);
  const [claimedName, setClaimedName] = useState("");
  const [merchantId, setMerchantId] = useState("default");

  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<any[]>([]);
  const [camera, setCamera] = useState(false);
  const timer = useRef<any>(null);

  // Detector 6 reads how this form was filled; created once so a remount does
  // not mint a new token and throw away the fill it had recorded.
  const telemetry = useRef<TelemetryCollector | null>(null);
  if (telemetry.current === null && typeof window !== "undefined") {
    telemetry.current = new TelemetryCollector();
  }

  useEffect(() => {
    api.gauntletManifest().then((m) => setSamples(m.items ?? [])).catch(() => setSamples([]));
    const t = telemetry.current;
    t?.start();
    return () => {
      t?.stop();
      if (timer.current) clearInterval(timer.current);
    };
  }, []);

  const startTimer = () => {
    const t0 = Date.now();
    setElapsed(0);
    timer.current = setInterval(() => setElapsed(Date.now() - t0), 80);
  };
  const stopTimer = () => timer.current && clearInterval(timer.current);

  const run = useCallback(async () => {
    if (!selfie && !idDoc && !video) return;
    setError(null);
    setResult(null);
    setRunning(true);
    startTimer();
    try {
      // Telemetry goes up first, against a token minted at form load: the files
      // can take seconds over a phone connection and the buffer should not die
      // with a failed upload.
      let token: string | null = null;
      const t = telemetry.current;
      if (t && t.eventCount > 0) {
        try {
          token = (await api.behavioral(t.snapshot(merchantId))).token;
        } catch {
          token = null;
        }
      }

      const form = new FormData();
      if (selfie) form.append("selfie", selfie);
      if (video) form.append("liveness_video", video);
      if (idDoc) form.append("id_document", idDoc);
      form.append("merchant_id", merchantId);
      if (claimedName) form.append("claimed_name", claimedName);
      if (token) form.append("behavioral_token", token);
      setResult(await api.verify(form));
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setRunning(false);
      stopTimer();
    }
  }, [selfie, video, idDoc, claimedName, merchantId]);

  const runSample = useCallback(async (id: string) => {
    setError(null);
    setResult(null);
    setRunning(true);
    startTimer();
    try {
      setResult(await api.rescore(id));
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setRunning(false);
      stopTimer();
    }
  }, []);

  const reviewAt = result?.policy?.min_risk_for_review ?? 0.4;
  const rejectAt = result?.policy?.min_risk_for_reject ?? 0.75;
  const linkage = result?.policy?.linkage;
  const hasInput = !!(selfie || idDoc || video);

  return (
    <div className="space-y-12">
      <PageHeader
        title="Verify"
        eyebrow="Live packet"
        actions={
          <div className="flex flex-col items-start gap-1.5 wide:items-end">
            <Hint
              align="right"
              title="Merchant policy"
              body="Who is asking. The detectors and the risk score do not change - the thresholds they are judged against do, so the same packet can pass for one merchant and go to review for another."
            >
              <span className="label cursor-help text-slate-500 underline decoration-edge-strong decoration-dotted underline-offset-4">
                Merchant policy
              </span>
            </Hint>
            <div className="segment" role="group" aria-label="Merchant policy">
              {POLICIES.map((p) => (
                <Hint key={p.id} align="right" title={p.label} body={p.blurb} meta={p.meta}>
                  <button data-active={merchantId === p.id} onClick={() => setMerchantId(p.id)}>
                    {p.label}
                  </button>
                </Hint>
              ))}
            </div>
          </div>
        }
      />

      <div className="grid gap-x-14 gap-y-12 md:grid-cols-[300px_minmax(0,1fr)]">
        {/* ---------------- input ---------------- */}
        <div className="space-y-6">
          <div className="space-y-2">
            <DropZone
              label="Selfie"
              hint="JPG / PNG"
              accept="image/*"
              file={selfie}
              onFile={setSelfie}
              className="h-[168px]"
            />
            <div className="grid grid-cols-2 gap-2">
              <DropZone
                label="ID document"
                hint="PAN / Aadhaar"
                accept="image/*"
                file={idDoc}
                onFile={setIdDoc}
                className="h-[104px]"
              />
              <DropZone
                label="Liveness"
                hint="MP4 / WebM"
                accept="video/*"
                file={video}
                onFile={setVideo}
                className="h-[104px]"
              />
            </div>
          </div>

          <div data-tele-field="claimed_name">
            <input
              value={claimedName}
              onChange={(e) => setClaimedName(e.target.value)}
              placeholder="Claimed name"
              className="input"
            />
          </div>

          <button
            onClick={run}
            disabled={running || !hasInput}
            className={clsx(
              "w-full py-2.5 text-sm font-medium transition-colors duration-150",
              // Accent when it is armed. This used to go white-on-black when
              // ready, which was the brightest thing on the page and therefore
              // read as the primary action - correct - but by a different rule
              // than every other primary action in the app. One rule: the thing
              // you are here to do is the accent, and nothing else is.
              hasInput && !running
                ? "bg-accent text-ink-1000 hover:bg-accent-soft"
                : "bg-ink-800 text-slate-600"
            )}
          >
            {running ? <Spinner label="Verifying" /> : "Verify"}
          </button>

          <p className="text-2xs leading-relaxed text-slate-700">
            Typing and pointer <em>timing</em> is recorded for behavioural analysis. Which keys you press is
            never recorded or sent.
          </p>

          {samples.length > 0 && (
            <div>
              <SectionLabel>Fixtures</SectionLabel>
              <div className="flex flex-wrap gap-1">
                {samples.slice(0, 10).map((s) => (
                  <button
                    key={s.submission_id}
                    onClick={() => runSample(s.submission_id)}
                    disabled={running}
                    title={s.attack_type ?? "genuine"}
                    className="num rounded px-1.5 py-1 text-2xs tracking-normal text-slate-600 transition-colors hover:bg-ink-800 hover:text-slate-300 disabled:opacity-30"
                  >
                    <span className={s.truth === "fake" ? "text-reject/70" : "text-pass/70"}>·</span>{" "}
                    {s.name}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Camera is opt-in and collapsed by default - it was a permanently
              open panel for a check most sessions never run. */}
          <div>
            <button
              onClick={() => setCamera((v) => !v)}
              className="label transition-colors hover:text-slate-400"
              aria-expanded={camera}
            >
              Live face match {camera ? "−" : "+"}
            </button>
            {camera && (
              <div className="mt-3">
                <LiveFaceMatch reference={idDoc} />
              </div>
            )}
          </div>
        </div>

        {/* ---------------- result ---------------- */}
        <div>
          {error && <ErrorBox error={error} />}
          {running && <PipelineRunning elapsed={elapsed} />}

          {!running && !result && !error && <WhatRuns />}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
              className="space-y-12"
            >
              <div>
                <ScoreMeter
                  score={result.final_score}
                  verdict={result.verdict}
                  reviewAt={reviewAt}
                  rejectAt={rejectAt}
                />
                <p className="mt-8 max-w-[64ch] text-sm leading-relaxed text-slate-400">
                  {result.explanation}
                </p>
                <div className="num mt-4 flex flex-wrap gap-x-5 gap-y-1 text-2xs tracking-normal text-slate-700">
                  <span>{result.latency_ms.toFixed(0)} ms</span>
                  <span>{result.fusion_model}</span>
                  {result.abstained && <span className="text-accent">abstained → human</span>}
                  {result.attack_pattern && result.attack_pattern !== "clean" && (
                    <span>{result.attack_pattern.replace(/_/g, " ")}</span>
                  )}
                  {result.generator_guess && <span>{result.generator_guess}</span>}
                </div>
              </div>

              {result.top_reasons.length > 0 && (
                <div>
                  <SectionLabel>Why</SectionLabel>
                  <ol className="space-y-3">
                    {result.top_reasons.map((r, i) => (
                      <li key={i} className="flex gap-4 text-sm leading-relaxed text-slate-300">
                        <span className="num shrink-0 text-2xs text-slate-700">
                          {String(i + 1).padStart(2, "0")}
                        </span>
                        <span className="max-w-[62ch]">{r}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              {linkage && (linkage.face_links?.length > 0 || linkage.asset_links?.length > 0) && (
                <div>
                  <SectionLabel
                    right={
                      linkage.asset_links?.some((l: any) => l.match === "exact")
                        ? undefined
                        : "similarity only - capped at review"
                    }
                  >
                    Linkage
                  </SectionLabel>
                  <div className="num space-y-1.5 text-xs text-slate-500">
                    {linkage.face_links?.map((l: any) => (
                      <div key={l.submission_id} className="flex flex-wrap items-baseline gap-3">
                        <span className="text-slate-700">face</span>
                        <span className="text-review">{(l.similarity * 100).toFixed(1)}%</span>
                        <span>{l.claimed_name ?? "unknown"}</span>
                        {l.name_differs && <span className="text-review">different name</span>}
                      </div>
                    ))}
                    {linkage.asset_links?.map((l: any) => (
                      <div key={l.submission_id} className="flex flex-wrap items-baseline gap-3">
                        <span className="text-slate-700">{l.kind}</span>
                        <span className={l.match === "exact" ? "text-reject" : "text-review"}>
                          {l.match === "exact" ? "byte-identical" : `${l.hamming} bits apart`}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {Object.keys(result.heatmaps).length > 0 && (
                <div>
                  <SectionLabel>Where the model looked</SectionLabel>
                  <div className="grid gap-3 sm:grid-cols-3">
                    {Object.entries(result.heatmaps).map(([k, url]) => (
                      <figure key={k}>
                        <img src={url} alt={`${k} heatmap`} className="w-full rounded" />
                        <figcaption className="mt-1.5 text-2xs text-slate-700">
                          {k.replace(/_/g, " ")}
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <SectionLabel>Detectors</SectionLabel>
                <div>
                  {DETECTOR_ORDER.filter((n) => result.detector_breakdown[n]).map((n) => (
                    <DetectorPanel
                      key={n}
                      detector={result.detector_breakdown[n]}
                      heatmap={result.heatmaps[n]}
                      reviewAt={reviewAt}
                      rejectAt={rejectAt}
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

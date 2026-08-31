"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { motion } from "framer-motion";
import { api, type VerifyResponse } from "@/lib/api";
import { DropZone } from "@/components/DropZone";
import { LiveFaceMatch } from "@/components/LiveFaceMatch";
import { DetectorPanel, PipelineRunning } from "@/components/DetectorPanel";
import { ErrorBox, PageHeader, ScoreDial, Spinner, VerdictBadge } from "@/components/ui";
import { TelemetryCollector } from "@/lib/telemetry";

const DETECTOR_ORDER = [
  "selfie_deepfake",
  "liveness_video",
  "id_forensics",
  "face_match",
  "metadata_exif",
  "behavioral",
];

const POLICIES = [
  { id: "default", label: "default" },
  { id: "crypto_exchange_01", label: "crypto (strict)" },
  { id: "gig_marketplace_02", label: "gig (lenient)" },
];

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
  const [events, setEvents] = useState(0);
  const timer = useRef<any>(null);

  // Detector 6 reads how this form was filled. The collector starts on mount so
  // the buffer covers the whole session, and it is deliberately created once —
  // remounting would mint a new token and throw away the fill it had recorded.
  const telemetry = useRef<TelemetryCollector | null>(null);
  if (telemetry.current === null && typeof window !== "undefined") {
    telemetry.current = new TelemetryCollector();
  }

  useEffect(() => {
    api.gauntletManifest().then((m) => setSamples(m.items ?? [])).catch(() => setSamples([]));
    const t = telemetry.current;
    t?.start();
    const tick = setInterval(() => setEvents(t?.eventCount ?? 0), 700);
    return () => {
      clearInterval(tick);
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
    if (!selfie && !idDoc && !video) {
      setError("Attach at least one asset — a selfie, an ID document or a liveness clip.");
      return;
    }
    setError(null);
    setResult(null);
    setRunning(true);
    startTimer();
    try {
      // Telemetry goes up first, against a token minted at form load. The files
      // can take seconds over a phone connection, and holding the buffer until
      // that multipart body is assembled would lose it whenever an upload fails.
      let token: string | null = null;
      const t = telemetry.current;
      if (t && t.eventCount > 0) {
        try {
          token = (await api.behavioral(t.snapshot(merchantId))).token;
        } catch {
          // Best-effort evidence: a failed telemetry post must not cost the
          // applicant their verification. The detector skips instead, which is
          // the same outcome as a server-to-server caller that never had a form.
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
    <div className="space-y-5">
      <PageHeader
        title="Verify"
        actions={
          <div className="segment" role="group" aria-label="Merchant policy">
            {POLICIES.map((p) => (
              <button key={p.id} data-active={merchantId === p.id} onClick={() => setMerchantId(p.id)}>
                {p.label}
              </button>
            ))}
          </div>
        }
      >
        Drop a KYC packet and watch all six detectors run. Every verdict comes back with the evidence behind
        it — no score without a reason.
      </PageHeader>

      {/* The split has to survive a ~1000px laptop viewport, which is below
          Tailwind's `lg`. Splitting at `md` with a fixed left column is what
          keeps the result panel beside the inputs instead of a screen below. */}
      <div className="grid gap-5 md:grid-cols-[330px_minmax(0,1fr)] xl:grid-cols-[380px_minmax(0,1fr)]">
        {/* ---------- input column ---------- */}
        <div className="space-y-3">
          <DropZone
            label="Selfie"
            hint="JPG / PNG — the live capture"
            accept="image/*"
            file={selfie}
            onFile={setSelfie}
            className="h-[150px]"
          />
          <div className="grid grid-cols-2 gap-3">
            <DropZone
              label="ID document"
              hint="PAN / Aadhaar"
              accept="image/*"
              file={idDoc}
              onFile={setIdDoc}
              className="h-[110px]"
            />
            <DropZone
              label="Liveness clip"
              hint="MP4 / WebM"
              accept="video/*"
              file={video}
              onFile={setVideo}
              className="h-[110px]"
            />
          </div>

          <div data-tele-field="claimed_name">
            <input
              value={claimedName}
              onChange={(e) => setClaimedName(e.target.value)}
              placeholder="Claimed name (optional — enables PAN surname check)"
              className="input"
            />
          </div>

          <button onClick={run} disabled={running || !hasInput} className="btn-primary w-full py-2">
            {running ? <Spinner label="Verifying…" /> : "Run verification"}
          </button>

          {/* Detector 6 is the only one whose input the applicant produces just
              by being here, so it is the only one that has to say so. */}
          <div className="flex items-start gap-2 px-0.5 text-2xs leading-relaxed text-slate-600">
            <span
              className={clsx("mt-1 h-1.5 w-1.5 shrink-0 rounded-full", events > 0 ? "bg-pass" : "bg-ink-700")}
              aria-hidden
            />
            <span>
              {events > 0 ? (
                <>
                  <span className="num text-slate-500">{events}</span> interaction
                  {events === 1 ? " event" : " events"} captured for behavioural analysis — timing only.
                  Which keys you pressed is never recorded or sent.
                </>
              ) : (
                <>
                  Behavioural analysis reads typing and pointer <em>timing</em>, never content. Type in the
                  field above to start the capture.
                </>
              )}
            </span>
          </div>

          {samples.length > 0 && (
            <div className="card-pad">
              <div className="label">Or score a loaded fixture</div>
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {samples.slice(0, 10).map((s) => (
                  <button
                    key={s.submission_id}
                    onClick={() => runSample(s.submission_id)}
                    disabled={running}
                    title={s.attack_type ?? "genuine"}
                    className="flex items-center gap-1.5 rounded border border-edge bg-ink-850 px-1.5 py-1 text-2xs text-slate-400 transition-colors hover:border-edge-strong hover:text-slate-200 disabled:opacity-40"
                  >
                    <span className={s.truth === "fake" ? "text-reject" : "text-pass"}>
                      {s.truth === "fake" ? "▲" : "●"}
                    </span>
                    {s.name}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-2xs leading-relaxed text-slate-600">
                Ground truth is shown only because these are eval fixtures.
              </p>
            </div>
          )}

          {/* Live capture. Separate from the pipeline above on purpose: this
              answers only "is this the person on the card", in real time, and
              stores nothing. The full verdict still comes from /verify. */}
          <div className="card-pad">
            <div className="label">Live face match</div>
            <p className="mb-3 mt-1 text-2xs leading-relaxed text-slate-500">
              Match your camera against the uploaded ID portrait. Identity alone — this does not produce a
              verdict.
            </p>
            <LiveFaceMatch reference={idDoc} />
          </div>
        </div>

        {/* ---------- result column ---------- */}
        <div className="space-y-4">
          {error && <ErrorBox error={error} />}
          {running && <PipelineRunning elapsed={elapsed} />}

          {!running && !result && !error && (
            <div className="card flex min-h-[420px] flex-col items-center justify-center gap-2.5 p-8 text-center">
              <svg
                viewBox="0 0 24 24"
                className="h-7 w-7 text-slate-700"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.3"
              >
                <path d="M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6l7-3z" strokeLinejoin="round" />
              </svg>
              <div className="text-sm text-slate-400">Nothing scored yet</div>
              <div className="max-w-[46ch] text-xs leading-relaxed text-slate-600">
                Attach a packet on the left, or score one of the loaded fixtures to see the full pipeline
                output — verdict, reasons, heatmaps and every detector&apos;s raw signals.
              </div>
            </div>
          )}

          {result && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
              className="space-y-4"
            >
              <div className="card-pad">
                <div className="flex flex-wrap items-center gap-5">
                  <ScoreDial score={result.final_score} verdict={result.verdict} reviewAt={reviewAt} rejectAt={rejectAt} />
                  <div className="min-w-[240px] flex-1 space-y-2.5">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <VerdictBadge verdict={result.verdict} size="lg" />
                      {result.abstained && (
                        <span className="chip bg-accent/10 text-accent ring-1 ring-accent/30">abstained → human</span>
                      )}
                      {result.attack_pattern && result.attack_pattern !== "clean" && (
                        <span className="chip bg-ink-800 text-slate-300 ring-1 ring-edge">
                          {result.attack_pattern.replace(/_/g, " ")}
                        </span>
                      )}
                      {result.generator_guess && (
                        <span className="chip bg-ink-800 text-slate-400 ring-1 ring-edge">{result.generator_guess}</span>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-slate-300">{result.explanation}</p>
                    <div className="num flex flex-wrap gap-x-4 gap-y-1 text-2xs text-slate-600">
                      <span>{result.latency_ms.toFixed(0)} ms</span>
                      <span>{result.fusion_model}</span>
                      <span>
                        review ≥ {reviewAt} · reject ≥ {rejectAt}
                      </span>
                      <span className="truncate">{result.submission_id}</span>
                    </div>
                  </div>
                </div>
              </div>

              {result.top_reasons.length > 0 && (
                <div className="card-pad">
                  <div className="label">Top reasons</div>
                  <ol className="mt-2.5 space-y-2">
                    {result.top_reasons.map((r, i) => (
                      <li key={i} className="flex gap-2.5 text-sm leading-relaxed text-slate-300">
                        <span className="num mt-px flex h-4 w-4 shrink-0 items-center justify-center rounded bg-ink-800 text-2xs text-slate-500 ring-1 ring-edge">
                          {i + 1}
                        </span>
                        {r}
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              {linkage && (linkage.face_links?.length > 0 || linkage.asset_links?.length > 0) && (
                <div className="card-pad border-review/30">
                  <div className="label text-review">Cross-submission linkage</div>
                  <div className="num mt-2 space-y-1.5 text-xs text-slate-300">
                    {linkage.face_links?.map((l: any) => (
                      <div key={l.submission_id} className="flex flex-wrap items-center gap-2">
                        <span className="text-slate-600">face</span>
                        <span className="text-review">{(l.similarity * 100).toFixed(1)}%</span>
                        <span>→ {l.claimed_name ?? "unknown"}</span>
                        <span className="text-slate-600">@ {l.merchant_id}</span>
                        {l.name_differs && <span className="chip bg-review/10 text-review">different name</span>}
                      </div>
                    ))}
                    {linkage.asset_links?.map((l: any) => (
                      <div key={l.submission_id} className="flex items-center gap-2">
                        <span className="text-slate-600">asset</span>
                        <span className={l.match === "exact" ? "text-reject" : "text-review"}>
                          {l.match === "exact" ? "byte-identical" : `${l.hamming} bits apart`}
                        </span>
                        <span className="text-slate-600">({l.kind})</span>
                      </div>
                    ))}
                  </div>
                  {/* A similarity hit is capped below the reject line on purpose;
                      see linkage.SAME_PERSON. Saying so here stops a reviewer
                      reading a face link as a finding. */}
                  {!linkage.asset_links?.some((l: any) => l.match === "exact") && (
                    <p className="mt-2.5 text-2xs leading-relaxed text-slate-600">
                      Face and near-hash links are similarity claims, not identity ones. They cap at review and
                      cannot reject on their own.
                    </p>
                  )}
                </div>
              )}

              {Object.keys(result.heatmaps).length > 0 && (
                <div className="card-pad">
                  <div className="label">Where the model looked</div>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                    {Object.entries(result.heatmaps).map(([k, url]) => (
                      <figure key={k} className="space-y-1.5">
                        <img src={url} alt={`${k} heatmap`} className="w-full rounded border border-edge" />
                        <figcaption className="text-2xs text-slate-500">{k.replace(/_/g, " ")}</figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <div className="label mb-2">Detector breakdown</div>
                <div className="grid gap-3 xl:grid-cols-2">
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

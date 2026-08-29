"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { api, fmtPct, type VerifyResponse } from "@/lib/api";
import { DropZone } from "@/components/DropZone";
import { DetectorPanel, PipelineRunning } from "@/components/DetectorPanel";
import { ErrorBox, ScoreDial, Spinner, VerdictBadge } from "@/components/ui";

const DETECTOR_ORDER = ["selfie_deepfake", "liveness_video", "id_forensics", "face_match", "metadata_exif"];

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
  const timer = useRef<any>(null);

  useEffect(() => {
    api.gauntletManifest().then((m) => setSamples(m.items ?? [])).catch(() => setSamples([]));
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
      const form = new FormData();
      if (selfie) form.append("selfie", selfie);
      if (video) form.append("liveness_video", video);
      if (idDoc) form.append("id_document", idDoc);
      form.append("merchant_id", merchantId);
      if (claimedName) form.append("claimed_name", claimedName);
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
      const res = await fetch(`/api/submissions/${id}/rescore`, {
        method: "POST",
        headers: { "X-API-Key": process.env.NEXT_PUBLIC_API_KEY || "verityne-demo-key" },
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      setResult(await res.json());
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

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-slate-100">Live Verify</h1>
          <p className="mt-1 max-w-2xl text-sm leading-relaxed text-slate-400">
            Drop a KYC packet and watch all five detectors run. Every verdict comes back with the evidence
            behind it — no score without a reason.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="label">Merchant policy</label>
          <select
            value={merchantId}
            onChange={(e) => setMerchantId(e.target.value)}
            className="rounded-lg border border-edge bg-ink-800 px-3 py-1.5 text-sm text-slate-200"
          >
            <option value="default">default</option>
            <option value="crypto_exchange_01">crypto_exchange_01 (strict)</option>
            <option value="gig_marketplace_02">gig_marketplace_02 (lenient)</option>
          </select>
        </div>
      </header>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,420px)_1fr]">
        {/* ---------- input column ---------- */}
        <div className="space-y-4">
          <div className="grid gap-3">
            <DropZone label="Selfie" hint="JPG / PNG — the live capture" accept="image/*" file={selfie} onFile={setSelfie} />
            <div className="grid grid-cols-2 gap-3">
              <DropZone label="ID document" hint="PAN / Aadhaar" accept="image/*" file={idDoc} onFile={setIdDoc} />
              <DropZone label="Liveness clip" hint="MP4 / WebM" accept="video/*" file={video} onFile={setVideo} />
            </div>
          </div>

          <input
            value={claimedName}
            onChange={(e) => setClaimedName(e.target.value)}
            placeholder="Claimed name (optional — enables PAN surname validation)"
            className="w-full rounded-lg border border-edge bg-ink-900 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-accent/60 focus:outline-none"
          />

          <button onClick={run} disabled={running} className="btn-primary w-full py-2.5">
            {running ? <Spinner label="Verifying…" /> : "Run verification"}
          </button>

          {samples.length > 0 && (
            <div className="card-pad">
              <div className="label">Or score a loaded fixture</div>
              <div className="mt-2.5 flex flex-wrap gap-1.5">
                {samples.slice(0, 10).map((s) => (
                  <button
                    key={s.submission_id}
                    onClick={() => runSample(s.submission_id)}
                    disabled={running}
                    className="rounded-md border border-edge bg-ink-800 px-2 py-1 text-[11px] text-slate-400 transition hover:border-accent/50 hover:text-slate-200 disabled:opacity-40"
                    title={s.attack_type ?? "genuine"}
                  >
                    {s.truth === "fake" ? "▲" : "●"} {s.name}
                  </button>
                ))}
              </div>
              <p className="mt-2 text-[11px] leading-relaxed text-slate-600">
                ▲ fake · ● genuine — ground truth is shown here only because these are eval fixtures.
              </p>
            </div>
          )}
        </div>

        {/* ---------- result column ---------- */}
        <div className="space-y-4">
          {error && <ErrorBox error={error} />}
          {running && <PipelineRunning elapsed={elapsed} />}

          {!running && !result && !error && (
            <div className="card flex h-full min-h-[380px] flex-col items-center justify-center gap-3 p-8 text-center">
              <svg viewBox="0 0 24 24" className="h-8 w-8 text-slate-700" fill="none" stroke="currentColor" strokeWidth="1.4">
                <path d="M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6l7-3z" strokeLinejoin="round" />
              </svg>
              <div className="text-sm text-slate-400">No submission scored yet</div>
              <div className="max-w-sm text-xs leading-relaxed text-slate-600">
                Attach a packet on the left, or click one of the loaded fixtures to see the full pipeline output.
              </div>
            </div>
          )}

          {result && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }} className="space-y-4">
              <div className="card-pad">
                <div className="flex flex-wrap items-center gap-6">
                  <ScoreDial score={result.final_score} verdict={result.verdict} reviewAt={reviewAt} rejectAt={rejectAt} />
                  <div className="min-w-[260px] flex-1 space-y-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <VerdictBadge verdict={result.verdict} size="lg" />
                      {result.abstained && (
                        <span className="chip bg-accent/12 text-accent ring-1 ring-accent/30">abstained → human</span>
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
                    <div className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-slate-500">
                      <span>{result.latency_ms.toFixed(0)} ms end-to-end</span>
                      <span>fusion: {result.fusion_model}</span>
                      <span>review ≥ {reviewAt} · reject ≥ {rejectAt}</span>
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
                      <li key={i} className="flex gap-3 text-sm leading-relaxed text-slate-300">
                        <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-ink-800 font-mono text-[11px] text-slate-400 ring-1 ring-edge">
                          {i + 1}
                        </span>
                        {r}
                      </li>
                    ))}
                  </ol>
                </div>
              )}

              {linkage && (linkage.face_links?.length > 0 || linkage.asset_links?.length > 0) && (
                <div className="card-pad border-reject/40">
                  <div className="label text-reject">Cross-submission linkage</div>
                  <div className="mt-2 space-y-1.5 text-xs text-slate-300">
                    {linkage.face_links?.map((l: any) => (
                      <div key={l.submission_id} className="flex flex-wrap items-center gap-2 font-mono">
                        <span className="text-slate-500">face</span>
                        <span className="text-reject">{(l.similarity * 100).toFixed(1)}%</span>
                        <span>→ {l.claimed_name ?? "unknown"}</span>
                        <span className="text-slate-500">@ {l.merchant_id}</span>
                        {l.name_differs && <span className="chip bg-reject/15 text-reject">different name</span>}
                      </div>
                    ))}
                    {linkage.asset_links?.map((l: any) => (
                      <div key={l.submission_id} className="flex items-center gap-2 font-mono">
                        <span className="text-slate-500">asset</span>
                        <span className="text-reject">{l.hamming} bits apart</span>
                        <span className="text-slate-500">({l.kind})</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {Object.keys(result.heatmaps).length > 0 && (
                <div className="card-pad">
                  <div className="label">Where the model looked</div>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                    {Object.entries(result.heatmaps).map(([k, url]) => (
                      <figure key={k} className="space-y-1.5">
                        <img src={url} alt={`${k} heatmap`} className="w-full rounded-lg border border-edge" />
                        <figcaption className="text-[11px] text-slate-500">{k.replace(/_/g, " ")}</figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <div className="label mb-2.5">Detector breakdown</div>
                <div className="grid gap-3 md:grid-cols-2">
                  {DETECTOR_ORDER.filter((n) => result.detector_breakdown[n]).map((n) => (
                    <DetectorPanel
                      key={n}
                      detector={result.detector_breakdown[n]}
                      heatmap={result.heatmaps[n]}
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

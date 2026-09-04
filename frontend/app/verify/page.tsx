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

/**
 * One heatmap tile, which may have no image behind it.
 *
 * A deployment can hold the verdict without holding the picture: the read-only
 * build ships the database and leaves the overlays behind. A broken <img> then
 * renders as the browser's torn-page icon with the alt text spilling out of it,
 * which reads as a bug rather than an absence.
 *
 * The placeholder is swapped in by state, not by the `hidden` attribute. That
 * was the first attempt and it silently did nothing: `hidden` sets
 * `display: none` at the lowest precedence, and any Tailwind display utility on
 * the same element - `grid`, here - simply wins. Every tile showed "overlay not
 * deployed" underneath a heatmap that had loaded perfectly.
 */
function Overlay({ src, label }: { src: string; label: string }) {
  const [failed, setFailed] = useState(false);
  return (
    <figure>
      {failed ? (
        <div className="grid aspect-[4/3] w-full place-items-center rounded border border-edge bg-ink-900 text-2xs text-slate-600">
          overlay not deployed
        </div>
      ) : (
        <img
          src={src}
          alt={`${label} heatmap`}
          className="aspect-[4/3] w-full rounded object-cover"
          onError={() => setFailed(true)}
        />
      )}
      <figcaption className="mt-1.5 text-2xs text-slate-700">{label}</figcaption>
    </figure>
  );
}

/**
 * One stored packet, as something that obviously wants to be clicked.
 *
 * These were ten monospace filenames at slate-600, wrapped inline with a
 * coloured interpunct in front of each - the same value as the page grid behind
 * them, which made the fastest route to seeing this system work the quietest
 * thing in the column. Anyone arriving without files of their own scrolled
 * straight past.
 *
 * A row carries the three things that decide whether you click it: what the
 * packet is, whether it is genuine or fraudulent before you score it, and the
 * fact that clicking scores it. The thumbnail is the strongest of the three and
 * the least reliable - a read-only build ships the verdicts without the media -
 * so it degrades to a tinted chip rather than to a torn-image icon.
 */
function FixtureRow({
  item,
  active,
  busy,
  onRun,
}: {
  item: any;
  active: boolean;
  busy: boolean;
  onRun: () => void;
}) {
  const [noThumb, setNoThumb] = useState(false);
  const fake = item.truth === "fake";
  const kind = fake ? (item.attack_type ? item.attack_type.replace(/_/g, " ") : "fraudulent") : "genuine";

  return (
    <button
      onClick={onRun}
      disabled={busy}
      aria-label={`Score fixture ${item.name} (${kind})`}
      className={clsx(
        "group flex w-full items-center gap-2.5 rounded-md border px-2 py-1.5 text-left transition-colors duration-150",
        "focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-accent",
        active
          ? "border-accent/60 bg-accent/[0.07]"
          : "border-edge bg-ink-900 hover:border-edge-strong hover:bg-ink-850",
        busy && !active && "opacity-40"
      )}
    >
      {item.thumb_url && !noThumb ? (
        <img
          src={item.thumb_url}
          alt=""
          onError={() => setNoThumb(true)}
          className="h-8 w-8 shrink-0 rounded object-cover ring-1 ring-edge"
        />
      ) : (
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded bg-ink-850 ring-1 ring-edge">
          <span className={clsx("h-1.5 w-1.5 rounded-full", fake ? "bg-reject" : "bg-pass")} />
        </span>
      )}

      <span className="min-w-0 flex-1">
        <span className="num block truncate text-xs tracking-normal text-slate-200">{item.name}</span>
        <span className={clsx("block truncate text-2xs", fake ? "text-reject/90" : "text-pass/90")}>{kind}</span>
      </span>

      {active && busy ? (
        <span
          className="h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-[1.5px] border-accent/25 border-t-accent"
          aria-hidden
        />
      ) : (
        <span className="label shrink-0 text-slate-700 transition-colors duration-150 group-hover:text-accent">
          score
        </span>
      )}
    </button>
  );
}

/** The disclosure header for the webcam panel. An outline camera, drawn rather
 *  than imported, because this is the only icon in the column. */
function CameraGlyph({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      <path d="M3 8.5A1.5 1.5 0 014.5 7h2.2l1.1-1.8A1 1 0 019 4.7h6a1 1 0 01.85.5L17 7h2.5A1.5 1.5 0 0121 8.5v9a1.5 1.5 0 01-1.5 1.5h-15A1.5 1.5 0 013 17.5v-9z" />
      <circle cx="12" cy="12.7" r="3.1" />
    </svg>
  );
}

export default function LiveVerifyPage() {
  const [selfie, setSelfie] = useState<File | null>(null);
  const [video, setVideo] = useState<File | null>(null);
  const [idDoc, setIdDoc] = useState<File | null>(null);
  const [claimedName, setClaimedName] = useState("");
  const [merchantId, setMerchantId] = useState("default");
  /* Whether this deployment can score at all, and whether what is on screen was
     scored just now or read back from the database. A read-only build ships
     without the models so it can run in 71 MiB; it still holds every verdict it
     ever computed. `null` means we have not heard from /health yet. */
  const [canScore, setCanScore] = useState<boolean | null>(null);
  const [stored, setStored] = useState(false);

  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<VerifyResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [samples, setSamples] = useState<any[]>([]);
  const [camera, setCamera] = useState(false);
  /** Which fixture the running pipeline belongs to, so the row you clicked is
   *  the row that shows the spinner. */
  const [activeSample, setActiveSample] = useState<string | null>(null);
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
    setActiveSample(null);
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

  useEffect(() => {
    api.health()
      .then((h) => setCanScore(h?.scoring_available !== false))
      .catch(() => setCanScore(null));
  }, []);

  /* A fixture, either scored live or read back.
   *
   * Where the models are present this rescores, because watching the pipeline
   * actually run is the point of the page. Where they are not, the stored
   * verdict is the same evidence - same score, same reasons, same detector
   * breakdown, same heatmaps - and the panel says so rather than passing a
   * database read off as a fresh computation. */
  const runSample = useCallback(async (id: string) => {
    setError(null);
    setResult(null);
    setStored(false);
    setActiveSample(id);
    setRunning(true);
    startTimer();
    try {
      if (canScore === false) {
        const v = await api.storedVerdict(id);
        if (!v) throw new Error("This fixture has no stored verdict to show.");
        setResult(v);
        setStored(true);
      } else {
        setResult(await api.rescore(id));
      }
    } catch (e: any) {
      setError(e.message ?? String(e));
    } finally {
      setRunning(false);
      stopTimer();
    }
  }, [canScore]);

  const reviewAt = result?.policy?.min_risk_for_review ?? 0.4;
  const rejectAt = result?.policy?.min_risk_for_reject ?? 0.75;
  const linkage = result?.policy?.linkage;
  const hasInput = !!(selfie || idDoc || video);
  const shownSamples = samples.slice(0, 10);
  const fraudCount = shownSamples.filter((s) => s.truth === "fake").length;
  const genuineCount = shownSamples.length - fraudCount;

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
            disabled={running || !hasInput || canScore === false}
            className={clsx(
              "w-full py-2.5 text-sm font-medium transition-colors duration-150",
              // Accent when it is armed. This used to go white-on-black when
              // ready, which was the brightest thing on the page and therefore
              // read as the primary action - correct - but by a different rule
              // than every other primary action in the app. One rule: the thing
              // you are here to do is the accent, and nothing else is.
              hasInput && !running && canScore !== false
                ? "bg-accent text-ink-1000 hover:bg-accent-soft"
                : "bg-ink-800 text-slate-600"
            )}
          >
            {running ? <Spinner label="Verifying" /> : "Verify"}
          </button>

          {canScore === false && (
            <p className="border border-edge bg-ink-900 px-2.5 py-2 text-2xs leading-relaxed text-slate-500">
              <span className="text-slate-300">This deployment reads, it does not score.</span>{" "}
              It runs without the detector models, which is what lets it fit a free tier - so a new
              packet cannot be scored here. The fixtures below still open their full stored verdict:
              same score, same reasons, same per-detector breakdown. Run it locally for live scoring.
            </p>
          )}

          <p className="text-2xs leading-relaxed text-slate-700">
            Typing and pointer <em>timing</em> is recorded for behavioural analysis. Which keys you press is
            never recorded or sent.
          </p>

          {shownSamples.length > 0 && (
            <div>
              <SectionLabel right={`${genuineCount} genuine · ${fraudCount} fraud`}>
                Stored packets
              </SectionLabel>
              <p className="mb-2.5 max-w-[42ch] text-2xs leading-relaxed text-slate-500">
                {canScore === false
                  ? "Nothing to upload? Click one and its full stored verdict opens on the right - same score, same reasons, same detector breakdown."
                  : "Nothing to upload? Click one. It goes through the same six detectors, live, and the verdict lands on the right in about four seconds."}
              </p>
              <div className="space-y-1.5">
                {shownSamples.map((s) => (
                  <FixtureRow
                    key={s.submission_id}
                    item={s}
                    active={activeSample === s.submission_id}
                    busy={running}
                    onRun={() => runSample(s.submission_id)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Camera is opt-in and collapsed by default - it was a permanently
              open panel for a check most sessions never run.
             
              Collapsed is not the same as hidden, and it had become the same
              thing: a 12px uppercase micro-label with a plus after it, in the
              same slate as the caption under the file slots, for the one part of
              this page that runs a model against your own face in real time. It
              is a panel now - an edge, a name at reading size, and a line saying
              what it needs before it can do anything. */}
          <div
            className={clsx(
              "rounded-md border bg-ink-900 transition-colors duration-200",
              camera ? "border-edge-strong" : "border-edge"
            )}
          >
            <button
              onClick={() => setCamera((v) => !v)}
              aria-expanded={camera}
              className="flex w-full items-center gap-2.5 rounded-md px-3 py-2.5 text-left transition-colors duration-150 hover:bg-ink-850 focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-accent"
            >
              <CameraGlyph
                className={clsx("h-4 w-4 shrink-0 transition-colors", camera ? "text-accent" : "text-slate-500")}
              />
              <span className="min-w-0 flex-1">
                <span className="block text-xs font-medium text-slate-200">Live face match</span>
                <span className="block truncate text-2xs text-slate-500">
                  {idDoc
                    ? "Webcam against that ID's portrait, live"
                    : "Needs an ID document above"}
                </span>
              </span>
              <span className="num shrink-0 text-sm leading-none text-slate-500">{camera ? "−" : "+"}</span>
            </button>
            {camera && (
              <div className="border-t border-edge p-3">
                <LiveFaceMatch reference={idDoc} onCapture={setSelfie} />
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
                  {/* Never let a database read wear the costume of a fresh run:
                      the latency below belongs to whenever this was actually
                      scored, and saying so is the whole point of the page. */}
                  {stored ? (
                    <span className="text-accent">stored verdict, not re-scored</span>
                  ) : (
                    <span>{result.latency_ms.toFixed(0)} ms</span>
                  )}
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
                        {/* "no name" is a third state, not a quiet form of
                            agreement: without a name on both sides nothing about
                            the identities was compared, and the row should not
                            read like a clean match. */}
                        {l.name_differs ? (
                          <span className="text-review">different name</span>
                        ) : l.name_comparable ? (
                          <span className="text-slate-600">same name</span>
                        ) : (
                          <span className="text-slate-600">name not compared</span>
                        )}
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
                      <Overlay key={k} src={url as string} label={k.replace(/_/g, " ")} />
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

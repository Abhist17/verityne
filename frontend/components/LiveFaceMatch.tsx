"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import clsx from "clsx";
import { api, FaceMatchResult } from "@/lib/api";

/**
 * Webcam capture, matched live against a document portrait.
 *
 * The camera stream never leaves the browser. Each poll grabs one frame from a
 * hidden <canvas>, posts it as a JPEG, and drops it - the endpoint holds nothing
 * on disk either. What the user sees is the similarity score moving in real time
 * against a threshold that was fitted on LFW rather than picked by hand.
 */
export function LiveFaceMatch({
  reference,
  onCapture,
}: {
  reference: File | null;
  /** Hand the current frame back as a file, to be used as the selfie. */
  onCapture?: (f: File) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const busyRef = useRef(false);

  const [on, setOn] = useState(false);
  const [live, setLive] = useState(false);
  const [result, setResult] = useState<FaceMatchResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [band, setBand] = useState<{ low: number; high: number; fitted_on: string } | null>(null);
  const [captured, setCaptured] = useState(false);
  /* How many frames have actually been scored this session.
   *
   * The panel had no way to say it was working. A live match against a moving
   * face produces a number that barely moves - 0.612, 0.611, 0.613 - so a
   * running loop and a frozen one look identical, and the first question anyone
   * asked of this panel was whether it was doing anything at all. A count that
   * ticks answers that without them having to trust the similarity. */
  const [checks, setChecks] = useState(0);

  useEffect(() => {
    api.faceMatchThreshold().then(setBand).catch(() => {});
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setOn(false);
    setLive(false);
    setChecks(0);
    setResult(null);
  }, []);

  useEffect(() => stop, [stop]);

  async function start() {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });
      streamRef.current = s;
      if (videoRef.current) {
        videoRef.current.srcObject = s;
        await videoRef.current.play();
      }
      setOn(true);
    } catch (e: any) {
      // A denied permission and an absent camera are different problems, and the
      // fix for each is different, so they are not collapsed into one message.
      setError(
        e?.name === "NotAllowedError"
          ? "Camera permission denied. Allow it in the browser's address bar, then start again."
          : e?.name === "NotFoundError"
          ? "No camera found. Plug one in, or upload a selfie file instead."
          : `Could not open the camera: ${e?.message || e}`
      );
    }
  }

  const grabFrame = useCallback((): Promise<Blob | null> => {
    const v = videoRef.current;
    const c = canvasRef.current;
    if (!v || !c || !v.videoWidth) return Promise.resolve(null);
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0, c.width, c.height);
    return new Promise((res) => c.toBlob((b) => res(b), "image/jpeg", 0.9));
  }, []);

  const matchOnce = useCallback(async () => {
    if (!reference || busyRef.current) return;
    busyRef.current = true;
    try {
      const frame = await grabFrame();
      if (!frame) {
        setError("The camera has not produced a frame yet. Give it a moment and try again.");
        return;
      }
      const fd = new FormData();
      fd.append("selfie", frame, "frame.jpg");
      fd.append("reference", reference, reference.name);
      setResult(await api.faceMatchLive(fd));
      setChecks((n) => n + 1);
      setError(null);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      busyRef.current = false;
    }
  }, [reference, grabFrame]);

  // Chained timeout rather than setInterval: a slow round-trip must not stack
  // requests behind itself, so the next poll is scheduled only once one lands.
  useEffect(() => {
    if (!live || !on || !reference) return;
    let cancelled = false;
    let timer: any;
    const loop = async () => {
      await matchOnce();
      if (!cancelled) timer = setTimeout(loop, 700);
    };
    loop();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [live, on, reference, matchOnce]);

  /* Take the current frame and hand it up as the selfie.
   *
   * Without this the panel was a similarity read-out and nothing more: it
   * scored the webcam against the ID and then threw the frame away, so a live
   * capture could never become the packet that gets verified. The file is
   * named and typed like an upload because that is exactly what it becomes. */
  const useAsSelfie = useCallback(async () => {
    const frame = await grabFrame();
    if (!frame) {
      setError("The camera has not produced a frame yet. Give it a moment and try again.");
      return;
    }
    onCapture?.(new File([frame], `live-selfie-${Date.now()}.jpg`, { type: "image/jpeg" }));
    setCaptured(true);
    setTimeout(() => setCaptured(false), 2500);
  }, [grabFrame, onCapture]);

  const sim = result?.similarity ?? null;
  const thr = result?.threshold ?? band?.low ?? 0.5;
  const state =
    result == null ? "idle" : sim == null ? "noface" : result.match ? "match" : "nomatch";
  const tone =
    state === "match" ? "text-pass" : state === "nomatch" ? "text-reject" : "text-slate-400";
  const ring =
    state === "match" ? "border-pass/60" : state === "nomatch" ? "border-reject/60" : "border-edge";

  return (
    <div className="space-y-3">
      <div className={`relative overflow-hidden rounded border bg-ink-950 ${ring}`}>
        <video ref={videoRef} playsInline muted className="w-full scale-x-[-1] bg-black" />
        <canvas ref={canvasRef} className="hidden" />

        {!on && (
          <div className="absolute inset-0 grid place-items-center bg-ink-950/80 text-center text-sm text-slate-400">
            <div className="space-y-2 px-6">
              <p>The camera stream stays in your browser.</p>
              <p className="text-xs text-slate-500">
                Single frames are posted for matching and never written to disk.
              </p>
            </div>
          </div>
        )}

        {/* Is it on, and is it working? Answered in the corner of the frame
            rather than in a caption underneath it, because that is where the
            eye already is once the camera opens. The dot pulses only while the
            loop is actually polling - a still dot means the camera is open and
            nothing is being scored, which is a real and previously invisible
            state. */}
        {on && (
          <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-ink-950/80 px-1.5 py-1 ring-1 ring-edge backdrop-blur-sm">
            <span
              className={clsx("h-1.5 w-1.5 shrink-0 rounded-full", live ? "animate-pulse bg-accent" : "bg-slate-500")}
              aria-hidden
            />
            <span className="label text-slate-300">{live ? "matching" : "camera on"}</span>
            {checks > 0 && <span className="num text-2xs text-slate-500">{checks}</span>}
          </div>
        )}

        {on && sim === null && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-ink-950 via-ink-950/85 to-transparent px-3 pb-3 pt-8">
            <p className="max-w-[42ch] text-2xs leading-snug text-slate-400">
              {state === "noface"
                ? "No face found in that frame - move into the light and centre your face, it will retry."
                : !reference
                ? "No ID document to match against yet. Add one in the slot above and the score appears here."
                : live
                ? "Scoring the first frame..."
                : "Press Match live and the similarity appears here, updating about every second."}
            </p>
          </div>
        )}

        {on && sim !== null && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-ink-950 via-ink-950/85 to-transparent px-3 pb-3 pt-8">
            <div className="flex items-end justify-between gap-3">
              <div className="min-w-0">
                <div className={`num text-2xl font-medium ${tone}`}>{sim.toFixed(3)}</div>
                <div className="num text-2xs text-slate-500">
                  threshold {thr.toFixed(3)} · {result?.confidence}
                </div>
              </div>
              {/* The verdict was the same size and weight as the caption beside
                  it. It is the answer; it gets a chip. */}
              <div
                className={clsx(
                  "shrink-0 rounded px-2 py-1 text-xs font-medium uppercase tracking-[0.12em] ring-1",
                  state === "match" ? "bg-pass/15 text-pass ring-pass/40" : "bg-reject/15 text-reject ring-reject/40"
                )}
              >
                {state === "match" ? "match" : "no match"}
              </div>
            </div>
            {/* Where this score sits relative to the line, at a glance. */}
            <div className="relative mt-2 h-1 bg-ink-800">
              <div
                className="absolute inset-y-0 left-0 bg-current opacity-70"
                style={{ width: `${Math.max(0, Math.min(1, (sim + 1) / 2)) * 100}%` }}
              />
              <div
                className="absolute -top-1 h-3 w-0.5 bg-slate-200"
                style={{ left: `${Math.max(0, Math.min(1, (thr + 1) / 2)) * 100}%` }}
              />
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {!on ? (
          <button
            onClick={start}
            className="btn-primary"
          >
            Start camera
          </button>
        ) : (
          <>
            {onCapture && (
              <button onClick={useAsSelfie} className="btn-accent">
                {captured ? "Selfie set ✓" : "Use as selfie"}
              </button>
            )}
            <button
              onClick={() => setLive((v) => !v)}
              disabled={!reference}
              className="btn-primary"
            >
              {live ? "Pause matching" : "Match live"}
            </button>
            <button
              onClick={matchOnce}
              disabled={!reference || live}
              className="btn"
            >
              Match once
            </button>
            <button
              onClick={stop}
              className="btn hover:border-reject/60 hover:text-reject"
            >
              Stop
            </button>
          </>
        )}
        {checks > 0 && (
          <span className="num text-2xs text-slate-500">
            {checks} {checks === 1 ? "frame scored" : "frames scored"}
            {result && ` · ${result.latency_ms.toFixed(0)} ms`}
          </span>
        )}
      </div>

      {/* This is a precondition, not a footnote: without an ID there is nothing
          to match a face against, and both matching buttons are disabled. In
          slate-500 at 12px that read as a caption and people sat looking at a
          dead panel. */}
      {!reference && (
        <p className="rounded border border-review/30 bg-review/[0.07] px-2.5 py-1.5 text-2xs leading-relaxed text-review">
          Add an ID document in the slot above - this scores your camera against the portrait on it.
        </p>
      )}
      {result?.detail && <p className="text-xs leading-relaxed text-slate-400">{result.detail}</p>}
      {band && (
        <p className="text-2xs leading-relaxed text-slate-600">
          Identity threshold {band.low.toFixed(3)} - {band.fitted_on}
        </p>
      )}
      {error && (
        <p className="rounded border border-reject/40 bg-reject/10 px-2.5 py-1.5 text-xs leading-relaxed text-reject">
          {error}
        </p>
      )}
    </div>
  );
}

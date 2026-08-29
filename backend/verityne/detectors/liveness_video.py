"""Detector 2 - Liveness Video Analyzer.

Individual face-swap frames can look clean; the giveaway is usually temporal.
We sample frames, score each with the same image detector, and then add three
temporal signals that a per-frame model cannot see:

  * identity drift - cosine distance between consecutive frame embeddings.
    A real face stays the same person frame to frame; a swap flickers.
  * pose jitter - second derivative of the head-pose proxy from face landmarks.
    Real head turns are smooth; blended frames snap.
  * boundary discontinuity - optical-flow residual spikes where a swapped
    region is re-composited.

Blink cadence is included as a *proxy* (eye-region intensity variance over
time), not as landmark-precise eyelid tracking. It is reported as such.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..config import MAX_VIDEO_FRAMES, VIDEO_FRAME_STRIDE
from ..schemas import DetectorOutput
from ..utils.artifacts import save_heatmap
from ..utils.images import crop_face, detect_faces, overlay_heatmap
from ..utils.spectral import spectral_features
from .base import Detector, SubmissionPayload, clamp
from .models import cosine, deepfake_classifier, face_embedder
from .selfie_deepfake import heuristic_spectral_score


def sample_frames(path: str, stride: int = VIDEO_FRAME_STRIDE, cap: int = MAX_VIDEO_FRAMES) -> Tuple[List[np.ndarray], Dict]:
    vc = cv2.VideoCapture(str(path))
    if not vc.isOpened():
        return [], {"error": "video could not be opened"}
    total = int(vc.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(vc.get(cv2.CAP_PROP_FPS) or 0.0)
    frames, idx = [], 0
    # If the clip is long, widen the stride so we still cover its whole span.
    if total > 0 and total // max(1, stride) > cap:
        stride = max(stride, total // cap)
    while len(frames) < cap:
        ok, frame = vc.read()
        if not ok:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            if max(h, w) > 720:
                s = 720 / max(h, w)
                rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(rgb)
        idx += 1
    vc.release()
    return frames, {"total_frames": total, "fps": round(fps, 2), "sampled": len(frames), "stride": stride}


def _pose_proxy(box, landmarks) -> Tuple[float, float]:
    """(roll, yaw) proxies from 5-point landmarks; falls back to box geometry."""
    if landmarks is None or len(landmarks) < 3:
        x1, y1, x2, y2 = box
        return 0.0, float((x2 - x1) / max(1.0, y2 - y1))
    le, re, nose = landmarks[0], landmarks[1], landmarks[2]
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0] + 1e-6)))
    eye_mid_x = (le[0] + re[0]) / 2.0
    eye_span = abs(re[0] - le[0]) + 1e-6
    yaw = float((nose[0] - eye_mid_x) / eye_span)  # nose offset between the eyes
    return roll, yaw


class LivenessVideoDetector(Detector):
    name = "liveness_video"
    heavy = True

    def applicable(self, payload: SubmissionPayload) -> bool:
        return payload.video_path is not None

    def _run(self, payload: SubmissionPayload) -> DetectorOutput:
        frames, meta = sample_frames(str(payload.video_path))
        reasons: List[str] = []
        signals: Dict[str, object] = dict(meta)

        if len(frames) < 3:
            return DetectorOutput(
                name=self.name, label=self.label, score=0.5, confidence=0.15, status="ok",
                reasons=["Liveness video was too short or unreadable to analyse temporally"],
                signals=signals, detail="fewer than 3 usable frames",
            )

        clf = deepfake_classifier()
        emb = face_embedder()

        crops, poses, eye_means, boxes = [], [], [], []
        for f in frames:
            fb = detect_faces(f)
            if not fb:
                crops.append(None)
                poses.append(None)
                eye_means.append(None)
                boxes.append(None)
                continue
            box = fb[0]
            boxes.append(box)
            crop = crop_face(f, box, size=224)
            crops.append(crop)
            poses.append(_pose_proxy(box, None))
            # Upper third of the face crop is the eye band; its brightness dips on a blink.
            eye_band = crop[int(0.22 * 224): int(0.45 * 224), :, :]
            eye_means.append(float(cv2.cvtColor(eye_band, cv2.COLOR_RGB2GRAY).mean()))

        valid = [c for c in crops if c is not None]
        face_rate = len(valid) / len(frames)
        signals["face_detection_rate"] = round(face_rate, 3)
        if face_rate < 0.4:
            return DetectorOutput(
                name=self.name, label=self.label, score=0.55, confidence=0.25,
                reasons=[f"A face was visible in only {face_rate:.0%} of sampled frames - liveness cannot be established"],
                signals=signals,
            )

        # --- per-frame appearance ---------------------------------------------------
        if clf is not None:
            per_frame = clf.predict_batch(valid)
            signals["cnn_model"] = clf.repo_id
        else:
            per_frame = [heuristic_spectral_score(spectral_features(c)) for c in valid]
        per_frame = [float(p) for p in per_frame]
        frame_mean = float(np.mean(per_frame))
        frame_p90 = float(np.percentile(per_frame, 90))
        signals["frame_scores"] = [round(p, 3) for p in per_frame]
        signals["frame_mean"] = round(frame_mean, 4)
        signals["frame_p90"] = round(frame_p90, 4)

        # --- identity drift ----------------------------------------------------------
        identity_drift = 0.0
        if emb is not None and len(valid) >= 2:
            vecs = [emb.embed(c) for c in valid]
            sims = [cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
            identity_drift = float(1.0 - np.mean(sims))
            signals["mean_consecutive_identity_similarity"] = round(float(np.mean(sims)), 4)
            signals["min_consecutive_identity_similarity"] = round(float(np.min(sims)), 4)
            payload.cache["video_embeddings"] = vecs
        signals["identity_drift"] = round(identity_drift, 4)

        # --- pose smoothness ---------------------------------------------------------
        rolls = [p[0] for p in poses if p is not None]
        pose_jitter = 0.0
        if len(rolls) >= 4:
            second_deriv = np.diff(np.diff(rolls))
            pose_jitter = float(np.std(second_deriv))
            signals["pose_roll_range_deg"] = round(float(np.ptp(rolls)), 2)
        signals["pose_jitter"] = round(pose_jitter, 3)

        # --- boundary discontinuity ---------------------------------------------------
        flow_spike = 0.0
        if len(valid) >= 3:
            residuals = []
            for a, b in zip(valid[:-1], valid[1:]):
                ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
                gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
                flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                residuals.append(float(np.linalg.norm(flow, axis=2).mean()))
            if residuals:
                med = float(np.median(residuals)) + 1e-6
                flow_spike = float(max(residuals) / med)
                signals["flow_residuals"] = [round(r, 3) for r in residuals]
        signals["flow_spike_ratio"] = round(flow_spike, 3)

        # --- blink proxy ---------------------------------------------------------------
        blink_events = 0
        eye_series = [e for e in eye_means if e is not None]
        if len(eye_series) >= 5:
            arr = np.array(eye_series, dtype=np.float32)
            baseline = float(np.median(arr))
            spread = float(np.std(arr)) + 1e-6
            dips = (arr < baseline - 1.2 * spread).astype(int)
            blink_events = int(np.sum(np.diff(dips) == 1))
            signals["eye_band_variation"] = round(spread, 3)
        duration_s = meta.get("sampled", 0) * meta.get("stride", 1) / max(1.0, meta.get("fps", 25.0))
        signals["blink_events_proxy"] = blink_events
        signals["approx_duration_s"] = round(duration_s, 2)
        signals["blink_signal_is_proxy"] = True

        # --- fuse the temporal picture --------------------------------------------------
        drift_score = clamp((identity_drift - 0.06) / 0.22)
        jitter_score = clamp((pose_jitter - 2.5) / 9.0)
        flow_score = clamp((flow_spike - 2.2) / 3.5)
        appearance = clamp(0.6 * frame_mean + 0.4 * frame_p90)
        temporal = clamp(0.45 * drift_score + 0.30 * jitter_score + 0.25 * flow_score)

        score = clamp(0.55 * appearance + 0.45 * temporal)
        confidence = clamp(0.35 + 0.45 * face_rate + 0.2 * min(1.0, len(valid) / 12.0))

        if drift_score > 0.5:
            reasons.append(
                f"Identity flickers between frames (mean consecutive similarity "
                f"{signals.get('mean_consecutive_identity_similarity', 0):.2f}) - characteristic of a face swap"
            )
        if jitter_score > 0.5:
            reasons.append(f"Head-pose transitions are not smooth (jitter {pose_jitter:.1f}°), suggesting frame-level compositing")
        if flow_score > 0.5:
            reasons.append(f"Optical-flow residual spikes {flow_spike:.1f}x above baseline at frame boundaries")
        if frame_p90 > 0.7:
            reasons.append(f"{sum(1 for p in per_frame if p > 0.7)}/{len(per_frame)} sampled frames independently score as synthetic")
        if duration_s > 6 and blink_events == 0:
            reasons.append(f"No blink detected across ~{duration_s:.0f}s of video (proxy signal) - humans blink every 4-6 seconds")
            score = clamp(score + 0.08)
        if not reasons:
            reasons.append(f"Temporal behaviour across {len(valid)} sampled frames is consistent with a live capture")

        heatmap_url = None
        if valid:
            worst = int(np.argmax(per_frame))
            g = cv2.cvtColor(valid[worst], cv2.COLOR_RGB2GRAY).astype(np.float32)
            cam = clf.saliency(valid[worst]) if clf is not None else None
            if cam is None:
                cam = np.abs(g - cv2.GaussianBlur(g, (0, 0), 2.0))
            try:
                heatmap_url = save_heatmap(overlay_heatmap(valid[worst], cam), payload.submission_id, "liveness")
                signals["heatmap_frame_index"] = worst
            except Exception:
                heatmap_url = None

        signals["subscores"] = {
            "appearance": round(appearance, 3), "temporal": round(temporal, 3),
            "identity_drift": round(drift_score, 3), "pose_jitter": round(jitter_score, 3),
            "flow_discontinuity": round(flow_score, 3),
        }
        return DetectorOutput(
            name=self.name, label=self.label, score=score, confidence=confidence,
            reasons=reasons, signals=signals, heatmap_url=heatmap_url,
        )

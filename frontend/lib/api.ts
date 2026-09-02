// Thin API client. Everything goes through the Next rewrite at /api so the
// browser stays same-origin (no CORS, and heatmap <img> src values just work).
export const API_KEY = process.env.NEXT_PUBLIC_API_KEY || "verityne-demo-key";

export type Verdict = "PASS" | "REVIEW" | "REJECT";

export interface DetectorOutput {
  name: string;
  label: string;
  score: number;
  confidence: number;
  status: "ok" | "skipped" | "error";
  reasons: string[];
  signals: Record<string, any>;
  heatmap_url: string | null;
  latency_ms: number;
  detail?: string | null;
}

export interface VerifyResponse {
  submission_id: string;
  merchant_id: string;
  verdict: Verdict;
  final_score: number;
  abstained: boolean;
  top_reasons: string[];
  explanation: string;
  attack_pattern: string | null;
  generator_guess: string | null;
  detector_breakdown: Record<string, DetectorOutput>;
  heatmaps: Record<string, string>;
  latency_ms: number;
  fusion_model: string;
  policy: Record<string, any>;
}

export interface AblationRow {
  detector: string;
  roc_auc: number;
  auc_drop: number;
  share_of_headline_auc_above_chance: number;
}

export interface AblationLeak {
  field: string;
  value: string;
  p_fraud: number;
  n: number;
  class: string;
  expected?: boolean;
  justification?: string | null;
}

export interface Ablation {
  protocol: string;
  n_train: number;
  n_test: number;
  full_model: { roc_auc: number };
  ranked: AblationRow[];
  leaks: AblationLeak[];
  one_sided_but_expected: AblationLeak[];
  corpus_clean: boolean;
  most_load_bearing: string;
  note: string;
}

export interface BehavioralStrategyResult {
  n: number;
  n_positive?: number;
  auc: number | null;
  ap: number | null;
  threshold: number;
  recall_at_threshold: number;
  false_positive_rate: number;
}

export interface BehavioralReport {
  corpus: {
    human_aalto_sessions: number;
    human_cmu_sessions: number;
    bot_sessions: number;
    strategies: Record<string, number>;
    human_source: string;
    bot_source: string;
  };
  shipped_feature_set: string;
  threshold: number;
  human_fpr_budget: number;
  hyperparameters: Record<string, number>;
  headline: {
    held_out_auc: number | null;
    held_out_recall: number;
    human_false_positive_rate: number;
    worst_unseen_strategy: string;
    worst_unseen_strategy_auc: number | null;
    transfer_false_positive_rate: number | null;
  };
  ablation: Record<string, {
    n_features: number;
    threshold: number;
    held_out_subject_disjoint: any;
    leave_one_strategy_out: Record<string, BehavioralStrategyResult>;
    transfer_to_unseen_population: any;
  }>;
  feature_importance: { feature: string; gain: number }[];
  sweep: { ran: boolean; configurations?: number; objective?: string; top?: any[] };
  note: string;
}

/** One generator family scored against its real control. `auc` is null when a
 *  group came back empty - drawn as an absence, never as a zero. */
export interface RealFacesFamily {
  n_real: number;
  n_fake: number;
  auc: number | null;
  mean_real?: number;
  mean_fake?: number;
  recall_at?: Record<string, number>;
  real_fpr_at?: Record<string, number>;
}

export interface RealFacesTrack {
  source: string;
  paired?: boolean;
  pairing_note?: string;
  /** Present on the StyleGAN track: a high score there may be memorisation,
   *  because the checkpoint names no training data. Rendered beside the number. */
  leakage_warning?: string;
  why?: string;
  families?: Record<string, string>;
  face_detection_rate?: Record<string, number>;
  false_positive_rate?: Record<string, Record<string, number>>;
  protocols?: Record<string, { per_family: Record<string, RealFacesFamily> }>;
}

export interface RealFacesReport {
  source: string;
  generated_at: string;
  detector: {
    active_checkpoint: string | null;
    cnn_weight: number;
    spectral_weight: number;
    note: string;
  };
  what_this_measures: string;
  protocols: Record<string, string>;
  operating_points: number[];
  tracks: Record<string, RealFacesTrack>;
  headline?: {
    protocol: string;
    per_family_auc: Record<string, number>;
    worst_family: string;
    worst_family_auc: number;
    best_family: string;
    best_family_auc: number;
    spread: number;
    note: string;
  };
}

export interface RealMetrics {
  available: string[];
  missing: { name: string; source: string; expected_at: string }[];
  reports: Record<string, any> & { selfie_faces?: RealFacesReport };
  note: string;
}

export interface CorrectionEvidence {
  file: string;
  path?: string | string[];
  source?: "code";
  symbol?: string;
  note?: string;
}

export interface Correction {
  id: string;
  order: number;
  title: string;
  believed: string;
  measured: string;
  metric: string;
  /** Present on entries that moved one number. */
  before?: number;
  after?: number;
  direction?: "up" | "down" | "down_is_honest";
  /** Present instead of before/after when the finding is several values at once. */
  series?: { label: string; value: number }[];
  reference?: { label: string; value: number };
  status: "fixed" | "open" | "designed_around";
  outcome: string;
  how_found: string;
  evidence: CorrectionEvidence[];
  readme: string;
}

export interface CorrectionsReport {
  generated_by: string;
  what_this_is: string;
  how_found_legend: Record<string, string>;
  counts: Record<string, number>;
  n: number;
  corrections: Correction[];
}

export interface FaceMatchResult {
  similarity: number | null;
  match: boolean | null;
  threshold: number;
  threshold_source: string;
  selfie_face_found: boolean;
  reference_face_found: boolean;
  margin: number | null;
  confidence: "strong" | "borderline" | "weak";
  detail: string;
  latency_ms: number;
}

export interface BehavioralAck {
  token: string;
  event_count: number;
  features_extracted: number;
  preview_score: number;
  preview_confidence: number;
  preview_reasons: string[];
}

export type LinkKind = "face" | "asset_exact" | "asset_near";

export interface ThreatNode {
  id: string;
  merchant_id: string;
  claimed_name: string | null;
  verdict: Verdict;
  score: number;
  created_at: string;
  generator_guess: string | null;
  thumb_url: string | null;
  /** Connected-component index, or null for an isolated submission. */
  ring: number | null;
}

export interface ThreatEdge {
  source: string;
  target: string;
  kind: LinkKind;
  /** Cosine similarity for a face link; 1.0 for a byte-identical asset. */
  weight: number;
  name_differs?: boolean;
}

export interface ThreatRing {
  id: number;
  size: number;
  merchants: string[];
  distinct_names: number;
  max_score: number;
  /** True when at least one edge in the ring is a shared SHA-256. That is the
   *  difference between a ring that is a fact and one that is an inference. */
  has_exact_asset_reuse: boolean;
}

export interface ThreatGraph {
  generated_at: string;
  window_hours: number;
  /** Provenance for the edge threshold, shown on the page. A graph drawn at the
   *  verification point rather than the search point wires every genuine
   *  applicant to a stranger - see linkage.SAME_PERSON. */
  threshold: { same_person: number; fitted_on: string };
  nodes: ThreatNode[];
  edges: ThreatEdge[];
  rings: ThreatRing[];
}

export interface GauntletResult {
  submission_id: string;
  name: string;
  truth: "real" | "fake";
  attack_type: string | null;
  verdict: Verdict;
  score: number;
  correct: boolean;
  top_reasons: string[];
  latency_ms: number;
  thumb_url: string | null;
}

export interface GauntletSummary {
  total: number;
  fakes_caught: number;
  fakes_total: number;
  reals_passed: number;
  reals_reviewed: number;
  reals_rejected: number;
  reals_total: number;
  detection_rate: number;
  false_accept_rate: number;
  false_reject_rate: number;
  accuracy: number;
  mean_latency_ms: number;
  results: GauntletResult[];
  wall_clock_ms?: number;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, {
    ...init,
    headers: { "X-API-Key": API_KEY, ...(init.headers || {}) },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}${body ? ` - ${body.slice(0, 300)}` : ""}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<any>("/health"),

  verify: (form: FormData) =>
    request<VerifyResponse>("/verify", { method: "POST", body: form }),

  faceMatchLive: (form: FormData) =>
    request<FaceMatchResult>("/face-match/live", { method: "POST", body: form }),

  faceMatchThreshold: () =>
    request<{ low: number; high: number; fitted_on: string }>("/face-match/threshold"),

  submissions: (params: Record<string, string | number> = {}) => {
    const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
    return request<any>(`/submissions?${qs}`);
  },

  submission: (id: string) => request<any>(`/submissions/${id}`),

  /** Re-run the pipeline over an already-stored packet, and return a full verdict. */
  rescore: (id: string) =>
    request<VerifyResponse>(`/submissions/${id}/rescore`, { method: "POST" }),

  /** Post the form-fill telemetry buffer. Returns the detector's preview of what
   *  it read, which is also what `/verify` will see once it binds the token. */
  behavioral: (buffer: unknown) =>
    request<BehavioralAck>("/behavioral", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buffer),
    }),

  gauntletManifest: () => request<any>("/gauntlet"),

  metrics: () => request<any>("/metrics"),

  ablation: () => request<Ablation>("/metrics/ablation"),

  /** Detector 6's evaluation report. 404s until the corpus has been built and
   *  the model fitted - the page reports that absence rather than drawing zeros. */
  behavioralMetrics: () => request<BehavioralReport>("/metrics/behavioral"),

  /** The audit trail. 404s until `make corrections` has assembled it. */
  corrections: () => request<CorrectionsReport>("/metrics/corrections"),

  /** Every detector result measured on third-party data. Each report inside is
   *  independent and any may be absent; `available` says which actually ran, so
   *  a dataset that was never downloaded reads as missing rather than as zero. */
  realMetrics: () => request<RealMetrics>("/metrics/real"),

  costCurve: (p: Record<string, number>) => {
    const qs = new URLSearchParams(Object.entries(p).map(([k, v]) => [k, String(v)]));
    return request<any>(`/metrics/cost-curve?${qs}`);
  },

  /** The fraud-ring graph, from `GET /threat/graph`. If the call fails the page
   *  says so rather than drawing invented edges - a fabricated ring in a fraud
   *  tool is worse than an empty panel. */
  threatGraph: (hours = 168) => request<ThreatGraph>(`/threat/graph?hours=${hours}`),

  attacks: (hours = 24) => request<any>(`/attacks?hours=${hours}`),

  reviewQueue: () => request<any>("/review-queue"),

  decide: (id: string, decision: string, note?: string) =>
    request<any>(`/review/${id}/decision`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, analyst: "ops", note }),
    }),

  reviewAgreement: () => request<any>("/review/agreement"),

  redteam: (count = 1) =>
    request<VerifyResponse[]>("/redteam/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ count, merchant_id: "default" }),
    }),

  batchVerify: (body: Record<string, any>) =>
    request<any>("/batch-verify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
};

export const verdictColor = (v: Verdict) =>
  v === "PASS" ? "pass" : v === "REVIEW" ? "review" : "reject";

export const fmtPct = (x: number | null | undefined, digits = 1) =>
  x === null || x === undefined || Number.isNaN(x) ? "-" : `${(x * 100).toFixed(digits)}%`;

export const fmtInr = (n: number) => {
  if (Math.abs(n) >= 1e7) return `₹${(n / 1e7).toFixed(2)}Cr`;
  if (Math.abs(n) >= 1e5) return `₹${(n / 1e5).toFixed(2)}L`;
  if (Math.abs(n) >= 1e3) return `₹${(n / 1e3).toFixed(1)}k`;
  return `₹${n.toFixed(0)}`;
};

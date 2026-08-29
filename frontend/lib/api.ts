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

  submissions: (params: Record<string, string | number> = {}) => {
    const qs = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
    return request<any>(`/submissions?${qs}`);
  },

  submission: (id: string) => request<any>(`/submissions/${id}`),

  gauntletManifest: () => request<any>("/gauntlet"),

  metrics: () => request<any>("/metrics"),

  costCurve: (p: Record<string, number>) => {
    const qs = new URLSearchParams(Object.entries(p).map(([k, v]) => [k, String(v)]));
    return request<any>(`/metrics/cost-curve?${qs}`);
  },

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
  x === null || x === undefined || Number.isNaN(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const fmtInr = (n: number) => {
  if (Math.abs(n) >= 1e7) return `₹${(n / 1e7).toFixed(2)}Cr`;
  if (Math.abs(n) >= 1e5) return `₹${(n / 1e5).toFixed(2)}L`;
  if (Math.abs(n) >= 1e3) return `₹${(n / 1e3).toFixed(1)}k`;
  return `₹${n.toFixed(0)}`;
};

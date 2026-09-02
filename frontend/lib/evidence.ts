import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Evidence files, read at build time.
 *
 * The landing page states numbers, and this project's whole argument is that a
 * stated number has a committed file behind it. A hard-coded "0.753" in a hero
 * would be the one figure on the site nothing checks - and the README already
 * has a page about what happens when a number drifts away from its evidence.
 *
 * Read on the server at build time rather than fetched: the landing page must
 * render before the API is up, and a marketing surface that spins a loader over
 * its own headline is worse than one that is simply correct.
 */

const EVAL = join(process.cwd(), "..", "eval");

function read<T>(name: string): T | null {
  try {
    return JSON.parse(readFileSync(join(EVAL, name), "utf8")) as T;
  } catch {
    // A missing report is survivable - the section that needs it renders its
    // absence. A wrong number is not, which is why nothing here has a default.
    return null;
  }
}

export interface Headline {
  fusionAuc: number | null;
  corrections: { n: number; fixed: number; open: number } | null;
  behavioral: { auc: number | null; worst: number | null; worstName: string | null } | null;
  realFaces: { worstFamily: string | null; worstAuc: number | null } | null;
  thresholds: { review: number | null; reject: number | null } | null;
  gauntletLatencyMs: number | null;
}

export function headline(): Headline {
  const m: any = read("metrics.json");
  const c: any = read("corrections.json");
  const b: any = read("behavioral.json");
  const rf: any = read("real_faces.json");

  return {
    fusionAuc: m?.fusion?.roc_auc ?? null,
    corrections: c
      ? { n: c.n, fixed: c.counts?.fixed ?? 0, open: c.counts?.open ?? 0 }
      : null,
    behavioral: b
      ? {
          auc: b.headline?.held_out_auc ?? null,
          worst: b.headline?.worst_unseen_strategy_auc ?? null,
          worstName: b.headline?.worst_unseen_strategy ?? null,
        }
      : null,
    realFaces: rf
      ? {
          worstFamily: rf.headline?.worst_family ?? null,
          worstAuc: rf.headline?.worst_family_auc ?? null,
        }
      : null,
    thresholds: m
      ? {
          review: m.fusion?.at_review_threshold?.threshold ?? null,
          reject: m.fusion?.at_reject_threshold?.threshold ?? null,
        }
      : null,
    gauntletLatencyMs: null,
  };
}

export interface PacketCell {
  score: number;
  fraud: boolean;
}

/**
 * Every held-out packet, as one cell.
 *
 * 105 rows is small enough to draw individually, which is the point: an AUC is
 * one number standing in for a distribution, and the distribution is the thing
 * worth looking at. Sorted by score, the overlap between the two classes - the
 * band where a genuine merchant and a fraudulent one score the same - is the
 * shape that makes 0.753 what it is, and no headline figure shows it.
 */
export function packets(): PacketCell[] {
  const m: any = read("metrics.json");
  const raw = m?.raw_scores;
  if (!raw?.y_true?.length) return [];
  return raw.y_true
    .map((y: number, i: number) => ({ score: raw.y_score[i] as number, fraud: y === 1 }))
    .sort((a: PacketCell, b: PacketCell) => a.score - b.score);
}

export interface CorrectionRow {
  order: number;
  title: string;
  status: string;
  metric: string;
  before?: number;
  after?: number;
  series?: { label: string; value: number }[];
}

export function corrections(): CorrectionRow[] {
  const c: any = read("corrections.json");
  return (c?.corrections ?? []).map((x: any) => ({
    order: x.order,
    title: x.title,
    status: x.status,
    metric: x.metric,
    before: x.before,
    after: x.after,
    series: x.series,
  }));
}

import { PASS, REJECT } from "@/lib/palette";
import type { PacketCell } from "@/lib/evidence";

/**
 * The held-out split, one cell per packet.
 *
 * A hero image on a fraud product is usually a stock lock or an abstract mesh.
 * This is the actual evaluation: 105 packets, sorted by the score the system
 * gave them, coloured by what they really were. Brightness is the score.
 *
 * It is on the landing page because it makes the honest argument faster than a
 * paragraph can. The two classes are not two blocks — they interleave through
 * the middle, and that band is every packet where a genuine merchant and a
 * fraudulent one look the same to the model. That overlap *is* 0.753. A vendor
 * showing you a single AUC is showing you a summary of this picture and hoping
 * you do not ask for the picture.
 *
 * Server-rendered from the committed report, so it cannot drift and costs the
 * browser nothing.
 */
export function ScoreField({
  cells,
  // 105 packets. The column count sets the aspect: 35 wide made a three-row
  // strip that read as a progress bar, and the point is a field you can scan
  // for the band where the two colours mix.
  cols = 15,
}: {
  cells: PacketCell[];
  cols?: number;
}) {
  if (!cells.length) return null;

  const rows = Math.ceil(cells.length / cols);
  const cell = 32;
  const gap = 4;
  const w = cols * cell + (cols - 1) * gap;
  const h = rows * cell + (rows - 1) * gap;

  // Brightness carries the score, so the ramp has to start visible: a cell at
  // 0.17 must still read as a cell rather than as background.
  const lo = Math.min(...cells.map((c) => c.score));
  const hi = Math.max(...cells.map((c) => c.score));
  const norm = (s: number) => (hi > lo ? (s - lo) / (hi - lo) : 0.5);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${w} ${h}`}
        width="100%"
        role="img"
        aria-label={
          `Every one of ${cells.length} held-out packets, sorted by risk score. ` +
          `Green cells are genuine merchants, red are fraudulent; brightness is the score. ` +
          `The two classes overlap through the middle of the range.`
        }
        className="block"
      >
        {cells.map((c, i) => {
          const x = (i % cols) * (cell + gap);
          const y = Math.floor(i / cols) * (cell + gap);
          const t = norm(c.score);
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={cell}
              height={cell}
              fill={c.fraud ? REJECT : PASS}
              fillOpacity={0.16 + t * 0.84}
              rx={1}
            />
          );
        })}
      </svg>
      <figcaption className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-1.5 text-2xs text-slate-600">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2" style={{ background: PASS }} />
          genuine merchant
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2" style={{ background: REJECT }} />
          fraudulent
        </span>
        <span>left to right, low risk to high · brightness is the score</span>
      </figcaption>
    </figure>
  );
}

/**
 * The wordmark's glyph: a V built out of whole pixels.
 *
 * It replaced a shield-with-a-tick, which was the stock icon for anything
 * security-adjacent and said nothing this product does. This is a bitmap letter
 * on the same lattice as the display face and the page grid — at 18px the
 * blocks land on device pixels, so it stays sharp where a stroked path goes
 * soft, and it is the one mark that could not have come from an icon set.
 *
 * The vertex is drawn at full accent and the arms one step down, so the eye
 * resolves the shape as a V rather than as six loose squares.
 */
export function Mark({ className = "h-[18px] w-[18px]" }: { className?: string }) {
  // column, row — a 6x5 lattice, drawn bottom-heavy so the point reads.
  const arms: [number, number][] = [
    [0, 0], [5, 0],
    [0, 1], [5, 1],
    [1, 2], [4, 2],
  ];
  const vertex: [number, number][] = [[2, 3], [3, 3]];
  return (
    <svg viewBox="0 0 6 5" className={className} shapeRendering="crispEdges" aria-hidden focusable="false">
      {arms.map(([x, y]) => (
        <rect key={`a${x}${y}`} x={x} y={y} width="1" height="1" fill="currentColor" opacity="0.55" />
      ))}
      {vertex.map(([x, y]) => (
        <rect key={`v${x}${y}`} x={x} y={y} width="1" height="1" fill="currentColor" />
      ))}
    </svg>
  );
}

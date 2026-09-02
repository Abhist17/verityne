"use client";

import { useCallback, useId, useLayoutEffect, useRef, useState } from "react";
import clsx from "clsx";

/**
 * A hover-and-focus explainer, anchored under whatever it wraps.
 *
 * This dashboard names things the way the people who built it talk about them -
 * Gauntlet, Corrections, Threat - and a first-time visitor cannot tell from the
 * word alone which of seven tabs answers their question. Every one of those
 * labels is now a trigger for a sentence saying what is behind it.
 *
 * Hover is not enough on its own: a keyboard user never fires one, so the panel
 * opens on focus too and is wired up with `aria-describedby`, which is what
 * makes a screen reader read the description as part of the link rather than as
 * loose text somewhere after it. It is a description, never the only copy of
 * something you need - the label still has to stand alone.
 *
 * The panel is `position: fixed`, measured from the trigger, rather than
 * `absolute` inside it. That is not a preference. The first version was
 * absolute, and the nav it lives in is `overflow-x: auto` so that seven tabs can
 * scroll on a phone - which makes the nav a clipping context, so every tooltip
 * was rendered, marked visible, and then clipped to nothing by the scroll box.
 * The same trap is waiting in `.segment` and in every `.scroll-x` table, so the
 * fix belongs here rather than in each caller.
 *
 * Deliberately not a dialog: no focus trap, no click to open, nothing to
 * dismiss. `pointer-events-none` means it can never eat a click meant for the
 * trigger underneath it.
 */
export function Hint({
  children,
  title,
  body,
  meta,
  align = "left",
  className,
}: {
  children: React.ReactNode;
  /** The term being explained. Omit when the trigger already reads as the title. */
  title?: string;
  body: React.ReactNode;
  /** An optional monospace line under the body: thresholds, counts, a file path. */
  meta?: React.ReactNode;
  align?: "left" | "right" | "center";
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const anchor = useRef<HTMLSpanElement>(null);
  const panel = useRef<HTMLSpanElement>(null);
  const id = useId();

  const WIDTH = 304; // 19rem, matched by the inline width below
  const GAP = 12;
  const MARGIN = 8; // keeps the panel off the viewport edge

  const place = useCallback(() => {
    const a = anchor.current?.getBoundingClientRect();
    if (!a) return;
    const w = Math.min(WIDTH, window.innerWidth - MARGIN * 2);
    let left =
      align === "right" ? a.right - w : align === "center" ? a.left + a.width / 2 - w / 2 : a.left;
    left = Math.max(MARGIN, Math.min(left, window.innerWidth - w - MARGIN));

    // Flip above the trigger when there is not room below it.
    const h = panel.current?.offsetHeight ?? 0;
    const below = a.bottom + GAP;
    const top = h && below + h > window.innerHeight - MARGIN ? a.top - GAP - h : below;
    setPos({ top, left });
  }, [align]);

  // Layout effect so the first paint is already in the right place; measuring in
  // a plain effect shows one frame at the top-left corner first.
  useLayoutEffect(() => {
    if (!open) return;
    place();
    const onMove = () => place();
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [open, place]);

  return (
    <span
      ref={anchor}
      className={clsx("relative inline-flex", className)}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocusCapture={() => setOpen(true)}
      onBlurCapture={() => setOpen(false)}
    >
      <span aria-describedby={open ? id : undefined} className="inline-flex">
        {children}
      </span>
      <span
        ref={panel}
        id={id}
        role="tooltip"
        hidden={!open}
        style={{
          position: "fixed",
          top: pos?.top ?? -9999,
          left: pos?.left ?? -9999,
          width: `min(${WIDTH}px, calc(100vw - ${MARGIN * 2}px))`,
        }}
        className={clsx(
          "pointer-events-none z-[60] border border-edge-strong bg-ink-900 p-3 text-left",
          "shadow-[0_8px_24px_rgba(0,0,0,0.6)] animate-rise"
        )}
      >
        {title && <span className="label block text-accent">{title}</span>}
        <span className={clsx("block text-xs leading-relaxed text-slate-300", title && "mt-1.5")}>
          {body}
        </span>
        {meta && (
          <span className="num mt-2 block border-t border-edge pt-2 text-2xs text-slate-500">
            {meta}
          </span>
        )}
      </span>
    </span>
  );
}

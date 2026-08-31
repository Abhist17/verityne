/**
 * Detector 6's client half: how the form was filled, not what was uploaded.
 *
 * The buffer this produces is posted to `POST /api/behavioral` before the files
 * go up, and `/verify` binds it by token afterwards — see routes_behavioral.py
 * for why the ordering is that way round.
 *
 * Two things here are easy to get wrong, and both are load-bearing.
 *
 * **Key identity is redacted at source.** The detector needs exactly two facts
 * about each keystroke: was it printable, and was it a correction. It never
 * needs the character. Sending the real `event.key` would ship the applicant's
 * PAN and date of birth to the server as plaintext keystroke logs, which is a
 * far worse liability than the fraud it detects. Printable keys are therefore
 * reported as a single placeholder character, which preserves the `len(key)==1`
 * test the extractor uses to count them, and control keys are reported by name
 * so `Backspace` still reads as a correction.
 *
 * **The timezone offset is the raw `getTimezoneOffset()` value.** That API
 * returns minutes *behind* UTC, so IST is `-330`, not `+330`. The extractor
 * documents that it flips the sign itself. Sending the true offset instead
 * makes every genuine Indian applicant's locale look incoherent with their
 * clock and trips a fraud rule on the honest path.
 */

/** Buffer caps. A careful human fill lands far under these; they exist so a long
 *  idle session cannot grow the payload past the server's 512 KB limit. */
const MAX_KEYS = 500;
const MAX_MOUSE = 400;
const MAX_FOCUS = 120;
const MAX_PASTE = 40;
/** Pointer sampling floor, ms. Raw mousemove fires far denser than the path
 *  statistics need, and the extractor reads interval *variance* — so this
 *  throttles on elapsed time without regularising it into a constant tick,
 *  which would itself look synthetic. */
const MOUSE_MIN_DT = 25;

/** Stands in for any printable character. One char, so it still counts as
 *  printable; no information about which character it was. */
const REDACTED_PRINTABLE = "·";

export interface KeyEvent {
  key: string;
  down: number;
  up?: number;
  field?: string;
}
export interface TelemetryBuffer {
  token: string;
  merchant_id: string;
  form_loaded_at: number;
  submitted_at: number;
  declared_country?: string;
  keys: KeyEvent[];
  mouse: { t: number; x: number; y: number }[];
  focus: { field: string; index: number; in: number; out?: number }[];
  paste: { field: string; t: number; length: number }[];
  env: Record<string, unknown>;
}

const fieldOf = (el: EventTarget | null): string | undefined => {
  const node = el as HTMLElement | null;
  if (!node || typeof node.closest !== "function") return undefined;
  const holder = node.closest<HTMLElement>("[data-tele-field]");
  return holder?.dataset.teleField;
};

/**
 * Collects one form-fill session. Constructed on form load; `snapshot()` returns
 * the buffer to post at submit time. Safe to construct outside a browser (the
 * constructor no-ops), so it can be imported from a server component.
 */
export class TelemetryCollector {
  readonly token: string;
  private t0 = 0;
  private keys: KeyEvent[] = [];
  private open = new Map<string, KeyEvent>();
  private mouse: TelemetryBuffer["mouse"] = [];
  private focus: TelemetryBuffer["focus"] = [];
  private paste: TelemetryBuffer["paste"] = [];
  private order: string[] = [];
  private lastMouse = 0;
  private detach: (() => void) | null = null;

  constructor() {
    this.token = TelemetryCollector.mintToken();
  }

  /** 32 hex chars — comfortably inside the envelope's 8..64 length bound. */
  private static mintToken(): string {
    const c = typeof crypto !== "undefined" ? crypto : undefined;
    if (c?.randomUUID) return c.randomUUID().replace(/-/g, "");
    const b = new Uint8Array(16);
    c?.getRandomValues?.(b);
    return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
  }

  private now() {
    return Math.round(performance.now() - this.t0);
  }

  start() {
    if (typeof window === "undefined" || this.detach) return;
    this.t0 = performance.now();

    const onKeyDown = (e: KeyboardEvent) => {
      if (this.keys.length >= MAX_KEYS) return;
      // Modifier-held combinations are shortcuts, not typing.
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const rec: KeyEvent = {
        key: e.key.length === 1 ? REDACTED_PRINTABLE : e.key,
        down: this.now(),
        field: fieldOf(e.target),
      };
      this.keys.push(rec);
      // Keyed by the real code so keyup pairs correctly; never stored.
      if (!this.open.has(e.code)) this.open.set(e.code, rec);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      const rec = this.open.get(e.code);
      if (rec) {
        rec.up = this.now();
        this.open.delete(e.code);
      }
    };

    const onMouseMove = (e: MouseEvent) => {
      const t = this.now();
      if (t - this.lastMouse < MOUSE_MIN_DT || this.mouse.length >= MAX_MOUSE) return;
      this.lastMouse = t;
      this.mouse.push({ t, x: e.clientX, y: e.clientY });
    };

    const onFocusIn = (e: FocusEvent) => {
      const field = fieldOf(e.target);
      if (!field || this.focus.length >= MAX_FOCUS) return;
      if (!this.order.includes(field)) this.order.push(field);
      this.focus.push({ field, index: this.order.indexOf(field), in: this.now() });
    };

    const onFocusOut = (e: FocusEvent) => {
      const field = fieldOf(e.target);
      if (!field) return;
      for (let i = this.focus.length - 1; i >= 0; i--) {
        if (this.focus[i].field === field && this.focus[i].out === undefined) {
          this.focus[i].out = this.now();
          return;
        }
      }
    };

    const onPaste = (e: ClipboardEvent) => {
      const field = fieldOf(e.target);
      if (!field || this.paste.length >= MAX_PASTE) return;
      // Length only. The pasted text itself is never read or transmitted.
      const length = e.clipboardData?.getData("text")?.length ?? 0;
      this.paste.push({ field, t: this.now(), length });
    };

    const w = window;
    w.addEventListener("keydown", onKeyDown, true);
    w.addEventListener("keyup", onKeyUp, true);
    w.addEventListener("mousemove", onMouseMove, { passive: true });
    w.addEventListener("focusin", onFocusIn, true);
    w.addEventListener("focusout", onFocusOut, true);
    w.addEventListener("paste", onPaste, true);

    this.detach = () => {
      w.removeEventListener("keydown", onKeyDown, true);
      w.removeEventListener("keyup", onKeyUp, true);
      w.removeEventListener("mousemove", onMouseMove);
      w.removeEventListener("focusin", onFocusIn, true);
      w.removeEventListener("focusout", onFocusOut, true);
      w.removeEventListener("paste", onPaste, true);
    };
  }

  stop() {
    this.detach?.();
    this.detach = null;
  }

  private env(): Record<string, unknown> {
    const nav = typeof navigator !== "undefined" ? navigator : undefined;
    return {
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone ?? "",
      // RAW getTimezoneOffset(): minutes behind UTC, so IST is -330. The server
      // flips the sign. See the note at the top of this file.
      timezone_offset_min: new Date().getTimezoneOffset(),
      languages: nav?.languages ? Array.from(nav.languages) : nav?.language ? [nav.language] : [],
      webdriver: nav?.webdriver === true,
      screen:
        typeof screen !== "undefined" ? { w: screen.width, h: screen.height } : undefined,
      hardware_concurrency: nav?.hardwareConcurrency,
      touch_points: nav?.maxTouchPoints,
    };
  }

  /** The buffer as posted. Cheap and side-effect free, so it can be called more
   *  than once (a retried submit reuses the same token deliberately). */
  snapshot(merchantId: string, declaredCountry = "IN"): TelemetryBuffer {
    return {
      token: this.token,
      merchant_id: merchantId,
      form_loaded_at: 0,
      submitted_at: this.now(),
      declared_country: declaredCountry,
      keys: this.keys,
      mouse: this.mouse,
      focus: this.focus,
      paste: this.paste,
      env: this.env(),
    };
  }

  get eventCount() {
    return this.keys.length + this.mouse.length + this.focus.length + this.paste.length;
  }
}

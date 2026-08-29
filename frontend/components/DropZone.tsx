"use client";

import { useCallback, useRef, useState } from "react";
import clsx from "clsx";

export function DropZone({
  label, hint, accept, file, onFile,
}: {
  label: string;
  hint: string;
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
}) {
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const take = useCallback(
    (f: File | null) => {
      onFile(f);
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return f && f.type.startsWith("image/") ? URL.createObjectURL(f) : null;
      });
    },
    [onFile]
  );

  const isVideo = file?.type.startsWith("video/");

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) take(f);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      className={clsx(
        "group relative flex h-44 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed p-4 text-center transition",
        over ? "border-accent bg-accent/10" : "border-edge bg-ink-900/50 hover:border-accent/50 hover:bg-ink-850"
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => take(e.target.files?.[0] ?? null)}
      />

      {preview ? (
        <img src={preview} alt={label} className="h-full w-full rounded-lg object-cover" />
      ) : isVideo ? (
        <div className="flex flex-col items-center gap-2">
          <svg viewBox="0 0 24 24" className="h-7 w-7 text-accent" fill="none" stroke="currentColor" strokeWidth="1.6">
            <rect x="3" y="6" width="13" height="12" rx="2" />
            <path d="M16 10l5-3v10l-5-3" />
          </svg>
          <span className="text-xs text-slate-300">{file?.name}</span>
        </div>
      ) : (
        <>
          <svg viewBox="0 0 24 24" className="h-6 w-6 text-slate-600 transition group-hover:text-accent"
            fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round">
            <path d="M12 16V4M8 8l4-4 4 4M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
          </svg>
          <div className="mt-2 text-sm font-medium text-slate-300">{label}</div>
          <div className="mt-0.5 text-[11px] text-slate-500">{hint}</div>
        </>
      )}

      {file && (
        <button
          onClick={(e) => { e.stopPropagation(); take(null); }}
          className="absolute right-2 top-2 rounded-md bg-ink-950/80 px-1.5 py-0.5 text-[11px] text-slate-400 ring-1 ring-edge hover:text-reject"
          aria-label={`Remove ${label}`}
        >
          clear
        </button>
      )}
    </div>
  );
}

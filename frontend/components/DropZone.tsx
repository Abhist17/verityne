"use client";

import { useCallback, useRef, useState } from "react";
import clsx from "clsx";

const KB = 1024;
const fmtSize = (n: number) => (n < KB * KB ? `${(n / KB).toFixed(0)} KB` : `${(n / KB / KB).toFixed(1)} MB`);

/**
 * A file slot. Empty it is a dashed target; filled it becomes the thumbnail
 * itself, so the input column shows the packet you are about to submit rather
 * than a column of identical grey boxes with filenames in them.
 */
export function DropZone({
  label,
  hint,
  accept,
  file,
  onFile,
  className,
}: {
  label: string;
  hint: string;
  accept: string;
  file: File | null;
  onFile: (f: File | null) => void;
  className?: string;
}) {
  const [over, setOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const take = useCallback(
    (f: File | null) => {
      onFile(f);
      setPreview((old) => {
        if (old) URL.revokeObjectURL(old);
        return f && (f.type.startsWith("image/") || f.type.startsWith("video/")) ? URL.createObjectURL(f) : null;
      });
    },
    [onFile]
  );

  const isVideo = !!file?.type.startsWith("video/");

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
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
      aria-label={file ? `${label}: ${file.name}. Click to replace.` : `Add ${label}`}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      className={clsx(
        "group relative cursor-pointer overflow-hidden rounded-lg transition-colors duration-150",
        "focus-visible:outline focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-accent",
        file
          ? "border border-edge bg-ink-900"
          : "border border-dashed bg-ink-900/60 " +
              (over ? "border-accent bg-accent/[0.07]" : "border-edge hover:border-edge-strong hover:bg-ink-850"),
        className
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => take(e.target.files?.[0] ?? null)}
      />

      {file ? (
        <>
          {preview &&
            (isVideo ? (
              <video src={preview} muted playsInline className="h-full w-full object-cover" />
            ) : (
              <img src={preview} alt="" className="h-full w-full object-cover" />
            ))}
          {/* Scrim only under the caption, so the thumbnail stays legible. */}
          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-ink-950 via-ink-950/85 to-transparent px-2.5 pb-1.5 pt-6">
            <div className="truncate text-xs font-medium text-slate-200">{file.name}</div>
            <div className="num text-2xs text-slate-500">
              {label} · {fmtSize(file.size)}
            </div>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              take(null);
            }}
            className="absolute right-1.5 top-1.5 rounded bg-ink-950/75 px-1.5 py-0.5 text-2xs text-slate-400 ring-1 ring-edge backdrop-blur-sm transition-colors hover:text-reject"
            aria-label={`Remove ${label}`}
          >
            clear
          </button>
        </>
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-1 p-3 text-center">
          <svg
            viewBox="0 0 24 24"
            className={clsx("h-4 w-4 transition-colors", over ? "text-accent" : "text-slate-600 group-hover:text-slate-400")}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M12 15V4M8.5 7.5L12 4l3.5 3.5M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3" />
          </svg>
          <div className="text-xs font-medium text-slate-300">{label}</div>
          <div className="text-2xs leading-tight text-slate-600">{hint}</div>
        </div>
      )}
    </div>
  );
}

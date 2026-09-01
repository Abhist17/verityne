import Link from "next/link";

/**
 * The close. Pages used to stop dead at their last row, which reads as a page
 * that ran out rather than one that finished.
 *
 * It carries the one claim this project wants a reader leaving with — that the
 * numbers have files behind them and the files are in the repository — rather
 * than the usual footer furniture of a company that does not exist.
 */

const EVIDENCE: [string, string][] = [
  ["metrics.json", "held-out report"],
  ["corrections.json", "the audit trail"],
  ["behavioral.json", "detector 6"],
  ["real_faces.json", "third-party fakes"],
  ["linkage_lfw.json", "linkage threshold"],
  ["thresholds.json", "why they are not constants"],
];

const REPO = "https://github.com/Abhist17/verityne";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-edge">
      <div className="mx-auto grid w-full max-w-[1180px] gap-10 px-6 py-10 wide:grid-cols-[minmax(0,1fr)_auto]">
        <div className="max-w-[46ch]">
          <div className="flex items-center gap-2">
            <svg viewBox="0 0 24 24" className="h-[15px] w-[15px] text-accent" fill="none"
                 stroke="currentColor" strokeWidth="2" aria-hidden>
              <path d="M12 3l7 3v6c0 4.2-2.9 7.7-7 9-4.1-1.3-7-4.8-7-9V6l7-3z" strokeLinejoin="round" />
              <path d="M9 12l2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="text-sm font-medium text-slate-300">Verityne</span>
          </div>
          {/* No counts here on purpose. The number of corrections changes every
              time one is found, and a hard-coded "nine" in a footer is exactly
              the kind of figure that goes stale silently — which is a thing this
              project has a page about. The count lives in one place, generated
              from the evidence, and this links to it. */}
          <p className="mt-2.5 text-xs leading-relaxed text-slate-500">
            Deepfake-aware KYC verification, and a published record of every belief this project
            measured and lost — the ones still open included.
          </p>
          <p className="mt-3 text-2xs leading-relaxed text-slate-600">
            Every figure in the dashboard is read from a committed evidence file, and the test
            suite fails if the two disagree. Reproduce with{" "}
            <span className="num text-slate-500">make pipeline</span> and{" "}
            <span className="num text-slate-500">make real</span>.
          </p>
        </div>

        <div className="flex flex-wrap gap-x-12 gap-y-6">
          <div>
            <div className="label mb-2">Evidence</div>
            <ul className="space-y-1">
              {EVIDENCE.map(([file, what]) => (
                <li key={file} className="text-2xs">
                  <span className="num text-slate-500">{file}</span>
                  <span className="ml-2 text-slate-600">{what}</span>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="label mb-2">Source</div>
            <ul className="space-y-1 text-2xs">
              <li>
                <a href={REPO} target="_blank" rel="noreferrer"
                   className="text-slate-500 transition-colors hover:text-slate-300">
                  github.com/Abhist17/verityne
                </a>
              </li>
              <li>
                <Link href="/corrections"
                      className="text-slate-500 transition-colors hover:text-slate-300">
                  What we got wrong
                </Link>
              </li>
              <li>
                <a href="/api/docs" target="_blank" rel="noreferrer"
                   className="text-slate-500 transition-colors hover:text-slate-300">
                  API reference
                </a>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </footer>
  );
}

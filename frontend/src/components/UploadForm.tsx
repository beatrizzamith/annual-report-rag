import { useState } from "react";

import { ApiError, uploadReport } from "../api";

const YEAR_PATTERN = /20\d{2}/;
const WORD_SEPARATORS = /[-_\s]+/;

/**
 * Best-effort guess of a report's company and fiscal year from its filename,
 * for example "shell-annual-report-2025.pdf" -> Shell, 2025.
 *
 * @param filename - The uploaded file's original name.
 * @returns A guessed company and year, either of which may be empty if no
 *   reliable match could be made.
 */
function guessMetadataFromFilename(filename: string): { company: string; year: number | "" } {
  const yearMatch = filename.match(YEAR_PATTERN);
  const year = yearMatch ? Number(yearMatch[0]) : "";
  const base = filename.replace(/\.[^./]+$/, "");
  const firstToken = base.split(WORD_SEPARATORS)[0] ?? "";
  const company = firstToken ? firstToken.charAt(0).toUpperCase() + firstToken.slice(1) : "";
  return { company, year };
}

const inputClasses =
  "border border-gray-200 rounded-lg px-3 py-2 text-sm outline-none transition-shadow focus:ring-2 focus:ring-brand-500/30 focus:border-brand-400";

/** Upload form: file picker plus company/year fields, pre-filled from the filename. */
export default function UploadForm({ onUploaded }: { onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [company, setCompany] = useState("");
  const [year, setYear] = useState<number | "">("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] ?? null;
    setFile(selected);
    if (selected) {
      const guess = guessMetadataFromFilename(selected.name);
      setCompany(guess.company);
      setYear(guess.year);
    }
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!file || !company || !year) {
      setError("Choose a PDF file, and fill in the company and year.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await uploadReport(file, company, Number(year));
      setFile(null);
      setCompany("");
      setYear("");
      onUploaded();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="rounded-xl border border-gray-200 bg-white shadow-card p-5">
      <h2 className="text-sm font-semibold text-gray-800 mb-3">Upload an annual report</h2>
      <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-xs font-medium text-gray-500 mb-1.5">PDF file</label>
          <label
            className={`flex items-center justify-center gap-2 text-sm rounded-lg border-2 border-dashed px-3 py-2 cursor-pointer transition-colors ${
              file
                ? "border-brand-300 bg-brand-50 text-brand-700"
                : "border-gray-200 text-gray-400 hover:border-gray-300 hover:text-gray-500"
            }`}
          >
            <span className="truncate">{file ? file.name : "Choose a PDF..."}</span>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange}
              className="hidden"
            />
          </label>
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1.5">Company</label>
          <input
            value={company}
            onChange={(event) => setCompany(event.target.value)}
            placeholder="Shell"
            className={`${inputClasses} w-36`}
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1.5">Fiscal year</label>
          <input
            type="number"
            value={year}
            onChange={(event) => setYear(event.target.value ? Number(event.target.value) : "")}
            placeholder="2025"
            className={`${inputClasses} w-24`}
          />
        </div>
        <button
          type="submit"
          disabled={submitting}
          className="bg-brand-600 hover:bg-brand-700 text-white rounded-lg px-5 py-2 text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? "Uploading…" : "Upload"}
        </button>
        {error && <p className="text-sm text-rose-600 w-full">{error}</p>}
      </form>
    </div>
  );
}

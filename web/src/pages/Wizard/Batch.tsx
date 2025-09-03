import { useEffect, useState } from "react";
import { API_BASE } from "../../lib/api";
import { useJobs } from "../../components/Jobs";
import type { JobEntry } from "@/types/jobs";

interface Props {
  primaryKey: string;
  embedFields: string[];
  mode: "api" | "excel";
  excelToken: string | null;
  excelIdCol: string | null;
  transcripts: File[];
  keepRatio: number;
  setKeepRatio: (n: number) => void;
  brief: string;
  setBrief: (s: string) => void;
  withTitles: boolean;
  setWithTitles: (b: boolean) => void;
}

export default function Batch({
  primaryKey,
  embedFields,
  mode,
  excelToken,
  excelIdCol,
  transcripts,
  keepRatio,
  setKeepRatio,
  brief,
  setBrief,
  withTitles,
  setWithTitles,
}: Props) {
  const [uid, setUid] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { addJob, jobs } = useJobs();
  const job: JobEntry | null = jobs.find((j) => j.uid === uid) || null;
  const [completed, setCompleted] = useState(false);

  useEffect(() => {
    if (uid && !job) setCompleted(true);
  }, [uid, job]);

  const launch = async () => {
    const form = new FormData();
    form.append("primary_key", primaryKey);
    embedFields.forEach((f) => form.append("embed_fields", f));
    form.append("mode", mode);
    if (mode === "excel" && excelToken) {
      form.append("excel_token", excelToken);
      if (excelIdCol) form.append("excel_id_col", excelIdCol);
    }
    transcripts.forEach((f) => form.append("transcripts", f));
    form.append("keep_ratio", String(keepRatio));
    if (brief) form.append("brief", brief);
    form.append("with_titles", String(withTitles));
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/segment/batch`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as { uid: string; status: string };
      setUid(json.uid);
      addJob(json.uid);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div>
          <label className="mr-2">Keep ratio:</label>
          <input
            type="number"
            step="0.1"
            value={keepRatio}
            onChange={(e) => setKeepRatio(parseFloat(e.target.value))}
            className="w-20 border p-1"
          />
        </div>
        <div>
          <label className="mr-2">Brief:</label>
          <input
            type="text"
            value={brief}
            onChange={(e) => setBrief(e.target.value)}
            className="border p-1"
          />
        </div>
        <div>
          <label className="mr-2">With titles:</label>
          <input
            type="checkbox"
            checked={withTitles}
            onChange={(e) => setWithTitles(e.target.checked)}
          />
        </div>
      </div>
      <button
        className="rounded bg-primary px-3 py-1 text-primary-foreground disabled:opacity-50"
        disabled={loading || !!uid}
        onClick={launch}
      >
        {loading ? "Launching…" : "Run batch"}
      </button>
      {error && <p className="text-red-600">{error}</p>}
      {job && (
        <p>
          Status: {job.status}
          {job.status === "running" &&
          "progress" in job &&
          "total" in job &&
          job.progress !== undefined &&
          job.total !== undefined
            ? ` – ${job.progress}/${job.total}`
            : job.note
            ? ` – ${job.note}`
            : ""}
        </p>
      )}
      {completed && <p>Batch completed. See Downloads panel.</p>}
    </div>
  );
}

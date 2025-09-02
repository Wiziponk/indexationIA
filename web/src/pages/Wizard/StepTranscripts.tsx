import { useState } from "react";
import { API_BASE } from "../../lib/api";

type PrepareResponse = {
  primary_key: string;
  count_included: number;
  count_excluded: number;
  excluded_ids: string[];
  sample_ids: string[];
};

type Props = {
  primaryKey: string;
  embedFields: string[];
  mode: "api" | "excel";
  excelToken: string | null;
  excelIdCol: string | null;
};

export default function StepTranscripts({
  primaryKey,
  embedFields,
  mode,
  excelToken,
  excelIdCol,
}: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [result, setResult] = useState<PrepareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const onPrepare = async () => {
    const form = new FormData();
    form.append("primary_key", primaryKey);
    embedFields.forEach((f) => form.append("embed_fields", f));
    form.append("mode", mode);
    if (mode === "excel" && excelToken) {
      form.append("excel_token", excelToken);
      if (excelIdCol) form.append("excel_id_col", excelIdCol);
    }
    files.forEach((f) => form.append("transcripts", f));
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/segment/prepare`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as PrepareResponse;
      setResult(json);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <input
          type="file"
          accept=".docx"
          multiple
          onChange={(e) => setFiles(Array.from(e.target.files || []))}
        />
      </div>
      <button
        className="rounded bg-primary px-3 py-1 text-primary-foreground disabled:opacity-50"
        disabled={files.length === 0 || loading}
        onClick={onPrepare}
      >
        {loading ? "Preparing…" : "Prepare"}
      </button>
      {error && <p className="text-red-600">{error}</p>}
      {result && (
        <div className="space-y-2">
          <p>
            Included: {result.count_included} • Excluded: {result.count_excluded}
          </p>
          <div>
            <label className="mr-2">Sample IDs:</label>
            <select className="border p-1">
              {result.sample_ids.map((id) => (
                <option key={id} value={id}>
                  {id}
                </option>
              ))}
            </select>
          </div>
        </div>
      )}
    </div>
  );
}


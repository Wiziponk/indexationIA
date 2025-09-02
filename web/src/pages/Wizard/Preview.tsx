import { useState } from "react";
import DataTable from "../../components/DataTable";
import { API_BASE } from "../../lib/api";

interface ClipSegment {
  score?: number;
  title?: string;
  summary?: string;
  text?: string;
}

interface SegmentPreviewResponse {
  pk_value: string;
  segments: ClipSegment[];
  n_segments: number;
  clip_dim: number;
  has_program_embedding: boolean;
}

interface Props {
  primaryKey: string;
  embedFields: string[];
  mode: "api" | "excel";
  excelToken: string | null;
  excelIdCol: string | null;
  transcripts: File[];
  sampleIds: string[];
  sampleId: string;
  setSampleId: (id: string) => void;
  keepRatio: number;
  setKeepRatio: (n: number) => void;
  brief: string;
  setBrief: (s: string) => void;
  withTitles: boolean;
  setWithTitles: (b: boolean) => void;
  setPreviewed: (ok: boolean) => void;
}

export default function Preview({
  primaryKey,
  embedFields,
  mode,
  excelToken,
  excelIdCol,
  transcripts,
  sampleIds,
  sampleId,
  setSampleId,
  keepRatio,
  setKeepRatio,
  brief,
  setBrief,
  withTitles,
  setWithTitles,
  setPreviewed,
}: Props) {
  const [result, setResult] = useState<SegmentPreviewResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onPreview = async () => {
    const form = new FormData();
    form.append("primary_key", primaryKey);
    embedFields.forEach((f) => form.append("embed_fields", f));
    form.append("mode", mode);
    if (mode === "excel" && excelToken) {
      form.append("excel_token", excelToken);
      if (excelIdCol) form.append("excel_id_col", excelIdCol);
    }
    transcripts.forEach((f) => form.append("transcripts", f));
    form.append("sample_pk_value", sampleId);
    form.append("keep_ratio", String(keepRatio));
    if (brief) form.append("brief", brief);
    form.append("with_titles", String(withTitles));
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/segment/preview`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as SegmentPreviewResponse;
      setResult(json);
      setPreviewed(true);
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
          <label className="mr-2">Sample ID:</label>
          <select
            className="border p-1"
            value={sampleId}
            onChange={(e) => setSampleId(e.target.value)}
          >
            {sampleIds.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </div>
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
        disabled={loading || !sampleId}
        onClick={onPreview}
      >
        {loading ? "Previewing…" : "Preview"}
      </button>
      {error && <p className="text-red-600">{error}</p>}
      {result && (
        <DataTable
          data={result.segments as Record<string, unknown>[]}
          columns={[
            { key: "score", label: "Score" },
            { key: "title", label: "Title" },
            { key: "summary", label: "Summary" },
            { key: "text", label: "Text" },
          ]}
        />
      )}
    </div>
  );
}

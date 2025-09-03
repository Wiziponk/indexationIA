import { Dispatch, SetStateAction, useState } from "react";
import { API_BASE } from "../../lib/api";

type UploadResponse = {
  token: string;
  columns: string[];
  preview: Record<string, string>[];
};

type Props = {
  mode: "api" | "excel";
  setMode: Dispatch<SetStateAction<"api" | "excel">>;
  token: string | null;
  setToken: Dispatch<SetStateAction<string | null>>;
  idCol: string | null;
  setIdCol: Dispatch<SetStateAction<string | null>>;
  columns: string[];
  setColumns: Dispatch<SetStateAction<string[]>>;
  fileName: string | null;
  setFileName: Dispatch<SetStateAction<string | null>>;
};

export default function StepScope({
  mode,
  setMode,
  token,
  setToken,
  idCol,
  setIdCol,
  columns,
  setColumns,
  fileName,
  setFileName,
}: Props) {
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append("excel", file);
    setUploading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/upload-ids`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) throw new Error(await res.text());
      const json = (await res.json()) as UploadResponse;
      setToken(json.token);
      setColumns(json.columns);
      setFileName(file.name);
      if (json.columns.length === 1) setIdCol(json.columns[0]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="mr-4">
          <input
            type="radio"
            name="mode"
            value="api"
            checked={mode === "api"}
            onChange={() => setMode("api")}
            className="mr-1"
          />
          Full API
        </label>
        <label>
          <input
            type="radio"
            name="mode"
            value="excel"
            checked={mode === "excel"}
            onChange={() => setMode("excel")}
            className="mr-1"
          />
          Excel/CSV
        </label>
      </div>

      {mode === "excel" && (
        <div className="space-y-2">
          <input type="file" accept=".csv,.xlsx,.xls" onChange={onFile} />
          {fileName && <p>Selected: {fileName}</p>}
          {uploading && <p>Uploading…</p>}
          {error && <p className="text-red-600">{error}</p>}
          {token && (
            <div>
              <label className="mr-2">ID column:</label>
              <select
                className="border p-1"
                value={idCol || ""}
                onChange={(e) => setIdCol(e.target.value)}
              >
                <option value="">Select…</option>
                {columns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


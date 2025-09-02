import { useEffect, useState } from "react";
import { api, API_BASE } from "../lib/api";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { useToast } from "../components/Toasts";

interface Dataset {
  uid: string;
  label?: string;
  raw_path: string;
  emb_path: string;
  created_at?: string;
  config: Record<string, unknown>;
}

export default function Datasets() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const toast = useToast();

  useEffect(() => {
    api<Dataset[]>("/datasets")
      .then((d) => setDatasets(d))
      .catch((e) => toast(e.message))
      .finally(() => setLoading(false));
  }, [toast]);

  const updateLabel = (uid: string, label: string) => {
    api<Dataset>(`/datasets/${uid}`, {
      method: "PUT",
      body: JSON.stringify({ label }),
    })
      .then((d) =>
        setDatasets((cur) => cur.map((x) => (x.uid === uid ? d : x)))
      )
      .catch((e) => toast(e.message));
  };

  const remove = (uid: string) => {
    if (!confirm("Delete dataset?")) return;
    api(`/datasets/${uid}`, { method: "DELETE" })
      .then(() => setDatasets((cur) => cur.filter((x) => x.uid !== uid)))
      .catch((e) => toast(e.message));
  };

  const rerun = (uid: string) => {
    api<Dataset>(`/datasets/${uid}/rerun`, { method: "POST" })
      .then((d) => {
        setDatasets((cur) => [d, ...cur]);
        toast(`Re-run created dataset ${d.uid}`);
      })
      .catch((e) => toast(e.message));
  };

  if (loading) return <p>Loading…</p>;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Datasets</h1>
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="border-b p-2 text-left">UID</th>
            <th className="border-b p-2 text-left">Label</th>
            <th className="border-b p-2 text-left">Created</th>
            <th className="border-b p-2 text-left">Downloads</th>
            <th className="border-b p-2"></th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((d) => (
            <tr key={d.uid} className="border-b align-top">
              <td className="p-2 font-mono">{d.uid}</td>
              <td className="p-2">
                <Input
                  value={d.label || ""}
                  onChange={(e) =>
                    setDatasets((cur) =>
                      cur.map((x) =>
                        x.uid === d.uid ? { ...x, label: e.target.value } : x
                      )
                    )
                  }
                  onBlur={(e) => updateLabel(d.uid, e.target.value)}
                />
              </td>
              <td className="p-2">
                {d.created_at
                  ? new Date(d.created_at).toLocaleString()
                  : ""}
              </td>
              <td className="p-2 space-x-2">
                <a
                  className="text-blue-600 underline"
                  href={`${API_BASE}/download/${d.raw_path}`}
                >
                  parquet
                </a>
                <a
                  className="text-blue-600 underline"
                  href={`${API_BASE}/download/${d.emb_path}`}
                >
                  embeddings
                </a>
              </td>
              <td className="p-2 space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => rerun(d.uid)}
                >
                  Re-run
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => remove(d.uid)}
                >
                  Delete
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


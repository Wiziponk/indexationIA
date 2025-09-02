import { useCallback, useMemo, useState } from "react";
import Plot from "react-plotly.js";

import DataTable from "../components/DataTable";
import FileDrop from "../components/FileDrop";
import FormField from "../components/FormField";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";

interface ClusterPoint {
  pk: string;
  clip_index?: number;
  cluster: number;
  x: number;
  y: number;
}

interface ClusterMeta {
  k: number;
  silhouette: number;
}

interface ClusterResponse {
  points: ClusterPoint[];
  meta: ClusterMeta;
}

type Tab = "clips" | "emissions";

export default function Clustering() {
  const [tab, setTab] = useState<Tab>("clips");
  const [files, setFiles] = useState<File[]>([]);
  const [algo, setAlgo] = useState("kmeans");
  const [kChoice, setKChoice] = useState("auto");
  const [proj, setProj] = useState("pca");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ClusterResponse | null>(null);

  const handleFiles = useCallback((fl: FileList) => {
    setFiles(Array.from(fl));
  }, []);

  const run = useCallback(async () => {
    if (!files.length) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("packages", f));
      fd.append("algo_choice", algo);
      fd.append("k_choice", kChoice);
      fd.append("proj_choice", proj);
      const res = await fetch(`/api/cluster/${tab}`, {
        method: "POST",
        body: fd,
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data: ClusterResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [files, algo, kChoice, proj, tab]);

  const summary = useMemo(() => {
    if (!result) return [] as { cluster: number; count: number }[];
    const counts: Record<number, number> = {};
    for (const p of result.points) {
      counts[p.cluster] = (counts[p.cluster] || 0) + 1;
    }
    return Object.entries(counts).map(([cluster, count]) => ({
      cluster: Number(cluster),
      count,
    }));
  }, [result]);

  const exportCsv = useCallback(() => {
    if (!result) return;
    const header =
      tab === "clips" ? "pk,clip_index,cluster,x,y" : "pk,cluster,x,y";
    const rows = result.points
      .map((p) =>
        [
          p.pk,
          tab === "clips" ? String(p.clip_index) : undefined,
          p.cluster,
          p.x,
          p.y,
        ]
          .filter((v) => v !== undefined)
          .join(",")
      )
      .join("\n");
    const blob = new Blob([`${header}\n${rows}`], {
      type: "text/csv",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = tab === "clips" ? "clusters_clips.csv" : "clusters_programmes.csv";
    a.click();
    URL.revokeObjectURL(url);
  }, [result, tab]);

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Clustering</h1>

      <div className="flex gap-2">
        <Button
          variant={tab === "clips" ? "default" : "outline"}
          onClick={() => setTab("clips")}
        >
          By Clips
        </Button>
        <Button
          variant={tab === "emissions" ? "default" : "outline"}
          onClick={() => setTab("emissions")}
        >
          By Programme
        </Button>
      </div>

      <FileDrop onFiles={handleFiles} />
      {files.length > 0 && (
        <ul className="list-disc pl-5 text-sm">
          {files.map((f) => (
            <li key={f.name}>{f.name}</li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap items-end gap-4">
        <FormField label="Algorithm">
          <select
            className="rounded border p-2"
            value={algo}
            onChange={(e) => setAlgo(e.target.value)}
          >
            <option value="kmeans">KMeans</option>
            <option value="dbscan">DBSCAN</option>
          </select>
        </FormField>

        {algo === "kmeans" && (
          <FormField label="k">
            <Input
              value={kChoice}
              onChange={(e) => setKChoice(e.target.value)}
              placeholder="auto"
            />
          </FormField>
        )}

        <FormField label="Projection">
          <select
            className="rounded border p-2"
            value={proj}
            onChange={(e) => setProj(e.target.value)}
          >
            <option value="pca">PCA</option>
            <option value="tsne">t-SNE</option>
          </select>
        </FormField>

        <Button onClick={run} disabled={loading || !files.length}>
          {loading ? "Running…" : "Run"}
        </Button>

        {result && (
          <Button variant="outline" onClick={exportCsv} disabled={!result}>
            Export CSV
          </Button>
        )}
      </div>

      {error && <p className="text-red-600">{error}</p>}

      {result && (
        <div className="space-y-4">
          {result.meta.silhouette >= 0 && (
            <p className="text-sm">
              Silhouette: {result.meta.silhouette.toFixed(3)}
            </p>
          )}
          <Plot
            data={[
              {
                x: result.points.map((p) => p.x),
                y: result.points.map((p) => p.y),
                type: "scattergl",
                mode: "markers",
                marker: {
                  color: result.points.map((p) => p.cluster),
                  colorscale: "Viridis",
                  showscale: false,
                },
                text: result.points.map((p) =>
                  tab === "clips" ? `${p.pk} #${p.clip_index}` : p.pk
                ),
              },
            ]}
            layout={{
              margin: { t: 0, r: 0, l: 40, b: 40 },
              hovermode: "closest",
            }}
            style={{ width: "100%", height: 500 }}
            config={{ displayModeBar: false }}
          />

          <DataTable
            data={summary}
            columns={[
              { key: "cluster", label: "Cluster" },
              { key: "count", label: "Count" },
            ]}
          />
        </div>
      )}
    </div>
  );
}

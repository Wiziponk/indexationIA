import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { useToast } from "../components/Toasts";

interface Project {
  id: number;
  name: string;
  programs: number;
  keep_ratio: number;
  with_titles: boolean;
  brief?: string | null;
}

interface Program {
  id: number;
  pk_value: string;
  num_clips: number;
  last_zip_path?: string | null;
}

interface Clip {
  id: number;
  idx: number;
  start?: number;
  end?: number;
  score?: number;
  title?: string | null;
  summary?: string | null;
  text?: string | null;
}

interface ProgramDetail {
  program: Program & { project_id: number };
  clips: Clip[];
}

export default function Library() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<number | null>(null);
  const [programs, setPrograms] = useState<Program[]>([]);
  const [selectedProgram, setSelectedProgram] = useState<number | null>(null);
  const [detail, setDetail] = useState<ProgramDetail | null>(null);
  const [loadingProjects, setLoadingProjects] = useState(true);
  const toast = useToast();

  // Rerun knobs
  const [keepRatio, setKeepRatio] = useState("");
  const [withTitles, setWithTitles] = useState(false);
  const [brief, setBrief] = useState("");

  useEffect(() => {
    api<{ projects: Project[] }>("/db/projects")
      .then((d) => setProjects(d.projects))
      .catch((e) => toast(e.message))
      .finally(() => setLoadingProjects(false));
  }, [toast]);

  useEffect(() => {
    if (selectedProject === null) return;
    setPrograms([]);
    setDetail(null);
    setSelectedProgram(null);
    api<{ programs: Program[] }>(`/db/programs?project_id=${selectedProject}`)
      .then((d) => setPrograms(d.programs))
      .catch((e) => toast(e.message));
  }, [selectedProject, toast]);

  useEffect(() => {
    if (selectedProgram === null) return;
    api<ProgramDetail>(`/db/programs/${selectedProgram}`)
      .then((d) => {
        setDetail(d);
        setKeepRatio("");
        const proj = projects.find((p) => p.id === d.program.project_id);
        setWithTitles(proj ? proj.with_titles : false);
        setBrief("");
      })
      .catch((e) => toast(e.message));
  }, [selectedProgram, toast, projects]);

  const updateClip = (clip: Clip) => {
    api(`/db/clips/${clip.id}`, {
      method: "PATCH",
      body: JSON.stringify({
        title: clip.title,
        summary: clip.summary,
        score: clip.score,
      }),
    }).catch((e) => toast(e.message));
  };

  const rerun = () => {
    if (selectedProgram === null) return;
    const payload: Record<string, unknown> = {};
    if (keepRatio) payload.keep_ratio = parseFloat(keepRatio);
    payload.with_titles = withTitles;
    if (brief.trim()) payload.brief = brief;
    api<{ ok: boolean; num_clips: number; zip: string }>(
      `/db/programs/${selectedProgram}/rerun`,
      {
        method: "POST",
        body: JSON.stringify(payload),
      }
    )
      .then((res) => {
        toast("Rerun completed");
        setPrograms((cur) =>
          cur.map((p) =>
            p.id === selectedProgram
              ? { ...p, num_clips: res.num_clips, last_zip_path: res.zip }
              : p
          )
        );
        return api<ProgramDetail>(`/db/programs/${selectedProgram}`);
      })
      .then((d) => setDetail(d))
      .catch((e) => toast(e.message));
  };

  if (loadingProjects) return <p>Loading…</p>;

  return (
    <div className="space-y-4">
      <h1 className="text-xl font-bold">Library</h1>
      <div className="flex gap-4">
        {/* Projects */}
        <div className="w-1/4 border rounded p-2">
          <h2 className="font-semibold mb-2">Projects</h2>
          <ul>
            {projects.map((p) => (
              <li key={p.id}>
                <button
                  className={`w-full text-left px-2 py-1 rounded hover:bg-muted ${
                    p.id === selectedProject ? "bg-muted" : ""
                  }`}
                  onClick={() => setSelectedProject(p.id)}
                >
                  {p.name} <span className="text-xs text-gray-500">({p.programs})</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Programs */}
        <div className="w-1/4 border rounded p-2">
          <h2 className="font-semibold mb-2">Programmes</h2>
          <ul>
            {programs.map((pr) => (
              <li key={pr.id}>
                <button
                  className={`w-full text-left px-2 py-1 rounded hover:bg-muted ${
                    pr.id === selectedProgram ? "bg-muted" : ""
                  }`}
                  onClick={() => setSelectedProgram(pr.id)}
                >
                  {pr.pk_value} ({pr.num_clips})
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Clips */}
        <div className="flex-1 border rounded p-2">
          {detail ? (
            <div className="space-y-4">
              <div>
                <h2 className="font-semibold">
                  {detail.program.pk_value} ({detail.program.num_clips} clips)
                </h2>
                {detail.program.last_zip_path && (
                  <a
                    href={detail.program.last_zip_path}
                    className="text-sm text-blue-600 underline"
                  >
                    Download ZIP
                  </a>
                )}
              </div>

              <div className="flex items-end gap-2">
                <div>
                  <label className="block text-sm">Keep ratio</label>
                  <Input
                    value={keepRatio}
                    onChange={(e) => setKeepRatio(e.target.value)}
                    placeholder={projects.find((p) => p.id === detail.program.project_id)?.keep_ratio.toString() || ""}
                  />
                </div>
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={withTitles}
                    onChange={(e) => setWithTitles(e.target.checked)}
                  />
                  <label className="text-sm">With titles</label>
                </div>
                <div className="flex-1">
                  <label className="block text-sm">Brief</label>
                  <Input
                    value={brief}
                    onChange={(e) => setBrief(e.target.value)}
                    placeholder={
                      projects.find((p) => p.id === detail.program.project_id)?.brief || ""
                    }
                  />
                </div>
                <Button onClick={rerun}>Re-run</Button>
              </div>

              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <th className="border-b p-2 text-left">#</th>
                    <th className="border-b p-2 text-left">Title</th>
                    <th className="border-b p-2 text-left">Summary</th>
                    <th className="border-b p-2 text-left">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {detail.clips.map((c) => (
                    <tr key={c.id} className="border-b align-top">
                      <td className="p-2">{c.idx}</td>
                      <td className="p-2 w-48">
                        <Input
                          value={c.title || ""}
                          onChange={(e) =>
                            setDetail((cur) =>
                              cur
                                ? {
                                    ...cur,
                                    clips: cur.clips.map((x) =>
                                      x.id === c.id
                                        ? { ...x, title: e.target.value }
                                        : x
                                    ),
                                  }
                                : cur
                            )
                          }
                          onBlur={() => {
                            const updated = detail?.clips.find((x) => x.id === c.id);
                            if (updated) updateClip(updated);
                          }}
                        />
                      </td>
                      <td className="p-2">
                        <textarea
                          className="w-full rounded border px-2 py-1 text-sm"
                          value={c.summary || ""}
                          onChange={(e) =>
                            setDetail((cur) =>
                              cur
                                ? {
                                    ...cur,
                                    clips: cur.clips.map((x) =>
                                      x.id === c.id
                                        ? { ...x, summary: e.target.value }
                                        : x
                                    ),
                                  }
                                : cur
                            )
                          }
                          onBlur={() => {
                            const updated = detail?.clips.find((x) => x.id === c.id);
                            if (updated) updateClip(updated);
                          }}
                        />
                      </td>
                      <td className="p-2 w-24">
                        <Input
                          type="number"
                          value={c.score?.toString() || ""}
                          onChange={(e) =>
                            setDetail((cur) => {
                              const val = e.target.value;
                              return cur
                                ? {
                                    ...cur,
                                    clips: cur.clips.map((x) =>
                                      x.id === c.id
                                        ? {
                                            ...x,
                                            score:
                                              val === "" ? undefined : parseFloat(val),
                                          }
                                        : x
                                    ),
                                  }
                                : cur;
                            })
                          }
                          onBlur={() => {
                            const updated = detail?.clips.find((x) => x.id === c.id);
                            if (updated) updateClip(updated);
                          }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p>Select a programme to view clips</p>
          )}
        </div>
      </div>
    </div>
  );
}

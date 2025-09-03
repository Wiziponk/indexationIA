import { useEffect } from "react";
import { API_BASE } from "../lib/api";
import { useJobs, type JobStatus } from "../components/Jobs";

export default function useJobPolling(uid: string | null) {
  const { updateJob, removeJob, setDownloads } = useJobs();

  useEffect(() => {
    if (!uid) return;
    let delay = 1000;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const res = await fetch(`${API_BASE}/segment/status/${uid}`);
        if (!res.ok) {
          updateJob(uid, {
            status: "error",
            message: `HTTP ${res.status}`,
          });
          return;
        }
        const json = (await res.json()) as JobStatus;
        updateJob(uid, json);
        if (json.status === "running") {
          delay = Math.min(delay * 1.5, 10000);
          timer = setTimeout(poll, delay);
        } else if (json.status === "done") {
          removeJob(uid);
          setDownloads(json.result);
        }
      } catch (e) {
        updateJob(uid, { status: "error", message: String(e) });
      }
    };
    poll();
    return () => clearTimeout(timer);
  }, [uid, updateJob, removeJob, setDownloads]);
}

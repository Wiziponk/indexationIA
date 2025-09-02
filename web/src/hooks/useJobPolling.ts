import { useEffect } from "react";
import { API_BASE } from "../lib/api";
import { useJobs, type JobStatus } from "../components/Jobs";

export default function useJobPolling(uid: string | null) {
  const { updateJob, removeJob, setDownloads } = useJobs();

  useEffect(() => {
    if (!uid) return;
    const interval = setInterval(async () => {
      const res = await fetch(`${API_BASE}/segment/status/${uid}`);
      if (!res.ok) return;
      const json = (await res.json()) as JobStatus;
      updateJob(uid, json);
      if (json.status === "done") {
        clearInterval(interval);
        removeJob(uid);
        if (json.result) setDownloads(json.result);
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [uid, updateJob, removeJob, setDownloads]);
}

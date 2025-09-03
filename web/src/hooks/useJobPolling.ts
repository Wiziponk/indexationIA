import { useEffect } from "react";
import { useJobs } from "../components/Jobs";
import { fetchJobStatus } from "../api/jobs";
import type { StatusResponse } from "@/types/jobs";

export default function useJobPolling(uid: string | null) {
  const { updateJob, removeJob, setDownloads } = useJobs();

  useEffect(() => {
    if (!uid) return;
    let delay = 1000;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const json: StatusResponse = await fetchJobStatus(uid);
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

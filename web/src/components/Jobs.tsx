import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";
import DownloadLink from "./DownloadLink";
import useJobPolling from "../hooks/useJobPolling";

export interface SegmentBatchResult {
  uid: string;
  count: number;
  master_zip: string;
  zips: string[];
}

export interface JobStatus {
  status: string;
  note?: string;
  result?: SegmentBatchResult;
}

interface JobEntry extends JobStatus {
  uid: string;
}

interface JobsContextValue {
  jobs: JobEntry[];
  addJob: (uid: string) => void;
  updateJob: (uid: string, job: JobStatus) => void;
  removeJob: (uid: string) => void;
  setDownloads: (res: SegmentBatchResult) => void;
}

const JobsContext = createContext<JobsContextValue>({
  jobs: [],
  addJob: () => {},
  updateJob: () => {},
  removeJob: () => {},
  setDownloads: () => {},
});

export function useJobs() {
  return useContext(JobsContext);
}

export function JobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const [downloads, setDownloads] = useState<SegmentBatchResult | null>(null);

  const addJob = (uid: string) =>
    setJobs((cur) => [...cur, { uid, status: "pending" }]);

  const updateJob = (uid: string, job: JobStatus) =>
    setJobs((cur) => cur.map((j) => (j.uid === uid ? { ...j, ...job } : j)));

  const removeJob = (uid: string) =>
    setJobs((cur) => cur.filter((j) => j.uid !== uid));

  return (
    <JobsContext.Provider
      value={{ jobs, addJob, updateJob, removeJob, setDownloads }}
    >
      {children}
      {jobs.map((j) => (
        <JobWatcher key={j.uid} uid={j.uid} />
      ))}
      <JobsDrawer jobs={jobs} />
      {downloads && <DownloadsPanel result={downloads} />}
    </JobsContext.Provider>
  );
}

function JobsDrawer({ jobs }: { jobs: JobEntry[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="fixed bottom-4 left-4 space-y-2 text-sm">
      <button
        className="rounded bg-primary px-3 py-1 text-primary-foreground"
        onClick={() => setOpen((o) => !o)}
      >
        Jobs ({jobs.length})
      </button>
      {open && jobs.length > 0 && (
        <div className="rounded border bg-background p-2 shadow">
          <ul className="space-y-1">
            {jobs.map((j) => (
              <li key={j.uid}>
                {j.status}
                {j.note ? ` – ${j.note}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function JobWatcher({ uid }: { uid: string }) {
  useJobPolling(uid);
  return null;
}

function DownloadsPanel({ result }: { result: SegmentBatchResult }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="fixed bottom-4 right-4 space-y-2 text-sm">
      <button
        className="rounded bg-primary px-3 py-1 text-primary-foreground"
        onClick={() => setOpen((o) => !o)}
      >
        Downloads
      </button>
      {open && (
        <div className="max-h-60 w-64 overflow-auto rounded border bg-background p-2 shadow">
          <div>
            <DownloadLink href={result.master_zip} />
          </div>
          <ul className="list-disc pl-4">
            {result.zips.map((z) => (
              <li key={z}>
                <DownloadLink href={z} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

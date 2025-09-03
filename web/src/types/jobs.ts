export type ZipInfo = { programme_id: string; path: string };
export type BatchResult = {
  uid: string;
  count: number;
  master_zip: string;
  zips: ZipInfo[];
};

export type StatusRunning = {
  status: "running";
  progress?: number;
  total?: number;
  message?: string;
};
export type StatusError = { status: "error"; message: string };
export type StatusNotFound = { status: "not_found"; message: string };
export type StatusDone = { status: "done"; result: BatchResult };

export type StatusResponse =
  | StatusRunning
  | StatusError
  | StatusNotFound
  | StatusDone;

export type JobEntry = StatusResponse & {
  uid: string;
  note?: string;
  createdAt?: number;
  updatedAt?: number;
};

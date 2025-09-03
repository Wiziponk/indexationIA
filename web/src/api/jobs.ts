import type { StatusResponse } from "@/types/jobs";

export async function fetchJobStatus(uid: string): Promise<StatusResponse> {
  const res = await fetch(`/api/segment/status/${uid}`);
  if (!res.ok) throw new Error(`Status ${res.status}`);
  return res.json() as Promise<StatusResponse>;
}

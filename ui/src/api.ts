import axios from "axios";
import type { Tool, JobStatus } from "./types";

const client = axios.create({ baseURL: "/", timeout: 15000 });

export async function fetchTools(): Promise<Tool[]> {
  const res = await client.get<Tool[]>("/api/tools");
  return res.data;
}

export async function submitResearch(
  query: string,
  tools: string[],
  outputMode: "chat" | "report"
): Promise<{ job_id: string; status: string }> {
  const res = await client.post("/api/research", {
    query,
    tools,
    output_mode: outputMode,
  });
  return res.data;
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  const res = await client.get<JobStatus>(`/api/research/${jobId}`);
  return res.data;
}

export async function cancelJob(jobId: string): Promise<void> {
  await client.delete(`/api/research/${jobId}`);
}

export async function submitFeedback(
  queryId: string,
  rating: "thumbs_up" | "thumbs_down",
  comment: string = ""
): Promise<void> {
  await client.post("/api/feedback", {
    query_id: queryId,
    rating,
    comment,
  });
}

export function streamJob(
  jobId: string,
  onEvent: (event: { type: string; node?: string; result?: string; error?: string }) => void,
  onError: (err: Error) => void
): () => void {
  const es = new EventSource(`/api/research/${jobId}/stream`);
  let receivedTerminal = false;

  es.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data);
      onEvent(event);
      if (["completed", "failed", "cancelled"].includes(event.type)) {
        receivedTerminal = true;
        es.close();
      }
    } catch {
      // Ignore unparseable messages (keepalives)
    }
  };

  es.onerror = () => {
    if (receivedTerminal) return;
    es.close();
    onError(new Error("SSE connection lost"));
  };

  // Return cleanup function
  return () => es.close();
}

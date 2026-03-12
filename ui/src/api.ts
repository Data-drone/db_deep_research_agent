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
  outputMode: "chat" | "report",
  responseMode: "quick" | "research",
  sessionId?: string
): Promise<{ job_id: string; status: string; session_id: string }> {
  const res = await client.post("/api/research", {
    query,
    tools,
    output_mode: outputMode,
    response_mode: responseMode,
    session_id: sessionId,
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
  onEvent: (event: {
    type: string;
    node?: string;
    result?: string;
    error?: string;
    content?: string;
    columns?: { name: string; type: string }[];
    rows?: string[][];
    sql?: string;
    chart?: { type: string; x: string; y: string[]; title?: string };
  }) => void,
  onError: (err: Error) => void
): () => void {
  let cancelled = false;
  let abortController = new AbortController();

  async function readStream() {
    try {
      const response = await fetch(`/api/research/${jobId}/stream`, {
        credentials: "include",
        signal: abortController.signal,
        headers: { Accept: "text/event-stream" },
      });

      if (!response.ok || !response.body) {
        throw new Error(`Stream failed: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (!cancelled) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const event = JSON.parse(line.slice(6));
            onEvent(event);
            if (["completed", "failed", "cancelled"].includes(event.type)) {
              return; // Terminal event received
            }
          } catch {
            // Ignore unparseable lines (keepalives, malformed)
          }
        }
      }
    } catch (err) {
      if (cancelled) return;
      // Stream failed — fall back to polling
      await pollFallback();
    }
  }

  async function pollFallback() {
    while (!cancelled) {
      try {
        const status = await pollJob(jobId);
        if (status.current_node) {
          onEvent({ type: "node_started", node: status.current_node });
        }
        if (status.status === "completed") {
          onEvent({ type: "completed", result: status.result || "" });
          return;
        }
        if (status.status === "failed") {
          onEvent({ type: "failed", error: status.error || "Research failed." });
          return;
        }
        if (status.status === "cancelled") {
          onEvent({ type: "cancelled" });
          return;
        }
      } catch {
        if (cancelled) return;
        onError(new Error("Failed to check research status."));
        return;
      }
      await new Promise((r) => setTimeout(r, 3000));
    }
  }

  readStream();

  // Return cleanup function
  return () => {
    cancelled = true;
    abortController.abort();
  };
}

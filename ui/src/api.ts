import axios from "axios";
import type { Tool, JobStatus } from "./types";

const client = axios.create({ baseURL: "/", timeout: 15000 });

const POLL_INTERVAL_MS = 3000;
/** Consecutive poll failures tolerated before giving up on a job. */
const MAX_POLL_FAILURES = 5;
/** How long to keep polling a job that never reaches a terminal status.
 *  Past the server's own deadline (a 900 s default time cap plus the reaper's
 *  grace period), so in normal operation we see the real terminal status first
 *  and this only catches a job the server has also lost track of. Without it the
 *  loop polls forever and the composer stays disabled with no way out but
 *  Cancel. */
const POLL_DEADLINE_MS = 20 * 60 * 1000;

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

export async function submitClarification(
  jobId: string,
  clarificationId: string,
  answer: string
): Promise<{ status: string; used_answer?: boolean; job_id?: string }> {
  const res = await client.post(`/api/research/${jobId}/clarify`, {
    clarification_id: clarificationId,
    answer,
  });
  return res.data;
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
    clarification_id?: string;
    question?: string;
    options?: string[];
    best_guess?: string;
    answer?: string;
    token_usage?: { input: number; output: number; scope?: string };
    unverified_citations?: string[];
    timeout_seconds?: number;
  }) => void,
  onError: (err: Error) => void
): () => void {
  let cancelled = false;
  let terminal = false;
  let abortController = new AbortController();

  function dispatch(event: Parameters<typeof onEvent>[0]) {
    if (["completed", "failed", "cancelled"].includes(event.type)) {
      terminal = true;
    }
    onEvent(event);
  }

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
            dispatch(event);
            if (terminal) return;
          } catch {
            // Ignore unparseable lines (keepalives, malformed)
          }
        }
      }
    } catch (err) {
      if (cancelled) return;
      // Stream failed — fall back to polling
      await pollFallback();
      return;
    }

    // The stream closed cleanly without a terminal frame (server restart, load
    // balancer idle timeout, graceful end). That is not an exception, so it
    // would otherwise leave the UI loading forever with nothing polling.
    if (!cancelled && !terminal) {
      await pollFallback();
    }
  }

  async function pollFallback() {
    let failures = 0;
    // Track the id rather than a boolean: a second prompt raised in the same
    // poll session would look identical to the first one still being pending,
    // and would never be dispatched.
    let shownClarificationId: string | null = null;
    const deadline = Date.now() + POLL_DEADLINE_MS;

    while (!cancelled) {
      try {
        const status = await pollJob(jobId);
        failures = 0;

        // Recover a clarification the user still needs to answer. The stream is
        // the normal channel for this, but if it dropped while the backend was
        // parked waiting, polling is the only way the prompt comes back.
        if (status.pending_clarification) {
          const id = status.pending_clarification.clarification_id;
          if (id !== shownClarificationId) {
            shownClarificationId = id;
            dispatch({
              type: "clarification_needed",
              clarification_id: id,
              question: status.pending_clarification.question,
              options: status.pending_clarification.options,
            });
          }
        } else {
          if (shownClarificationId !== null) {
            shownClarificationId = null;
            dispatch({ type: "clarification_resolved" });
          }
          // Only report node progress when no prompt is outstanding — a
          // node_started clears the clarification prompt in the UI.
          if (status.current_node) {
            dispatch({ type: "node_started", node: status.current_node });
          }
        }

        if (status.status === "completed") {
          dispatch({
            type: "completed",
            result: status.result || "",
            token_usage: status.token_usage,
            unverified_citations: status.unverified_citations,
          });
          return;
        }
        if (status.status === "failed") {
          dispatch({ type: "failed", error: status.error || "Research failed." });
          return;
        }
        if (status.status === "cancelled") {
          dispatch({ type: "cancelled" });
          return;
        }
      } catch {
        if (cancelled) return;
        failures += 1;
        // A single 502 from a proxy or a brief blip should not make us abandon
        // a job that is still running server-side.
        if (failures >= MAX_POLL_FAILURES) {
          onError(new Error("Failed to check research status."));
          return;
        }
      }

      if (Date.now() >= deadline) {
        // The job is not gone — we just stop following it. Saying so is the
        // honest state; reporting a failure would be a lie, and staying in the
        // loop leaves the user with a spinner and a disabled composer.
        dispatch({ type: "abandoned" });
        return;
      }
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    }
  }

  readStream();

  // Return cleanup function
  return () => {
    cancelled = true;
    abortController.abort();
  };
}

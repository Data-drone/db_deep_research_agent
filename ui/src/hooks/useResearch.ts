import { useCallback, useRef, useState } from "react";
import { submitResearch, pollJob, cancelJob, submitFeedback } from "../api";
import type { Message, JobStatus, OutputMode } from "../types";

let nextId = 1;
function genId(): string {
  return `msg-${nextId++}`;
}

const POLL_INTERVAL_MS = 1500;

export function useResearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const send = useCallback(
    async (query: string, tools: string[], outputMode: OutputMode) => {
      const userMsg: Message = {
        id: genId(),
        role: "user",
        content: query,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      try {
        const { job_id } = await submitResearch(query, tools, outputMode);
        setCurrentJob({ job_id, status: "pending" });

        pollRef.current = setInterval(async () => {
          try {
            const status = await pollJob(job_id);
            setCurrentJob(status);

            if (
              status.status === "completed" ||
              status.status === "failed" ||
              status.status === "cancelled"
            ) {
              stopPolling();
              setIsLoading(false);

              if (status.status === "completed" && status.result) {
                const assistantMsg: Message = {
                  id: genId(),
                  role: "assistant",
                  content: status.result,
                  timestamp: new Date(),
                  jobId: job_id,
                };
                setMessages((prev) => [...prev, assistantMsg]);
              } else if (status.status === "failed") {
                const errorMsg: Message = {
                  id: genId(),
                  role: "assistant",
                  content: "Research failed. Please try again.",
                  timestamp: new Date(),
                  jobId: job_id,
                };
                setMessages((prev) => [...prev, errorMsg]);
              }
              setCurrentJob(null);
            }
          } catch {
            stopPolling();
            setIsLoading(false);
            setCurrentJob(null);
          }
        }, POLL_INTERVAL_MS);
      } catch {
        setIsLoading(false);
        const errorMsg: Message = {
          id: genId(),
          role: "assistant",
          content: "Failed to submit research query.",
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
      }
    },
    [stopPolling]
  );

  const cancel = useCallback(async () => {
    if (currentJob) {
      try {
        await cancelJob(currentJob.job_id);
      } catch {
        /* ignore */
      }
      stopPolling();
      setIsLoading(false);
      setCurrentJob(null);
    }
  }, [currentJob, stopPolling]);

  const rate = useCallback(
    async (messageId: string, rating: "thumbs_up" | "thumbs_down") => {
      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, rating } : m))
      );
      const msg = messages.find((m) => m.id === messageId);
      if (msg?.jobId) {
        await submitFeedback(msg.jobId, rating);
      }
    },
    [messages]
  );

  return { messages, currentJob, isLoading, send, cancel, rate };
}

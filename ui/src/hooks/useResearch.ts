import { useCallback, useEffect, useRef, useState } from "react";
import { submitResearch, pollJob, cancelJob, submitFeedback } from "../api";
import type { Message, JobStatus, OutputMode } from "../types";

const POLL_INTERVAL_MS = 1500;

export function useResearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeJobIdRef = useRef<string | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mountedRef.current = false;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, []);

  const stopPolling = useCallback(() => {
    activeJobIdRef.current = null;
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  }, []);

  const poll = useCallback(
    async (jobId: string) => {
      if (!mountedRef.current || activeJobIdRef.current !== jobId) return;

      try {
        const status = await pollJob(jobId);
        if (!mountedRef.current || activeJobIdRef.current !== jobId) return;

        setCurrentJob(status);

        const terminal = ["completed", "failed", "cancelled"].includes(
          status.status
        );
        if (terminal) {
          stopPolling();
          setIsLoading(false);

          if (status.status === "completed" && status.result) {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: status.result!,
                timestamp: new Date().toISOString(),
                jobId,
              },
            ]);
          } else if (status.status === "failed") {
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: "Research failed. Please try again.",
                timestamp: new Date().toISOString(),
                jobId,
              },
            ]);
          }
          setCurrentJob(null);
          return;
        }

        // Schedule next poll
        timeoutRef.current = setTimeout(() => poll(jobId), POLL_INTERVAL_MS);
      } catch {
        if (!mountedRef.current || activeJobIdRef.current !== jobId) return;
        stopPolling();
        setIsLoading(false);
        setCurrentJob(null);
        setError("Lost connection while polling for results.");
      }
    },
    [stopPolling]
  );

  const send = useCallback(
    async (query: string, tools: string[], outputMode: OutputMode) => {
      if (isLoading) return;

      setError(null);

      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: query,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);

      // Stop any existing polling before starting new job
      stopPolling();

      try {
        const { job_id } = await submitResearch(query, tools, outputMode);
        if (!mountedRef.current) return;

        activeJobIdRef.current = job_id;
        setCurrentJob({ job_id, status: "pending" });

        // Start polling with setTimeout (not setInterval)
        timeoutRef.current = setTimeout(() => poll(job_id), POLL_INTERVAL_MS);
      } catch {
        if (!mountedRef.current) return;
        setIsLoading(false);
        setError("Failed to submit research query.");
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: "Failed to submit research query.",
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    },
    [isLoading, stopPolling, poll]
  );

  const cancel = useCallback(async () => {
    const jobId = activeJobIdRef.current;
    if (!jobId) return;

    stopPolling();
    setIsLoading(false);
    setCurrentJob(null);

    try {
      await cancelJob(jobId);
    } catch {
      // Best-effort cancellation
    }
  }, [stopPolling]);

  const rate = useCallback(
    async (messageId: string, rating: "thumbs_up" | "thumbs_down") => {
      let targetJobId: string | undefined;

      setMessages((prev) =>
        prev.map((m) => {
          if (m.id === messageId) {
            targetJobId = m.jobId;
            return { ...m, rating };
          }
          return m;
        })
      );

      if (targetJobId) {
        try {
          await submitFeedback(targetJobId, rating);
        } catch {
          // Feedback submission is best-effort
        }
      }
    },
    []
  );

  return { messages, currentJob, isLoading, error, send, cancel, rate };
}

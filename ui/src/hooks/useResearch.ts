import { useCallback, useEffect, useRef, useState } from "react";
import { submitResearch, streamJob, cancelJob, submitFeedback } from "../api";
import type { Message, JobStatus, OutputMode } from "../types";

export function useResearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeJobIdRef = useRef<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    return () => {
      mountedRef.current = false;
      cleanupRef.current?.();
      cleanupRef.current = null;
    };
  }, []);

  const stopStream = useCallback(() => {
    activeJobIdRef.current = null;
    cleanupRef.current?.();
    cleanupRef.current = null;
  }, []);

  const startStream = useCallback(
    (jobId: string) => {
      if (!mountedRef.current) return;

      const cleanup = streamJob(
        jobId,
        (event) => {
          if (!mountedRef.current || activeJobIdRef.current !== jobId) return;

          if (event.type === "node_started" && event.node) {
            setCurrentJob((prev) =>
              prev ? { ...prev, status: "running", current_node: event.node } : prev
            );
          } else if (event.type === "token" && event.content) {
            const tokenText = event.content;
            setMessages((prev) => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.streaming && lastMsg.jobId === jobId) {
                // Append to existing streaming message
                return [
                  ...prev.slice(0, -1),
                  { ...lastMsg, content: lastMsg.content + tokenText },
                ];
              } else {
                // Create new streaming message
                return [
                  ...prev,
                  {
                    id: crypto.randomUUID(),
                    role: "assistant" as const,
                    content: tokenText,
                    timestamp: new Date().toISOString(),
                    jobId,
                    streaming: true,
                  },
                ];
              }
            });
          } else if (event.type === "completed") {
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            setMessages((prev) => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.streaming && lastMsg.jobId === jobId) {
                // Finalize streaming message with full result
                return [
                  ...prev.slice(0, -1),
                  {
                    ...lastMsg,
                    content: event.result || lastMsg.content,
                    streaming: false,
                  },
                ];
              } else if (event.result) {
                // No streaming happened — create message from full result
                return [
                  ...prev,
                  {
                    id: crypto.randomUUID(),
                    role: "assistant" as const,
                    content: event.result,
                    timestamp: new Date().toISOString(),
                    jobId,
                  },
                ];
              }
              return prev;
            });
          } else if (event.type === "failed") {
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            setMessages((prev) => [
              ...prev,
              {
                id: crypto.randomUUID(),
                role: "assistant",
                content: event.error || "Research failed. Please try again.",
                timestamp: new Date().toISOString(),
                jobId,
              },
            ]);
          }
        },
        () => {
          if (!mountedRef.current || activeJobIdRef.current !== jobId) return;
          stopStream();
          setIsLoading(false);
          setCurrentJob(null);
          setError("Lost connection to research stream.");
        }
      );

      cleanupRef.current = cleanup;
    },
    [stopStream]
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

      stopStream();

      try {
        const { job_id } = await submitResearch(query, tools, outputMode);
        if (!mountedRef.current) return;

        activeJobIdRef.current = job_id;
        setCurrentJob({ job_id, status: "pending" });

        startStream(job_id);
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
    [isLoading, stopStream, startStream]
  );

  const cancel = useCallback(async () => {
    const jobId = activeJobIdRef.current;
    if (!jobId) return;

    stopStream();
    setIsLoading(false);
    setCurrentJob(null);

    try {
      await cancelJob(jobId);
    } catch {
      // Best-effort cancellation
    }
  }, [stopStream]);

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

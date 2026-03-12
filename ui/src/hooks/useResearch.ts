import { useCallback, useEffect, useRef, useState } from "react";
import { submitResearch, streamJob, cancelJob, submitFeedback, submitClarification } from "../api";
import type { Message, JobStatus, OutputMode, ResponseMode, TableData, ClarificationRequest } from "../types";

export function useResearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentJob, setCurrentJob] = useState<JobStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clarificationRequest, setClarificationRequest] = useState<ClarificationRequest | null>(null);
  const [clarificationSubmitting, setClarificationSubmitting] = useState(false);

  const activeJobIdRef = useRef<string | null>(null);
  const cleanupRef = useRef<(() => void) | null>(null);
  const mountedRef = useRef(true);
  const sessionIdRef = useRef<string | null>(null);
  const pendingTableDataRef = useRef<TableData | null>(null);

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
            setClarificationRequest(null); // Defensive: clear stale prompts on graph progress
            setCurrentJob((prev) =>
              prev ? { ...prev, status: "running", current_node: event.node } : prev
            );
          } else if (event.type === "token" && event.content) {
            const tokenText = event.content;
            setMessages((prev) => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.streaming && lastMsg.jobId === jobId) {
                return [
                  ...prev.slice(0, -1),
                  { ...lastMsg, content: lastMsg.content + tokenText },
                ];
              } else {
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
          } else if (event.type === "table_data") {
            pendingTableDataRef.current = {
              columns: event.columns || [],
              rows: event.rows || [],
              sql: event.sql,
              chart: event.chart as TableData["chart"],
            };
          } else if (event.type === "clarification_needed") {
            setClarificationRequest({
              jobId,
              clarificationId: event.clarification_id || "",
              question: event.question || "Could you clarify your question?",
              options: event.options || [],
            });
          } else if (event.type === "clarification_resolved" || event.type === "clarification_timeout") {
            setClarificationRequest(null);
            setClarificationSubmitting(false);
            if (event.type === "clarification_timeout" && event.best_guess) {
              setMessages((prev) => [
                ...prev,
                {
                  id: crypto.randomUUID(),
                  role: "assistant" as const,
                  content: `No response received — proceeding with: "${event.best_guess}"`,
                  timestamp: new Date().toISOString(),
                  jobId,
                },
              ]);
            }
          } else if (event.type === "completed") {
            const pendingTableData = pendingTableDataRef.current;
            pendingTableDataRef.current = null;
            const rawScope = event.token_usage?.scope;
            const tokenUsage = event.token_usage
              ? { input: event.token_usage.input, output: event.token_usage.output, scope: (rawScope === "answer" ? "answer" : "job") as "answer" | "job" }
              : undefined;
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            setClarificationRequest(null);
            setClarificationSubmitting(false);
            setMessages((prev) => {
              const lastMsg = prev[prev.length - 1];
              if (lastMsg?.streaming && lastMsg.jobId === jobId) {
                return [
                  ...prev.slice(0, -1),
                  {
                    ...lastMsg,
                    content: event.result || lastMsg.content,
                    streaming: false,
                    tableData: pendingTableData ?? undefined,
                    tokenUsage,
                  },
                ];
              } else if (event.result) {
                return [
                  ...prev,
                  {
                    id: crypto.randomUUID(),
                    role: "assistant" as const,
                    content: event.result,
                    timestamp: new Date().toISOString(),
                    jobId,
                    tableData: pendingTableData ?? undefined,
                    tokenUsage,
                  },
                ];
              }
              return prev;
            });
          } else if (event.type === "failed") {
            stopStream();
            setIsLoading(false);
            setCurrentJob(null);
            setClarificationRequest(null);
            setClarificationSubmitting(false);
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
    async (query: string, tools: string[], outputMode: OutputMode, responseMode: ResponseMode) => {
      if (isLoading) return;

      setError(null);
      pendingTableDataRef.current = null;

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
        const { job_id, session_id } = await submitResearch(
          query, tools, outputMode, responseMode, sessionIdRef.current ?? undefined
        );
        if (!mountedRef.current) return;

        sessionIdRef.current = session_id;
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
    setClarificationRequest(null);
    setClarificationSubmitting(false);

    try {
      await cancelJob(jobId);
    } catch {
      // Best-effort cancellation
    }
  }, [stopStream]);

  const answerClarification = useCallback(async (answer: string) => {
    const req = clarificationRequest;
    if (!req || clarificationSubmitting) return;

    setClarificationSubmitting(true);

    try {
      const result = await submitClarification(req.jobId, req.clarificationId, answer);

      if (result.status === "accepted") {
        setClarificationRequest(null);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "user",
            content: answer,
            timestamp: new Date().toISOString(),
          },
        ]);
      } else {
        // expired / already_answered / cancelled — clear prompt, show info
        setClarificationRequest(null);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant" as const,
            content: "Clarification was not used — the research has already proceeded.",
            timestamp: new Date().toISOString(),
          },
        ]);
      }
    } catch {
      setError("Failed to submit clarification.");
    } finally {
      setClarificationSubmitting(false);
    }
  }, [clarificationRequest, clarificationSubmitting]);

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

  return { messages, currentJob, isLoading, error, clarificationRequest, clarificationSubmitting, send, cancel, answerClarification, rate };
}

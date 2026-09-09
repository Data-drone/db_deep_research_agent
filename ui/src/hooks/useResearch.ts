import { useCallback, useEffect, useRef, useState } from "react";
import { submitResearch, streamJob, cancelJob, submitFeedback, submitClarification } from "../api";
import type { Message, JobStatus, OutputMode, ResponseMode, TableData, ClarificationRequest, TokenScope } from "../types";

const TOKEN_SCOPES: readonly TokenScope[] = [
  "job",
  "answer",
  "synthesizer_only",
  "partial",
  "unavailable",
];

/** Never widen a scope we do not recognise into "job" — that would present a
 *  partial count as a whole-run total. */
function normalizeScope(raw: string | undefined): TokenScope {
  return TOKEN_SCOPES.includes(raw as TokenScope) ? (raw as TokenScope) : "unknown";
}

/** Distinct copy per rejection reason. Collapsing these was actively wrong:
 *  "cancelled" does not mean the research proceeded, and "mismatch" is a bug. */
const CLARIFICATION_REJECTION_COPY: Record<string, string> = {
  already_answered: "Your answer was already recorded.",
  expired: "No answer in time — the research proceeded with the agent's best guess.",
  cancelled: "Research was cancelled.",
};

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

  const messagesRef = useRef<Message[]>([]);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    // Must be set on every mount: under StrictMode the effect runs
    // mount -> cleanup -> mount, so a ref that is only ever set to false
    // leaves every handler in this hook a no-op after the first render.
    mountedRef.current = true;
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
            // Deliberately does NOT clear clarificationRequest. The backend
            // parks on the clarifier node while it waits for an answer, so any
            // node_started seen meanwhile (including synthesized ones from the
            // poll fallback) would wipe a prompt the user still has to answer.
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
            if (!event.clarification_id) {
              // Without an id every answer would come back as a mismatch, so a
              // prompt here would be a dead end. Better to fall through to the
              // backend's timeout and best guess.
              console.warn("Ignoring clarification_needed with no clarification_id");
              return;
            }
            setClarificationRequest({
              jobId,
              clarificationId: event.clarification_id,
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
                  kind: "notice" as const,
                },
              ]);
            }
          } else if (event.type === "completed") {
            const pendingTableData = pendingTableDataRef.current;
            pendingTableDataRef.current = null;
            const tokenUsage = event.token_usage
              ? {
                  input: event.token_usage.input,
                  output: event.token_usage.output,
                  scope: normalizeScope(event.token_usage.scope),
                }
              : undefined;
            const unverifiedCitations = event.unverified_citations?.length
              ? event.unverified_citations
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
                    unverifiedCitations,
                    kind: "report" as const,
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
                    unverifiedCitations,
                    kind: "report" as const,
                  },
                ];
              }
              return prev;
            });
          } else if (event.type === "cancelled") {
            // Without this branch a job cancelled anywhere other than this tab
            // left the UI loading forever, with the textarea disabled and an
            // open clarification prompt still clickable.
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
                content: "Research was cancelled.",
                timestamp: new Date().toISOString(),
                jobId,
                kind: "notice" as const,
              },
            ]);
          } else if (event.type === "abandoned") {
            // Polling hit its deadline. The job may well still be running on the
            // server, so don't claim it failed — release the UI and say plainly
            // that we stopped watching.
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
                content:
                  "This is taking longer than expected, so I have stopped watching it. " +
                  "The research may still be running — reload to check back on it.",
                timestamp: new Date().toISOString(),
                jobId,
                kind: "notice" as const,
              },
            ]);
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
                kind: "notice" as const,
              },
            ]);
          }
        },
        () => {
          if (!mountedRef.current || activeJobIdRef.current !== jobId) return;
          stopStream();
          setIsLoading(false);
          setCurrentJob(null);
          // Must clear the clarification state too. stopStream() nulls
          // activeJobIdRef, so a prompt left on screen here is unanswerable: the
          // submit handler's own staleness guard sees the job id no longer
          // matches and returns before resetting anything, leaving the button
          // spinning for good.
          setClarificationRequest(null);
          setClarificationSubmitting(false);
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
      // A prompt belonging to the previous job would otherwise stay rendered
      // over the new one until that new job reaches a terminal event.
      setClarificationRequest(null);
      setClarificationSubmitting(false);

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
            kind: "notice" as const,
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

    const jobId = req.jobId;
    setClarificationSubmitting(true);

    try {
      const result = await submitClarification(jobId, req.clarificationId, answer);

      // The POST is slow enough that the user can cancel and start a new query
      // while it is in flight. Applying a stale result would drop the old
      // answer, or a message about it, into the new conversation.
      if (!mountedRef.current || activeJobIdRef.current !== jobId) return;

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
      } else if (Object.hasOwn(CLARIFICATION_REJECTION_COPY, result.status)) {
        // "already_answered" is normally reached after the first POST timed out
        // client-side, so its catch never wrote the answer into the transcript.
        // Quote it back or the notice refers to something the user cannot see.
        const notice =
          result.status === "already_answered"
            ? `Your answer was already recorded: "${answer}"`
            : CLARIFICATION_REJECTION_COPY[result.status];
        setClarificationRequest(null);
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: "assistant" as const,
            content: notice,
            timestamp: new Date().toISOString(),
            jobId,
            kind: "notice" as const,
          },
        ]);
      } else {
        // "mismatch" or anything unrecognised is a bug on one side or the
        // other, not a normal outcome. Don't tell the user their answer was
        // used, and don't claim the research moved on.
        console.error("Unexpected clarification status", result.status);
        setClarificationRequest(null);
        setError("Something went wrong with that clarification.");
      }
    } catch {
      if (!mountedRef.current || activeJobIdRef.current !== jobId) return;
      setError("Failed to submit clarification.");
    } finally {
      if (mountedRef.current && activeJobIdRef.current === jobId) {
        setClarificationSubmitting(false);
      }
    }
  }, [clarificationRequest, clarificationSubmitting]);

  const rate = useCallback(
    async (messageId: string, rating: "thumbs_up" | "thumbs_down") => {
      // Read the job id from a ref rather than assigning it inside the
      // setMessages updater: React defers the updater whenever the update queue
      // is non-empty (the normal case while tokens stream), so the assignment
      // had not happened yet and feedback was silently never sent.
      const targetJobId = messagesRef.current.find((m) => m.id === messageId)?.jobId;

      setMessages((prev) =>
        prev.map((m) => (m.id === messageId ? { ...m, rating } : m))
      );

      if (!targetJobId) {
        console.warn("No job id for rated message", messageId);
        return;
      }

      try {
        await submitFeedback(targetJobId, rating);
      } catch {
        // Feedback submission is best-effort
      }
    },
    []
  );

  return { messages, currentJob, isLoading, error, clarificationRequest, clarificationSubmitting, send, cancel, answerClarification, rate };
}

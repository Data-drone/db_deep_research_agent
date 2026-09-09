// @vitest-environment jsdom
// Only this file renders components, so jsdom is scoped here rather than made
// the default for the whole suite.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, render } from "@testing-library/react";
import { useEffect } from "react";

const {
  submitResearch,
  streamJob,
  cancelJob,
  submitFeedback,
  submitClarification,
} = vi.hoisted(() => ({
  submitResearch: vi.fn(),
  streamJob: vi.fn(),
  cancelJob: vi.fn(),
  submitFeedback: vi.fn(),
  submitClarification: vi.fn(),
}));

vi.mock("../api", () => ({ submitResearch, streamJob, cancelJob, submitFeedback, submitClarification }));

import { useResearch } from "../hooks/useResearch";

type Hook = ReturnType<typeof useResearch>;

/** Render the hook and keep a live handle on its return value. */
function mountHook() {
  const ref: { current: Hook | null } = { current: null };
  function Probe() {
    const api = useResearch();
    useEffect(() => {
      ref.current = api;
    });
    ref.current = api;
    return null;
  }
  render(<Probe />);
  return ref as { current: Hook };
}

/** The onEvent / onError callbacks the hook handed to streamJob. */
function streamHandlers() {
  const call = streamJob.mock.calls.at(-1);
  if (!call) throw new Error("streamJob was never called");
  return { onEvent: call[1] as (e: unknown) => void, onError: call[2] as (e: Error) => void };
}

async function startJob(hook: { current: Hook }, jobId = "job-1") {
  submitResearch.mockResolvedValue({ job_id: jobId, status: "pending", session_id: "s1" });
  await act(async () => {
    await hook.current.send("q", [], "chat", "research");
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  streamJob.mockReturnValue(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("clarification lifecycle", () => {
  it("shows a prompt raised on the stream", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({
        type: "clarification_needed",
        clarification_id: "c1",
        question: "Which market?",
        options: ["AU", "US"],
      });
    });

    expect(hook.current.clarificationRequest).toMatchObject({
      jobId: "job-1",
      clarificationId: "c1",
      question: "Which market?",
    });
  });

  it("ignores a prompt with no id rather than rendering a dead end", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({ type: "clarification_needed", question: "q" });
    });

    expect(hook.current.clarificationRequest).toBeNull();
  });

  it("keeps the prompt through a node_started, which the backend emits while parked", async () => {
    const hook = mountHook();
    await startJob(hook);
    const { onEvent } = streamHandlers();

    await act(async () => {
      onEvent({ type: "clarification_needed", clarification_id: "c1", question: "q", options: [] });
      onEvent({ type: "node_started", node: "clarifier" });
    });

    expect(hook.current.clarificationRequest).not.toBeNull();
  });

  it("clears the prompt when the stream dies, so it cannot strand the user", async () => {
    // Regression: onError reset isLoading but left the prompt rendered. Because
    // it also nulls the active job id, the submit handler's staleness guard
    // returned before its own reset, so the button span forever.
    const hook = mountHook();
    await startJob(hook);
    const { onEvent, onError } = streamHandlers();

    await act(async () => {
      onEvent({ type: "clarification_needed", clarification_id: "c1", question: "q", options: [] });
    });
    expect(hook.current.clarificationRequest).not.toBeNull();

    await act(async () => {
      onError(new Error("lost"));
    });

    expect(hook.current.clarificationRequest).toBeNull();
    expect(hook.current.clarificationSubmitting).toBe(false);
    expect(hook.current.isLoading).toBe(false);
  });

  it("starts the next job clean after a prompt was lost to a dead stream", async () => {
    // The full sequence Fable described: prompt raised, stream drops, the user
    // gives up and asks something else. send() clears the clarification state
    // too, so a prompt cannot outlive the job it belongs to even if some future
    // path reaches send() with one still set.
    const hook = mountHook();
    await startJob(hook, "job-1");
    await act(async () => {
      streamHandlers().onEvent({
        type: "clarification_needed",
        clarification_id: "c1",
        question: "q",
        options: [],
      });
    });
    await act(async () => {
      streamHandlers().onError(new Error("lost"));
    });

    await startJob(hook, "job-2");

    expect(hook.current.clarificationRequest).toBeNull();
    expect(hook.current.currentJob?.job_id).toBe("job-2");
  });

  it("quotes the recorded answer back when the server says it already had one", async () => {
    const hook = mountHook();
    await startJob(hook);
    await act(async () => {
      streamHandlers().onEvent({
        type: "clarification_needed",
        clarification_id: "c1",
        question: "q",
        options: ["AU"],
      });
    });

    submitClarification.mockResolvedValue({ status: "already_answered" });
    await act(async () => {
      await hook.current.answerClarification("AU");
    });

    const last = hook.current.messages.at(-1);
    expect(last?.content).toContain("AU");
    expect(last?.kind).toBe("notice");
  });

  it("treats an unrecognised status as an error, not a success", async () => {
    const hook = mountHook();
    await startJob(hook);
    await act(async () => {
      streamHandlers().onEvent({
        type: "clarification_needed",
        clarification_id: "c1",
        question: "q",
        options: ["AU"],
      });
    });

    vi.spyOn(console, "error").mockImplementation(() => {});
    submitClarification.mockResolvedValue({ status: "mismatch" });
    await act(async () => {
      await hook.current.answerClarification("AU");
    });

    expect(hook.current.error).toBeTruthy();
    expect(hook.current.messages.some((m) => m.role === "user" && m.content === "AU")).toBe(false);
  });

  it("does not treat a status inherited from Object.prototype as a known rejection", async () => {
    const hook = mountHook();
    await startJob(hook);
    await act(async () => {
      streamHandlers().onEvent({
        type: "clarification_needed",
        clarification_id: "c1",
        question: "q",
        options: ["AU"],
      });
    });

    vi.spyOn(console, "error").mockImplementation(() => {});
    submitClarification.mockResolvedValue({ status: "constructor" });
    await act(async () => {
      await hook.current.answerClarification("AU");
    });

    // The old `in` check passed here and put a function into a message body.
    expect(hook.current.error).toBeTruthy();
    expect(hook.current.messages.every((m) => typeof m.content === "string")).toBe(true);
    expect(hook.current.messages.some((m) => m.content.includes("function"))).toBe(false);
  });
});

describe("terminal events", () => {
  it("releases the UI on a cancel raised anywhere else", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({ type: "cancelled" });
    });

    expect(hook.current.isLoading).toBe(false);
    expect(hook.current.currentJob).toBeNull();
    expect(hook.current.messages.at(-1)?.kind).toBe("notice");
  });

  it("says it stopped watching rather than claiming failure when polling gives up", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({ type: "abandoned" });
    });

    expect(hook.current.isLoading).toBe(false);
    const last = hook.current.messages.at(-1);
    expect(last?.kind).toBe("notice");
    expect(last?.content).toMatch(/still be running/i);
    expect(last?.content).not.toMatch(/failed/i);
  });

  it("marks the report as a report and keeps notices out of it", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({
        type: "completed",
        result: "the report",
        token_usage: { input: 10, output: 5, scope: "job" },
        unverified_citations: ["E9"],
      });
    });

    const report = hook.current.messages.at(-1);
    expect(report?.kind).toBe("report");
    expect(report?.tokenUsage).toEqual({ input: 10, output: 5, scope: "job" });
    expect(report?.unverifiedCitations).toEqual(["E9"]);
  });

  it("does not widen an unrecognised token scope into a whole-run total", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({
        type: "completed",
        result: "r",
        token_usage: { input: 1, output: 1, scope: "something_new" },
      });
    });

    expect(hook.current.messages.at(-1)?.tokenUsage?.scope).toBe("unknown");
  });

  it("leaves unverifiedCitations undefined when the report cited nothing fake", async () => {
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({ type: "completed", result: "r", unverified_citations: [] });
    });

    expect(hook.current.messages.at(-1)?.unverifiedCitations).toBeUndefined();
  });
});

describe("feedback", () => {
  it("sends feedback for a message rated while tokens are still streaming", async () => {
    // Regression: the job id was assigned inside a setMessages updater, which
    // React defers whenever the update queue is non-empty, so the rating flipped
    // in the UI and nothing was ever sent.
    const hook = mountHook();
    await startJob(hook);

    await act(async () => {
      streamHandlers().onEvent({ type: "token", content: "partial " });
    });
    const msgId = hook.current.messages.at(-1)!.id;

    submitFeedback.mockResolvedValue(undefined);
    await act(async () => {
      streamHandlers().onEvent({ type: "token", content: "more" });
      await hook.current.rate(msgId, "thumbs_up");
    });

    expect(submitFeedback).toHaveBeenCalledWith("job-1", "thumbs_up");
  });
});

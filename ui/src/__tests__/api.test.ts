import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Hoisted: vi.mock's factory runs before module-level consts are initialised.
const { get } = vi.hoisted(() => ({ get: vi.fn() }));

vi.mock("axios", () => ({
  default: {
    create: () => ({ get, post: vi.fn(), delete: vi.fn() }),
  },
}));

import { streamJob } from "../api";

const POLL_INTERVAL_MS = 3000;

type Event = Record<string, unknown>;

/** Force the SSE attempt to fail so every test exercises the poll fallback. */
function withFailingStream() {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("stream down")));
}

function run() {
  const events: Event[] = [];
  const errors: Error[] = [];
  const cleanup = streamJob("job-1", (e) => events.push(e as Event), (e) => errors.push(e));
  return { events, errors, cleanup };
}

/** Let the current poll's promise chain settle, then advance to the next tick. */
async function nextPoll() {
  await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS);
}

beforeEach(() => {
  get.mockReset();
  vi.useFakeTimers();
  withFailingStream();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("poll fallback", () => {
  it("recovers a pending clarification the dropped stream had already shown", async () => {
    get.mockResolvedValue({
      data: {
        job_id: "job-1",
        status: "running",
        current_node: "clarifier",
        pending_clarification: { clarification_id: "c1", question: "Which market?", options: ["AU"] },
      },
    });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    cleanup();

    expect(events).toEqual([
      { type: "clarification_needed", clarification_id: "c1", question: "Which market?", options: ["AU"] },
    ]);
  });

  it("does not re-dispatch the same clarification on every tick", async () => {
    get.mockResolvedValue({
      data: {
        job_id: "job-1",
        status: "running",
        pending_clarification: { clarification_id: "c1", question: "q", options: [] },
      },
    });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await nextPoll();
    await nextPoll();
    cleanup();

    expect(events.filter((e) => e.type === "clarification_needed")).toHaveLength(1);
  });

  it("dispatches a second clarification round with a different id", async () => {
    // The guard used to be a boolean, so a fresh prompt raised in the same poll
    // session looked identical to the first one still being pending.
    const pending = (id: string) => ({
      data: {
        job_id: "job-1",
        status: "running",
        pending_clarification: { clarification_id: id, question: "q", options: [] },
      },
    });
    get
      .mockResolvedValueOnce(pending("c1"))
      .mockResolvedValueOnce(pending("c2"))
      .mockResolvedValue({ data: { job_id: "job-1", status: "running" } });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await nextPoll();
    cleanup();

    const ids = events
      .filter((e) => e.type === "clarification_needed")
      .map((e) => e.clarification_id);
    expect(ids).toEqual(["c1", "c2"]);
  });

  it("suppresses node progress while a prompt is outstanding", async () => {
    // node_started is what used to wipe a live prompt every 3 seconds.
    get.mockResolvedValue({
      data: {
        job_id: "job-1",
        status: "running",
        current_node: "clarifier",
        pending_clarification: { clarification_id: "c1", question: "q", options: [] },
      },
    });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await nextPoll();
    cleanup();

    expect(events.some((e) => e.type === "node_started")).toBe(false);
  });

  it("reports a resolved clarification once the prompt disappears", async () => {
    get
      .mockResolvedValueOnce({
        data: {
          job_id: "job-1",
          status: "running",
          pending_clarification: { clarification_id: "c1", question: "q", options: [] },
        },
      })
      .mockResolvedValue({ data: { job_id: "job-1", status: "running", current_node: "planner" } });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await nextPoll();
    cleanup();

    expect(events.map((e) => e.type)).toEqual([
      "clarification_needed",
      "clarification_resolved",
      "node_started",
    ]);
  });

  it("passes token usage and unverified citations through the completed frame", async () => {
    get.mockResolvedValue({
      data: {
        job_id: "job-1",
        status: "completed",
        result: "the report",
        token_usage: { input: 10, output: 4, scope: "job" },
        unverified_citations: ["E9"],
      },
    });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    cleanup();

    expect(events).toEqual([
      {
        type: "completed",
        result: "the report",
        token_usage: { input: 10, output: 4, scope: "job" },
        unverified_citations: ["E9"],
      },
    ]);
  });

  it("survives a transient failure instead of abandoning a live job", async () => {
    get
      .mockRejectedValueOnce(new Error("502"))
      .mockResolvedValue({ data: { job_id: "job-1", status: "completed", result: "r" } });

    const { events, errors, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await nextPoll();
    cleanup();

    expect(errors).toHaveLength(0);
    expect(events.map((e) => e.type)).toEqual(["completed"]);
  });

  it("gives up after five consecutive failures", async () => {
    get.mockRejectedValue(new Error("502"));

    const { errors, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    for (let i = 0; i < 5; i++) await nextPoll();
    cleanup();

    expect(errors).toHaveLength(1);
    expect(get).toHaveBeenCalledTimes(5);
  });

  it("stops watching a job that never terminates, without claiming it failed", async () => {
    get.mockResolvedValue({ data: { job_id: "job-1", status: "running" } });

    const { events, errors, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(21 * 60 * 1000);
    cleanup();

    expect(errors).toHaveLength(0);
    expect(events.filter((e) => e.type === "abandoned")).toHaveLength(1);
    expect(events.some((e) => e.type === "failed")).toBe(false);

    // And it really stopped: no further polling after the deadline.
    const callsAtDeadline = get.mock.calls.length;
    await vi.advanceTimersByTimeAsync(60 * 1000);
    expect(get.mock.calls.length).toBe(callsAtDeadline);
  });

  it("keeps polling right up to the deadline", async () => {
    get.mockResolvedValue({ data: { job_id: "job-1", status: "running" } });

    const { events, cleanup } = run();
    await vi.advanceTimersByTimeAsync(0);
    await vi.advanceTimersByTimeAsync(10 * 60 * 1000);
    cleanup();

    expect(events.some((e) => e.type === "abandoned")).toBe(false);
    expect(get.mock.calls.length).toBeGreaterThan(100);
  });
});

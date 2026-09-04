import { describe, expect, it, vi } from "vitest";
import {
  waitForJob,
  JOB_WAIT_HARD_CAP_S,
  type JobStatus,
} from "../../services/queue-manager.js";
import type { HistoryEntry } from "../../comfyui/client.js";

/**
 * `waitForJob` is the blocking single-prompt wait any MCP client can call
 * (queue action:"wait"). It loops on the SAME status source as
 * queue (action:"status") until a TERMINAL state, then reports the terminal
 * verdict explicitly and the run's outputs: the async completion the panel
 * gateway delivers only to its OWN agent, made pollable for every other client.
 *
 * The status function is injected here so successive states drive the loop with
 * no real ComfyUI server, exactly as the loop will see them in production.
 */

const PROMPT_ID = "f8969f48-7bec-4c3c-bd16-3d6d30cad279";

/** A status function that returns each scripted state in turn, then sticks on
 *  the last one (the steady terminal/running state). */
function scriptedStatus(seq: JobStatus[]): (promptId: string) => Promise<JobStatus> {
  let i = 0;
  return vi.fn(async () => seq[Math.min(i++, seq.length - 1)]);
}

function historyFn(
  entry: HistoryEntry,
): (promptId: string) => Promise<Record<string, HistoryEntry>> {
  return vi.fn(async () => ({ [PROMPT_ID]: entry }));
}

const RUNNING: JobStatus = { running: true, pending: false, done: false };

describe("waitForJob: blocking single-prompt wait", () => {
  it("returns success with the run's outputs once the job is done", async () => {
    const statusFn = scriptedStatus([
      RUNNING,
      {
        running: false,
        pending: false,
        done: true,
        status_str: "success",
        execution_stats: { total_duration_ms: 1234, nodes: {} },
      },
    ]);
    const history = historyFn({
      prompt: {},
      outputs: {
        "9": { images: [{ filename: "out_0001.png", subfolder: "", type: "output" }] },
      },
      status: {
        status_str: "success",
        completed: true,
        messages: [["execution_success", { timestamp: Date.now() }]],
      },
    });

    const res = await waitForJob(PROMPT_ID, 5, { pollIntervalMs: 5, statusFn, historyFn: history });

    expect(res.found).toBe(true);
    expect(res.done).toBe(true);
    expect(res.state).toBe("success");
    expect(res.timed_out).toBe(false);
    expect(res.waited_s).toBeGreaterThanOrEqual(0);
    expect(res.execution_stats).toEqual({ total_duration_ms: 1234, nodes: {} });
    expect(res.outputs).toEqual([
      { node_id: "9", files: [{ filename: "out_0001.png", subfolder: "", type: "output" }] },
    ]);
  });

  it("reports an errored run as state:\"error\" with the ComfyUI message, never silent", async () => {
    const error = {
      node_id: "3",
      node_type: "KSampler",
      exception_message: "CUDA out of memory",
    };
    const statusFn = scriptedStatus([
      RUNNING,
      { running: false, pending: false, done: true, status_str: "error", error },
    ]);
    const history = historyFn({
      prompt: {},
      outputs: {},
      status: {
        status_str: "error",
        completed: false,
        messages: [["execution_error", { ...error }]],
      },
    });

    const res = await waitForJob(PROMPT_ID, 5, { pollIntervalMs: 5, statusFn, historyFn: history });

    expect(res.done).toBe(true);
    expect(res.state).toBe("error");
    expect(res.error?.exception_message).toBe("CUDA out of memory");
  });

  it("reports an interrupted run as state:\"interrupted\"", async () => {
    const statusFn = scriptedStatus([
      RUNNING,
      { running: false, pending: false, done: true, status_str: "error" },
    ]);
    const history = historyFn({
      prompt: {},
      outputs: {},
      status: {
        status_str: "error",
        completed: false,
        messages: [["execution_interrupted", { timestamp: Date.now() }]],
      },
    });

    const res = await waitForJob(PROMPT_ID, 5, { pollIntervalMs: 5, statusFn, historyFn: history });

    expect(res.state).toBe("interrupted");
    expect(res.done).toBe(true);
  });

  it("times out without lying about the state when the run never finishes", async () => {
    const statusFn = scriptedStatus([RUNNING]);

    const res = await waitForJob(PROMPT_ID, 0.2, { pollIntervalMs: 20, statusFn });

    expect(res.timed_out).toBe(true);
    expect(res.done).toBe(false);
    expect(res.found).not.toBe(false); // it exists, it just did not finish
    expect(res.state).toBe("unknown"); // never claims success/error on a timeout
    expect(res.waited_s).toBeLessThanOrEqual(JOB_WAIT_HARD_CAP_S);
  });

  it("returns found:false at once for a prompt ComfyUI never executed, no waiting it out", async () => {
    const statusFn = scriptedStatus([
      {
        running: false,
        pending: false,
        done: false,
        found: false,
        message: "ComfyUI has no record of this prompt: not running, not queued, absent from /history.",
      },
    ]);

    const started = Date.now();
    const res = await waitForJob(PROMPT_ID, 300, { pollIntervalMs: 20, statusFn });
    const elapsedMs = Date.now() - started;

    expect(res.found).toBe(false);
    expect(res.done).toBe(false);
    expect(res.timed_out).toBe(false);
    expect(res.state).toBe("unknown");
    expect(res.message).toMatch(/no record of this prompt/i);
    // It must NOT have polled to the 300s deadline: one status read, then out.
    expect(statusFn).toHaveBeenCalledTimes(1);
    expect(elapsedMs).toBeLessThan(1000);
  });
});

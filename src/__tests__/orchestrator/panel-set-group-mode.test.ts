// panel_set_group_mode — switch every node of a GROUP on or off in one call.
// A group box is only a frame on the canvas; the frontend knows which nodes
// it holds (the `node_ids` of panel_query_graph's groups index), and this tool
// applies one execution mode to each of them through the same command
// panel_set_node_mode uses, so the per-node rules (force, refusals) hold.

import { describe, expect, it } from "vitest";

import { buildPanelToolDefs, makePanelToolCtx } from "../../orchestrator/panel-tools.js";
import type { PanelToolCtx, ToolResult } from "../../orchestrator/panel-tools.js";
import { WorkflowTargetStore } from "../../services/workflow-target-store.js";

const TAB = "11111111-2222-3333-4444-555555555555";

const textOf = (res: ToolResult): string =>
  res.content.map((c) => (c as { text?: string }).text ?? "").join(" ");

type Sent = Array<Record<string, unknown>>;

function harness(opts: { groups?: unknown[]; failOnNode?: number; truncated?: boolean } = {}) {
  const sent: Sent = [];
  // The live groups index spells member ids as STRINGS ("10"); the tool hands
  // numbers on, the form every other panel tool takes.
  const groups = opts.groups ?? [
    { id: 3, title: "Refiner", node_ids: ["10", "11"] },
    { id: 4, title: "Upscale", node_ids: [20] },
  ];
  const bridge = {
    send: async (cmd: Record<string, unknown>) => {
      sent.push(cmd);
      switch (cmd.cmd) {
        case "graph_query":
          return { nodes: [], groups, ...(opts.truncated ? { groups_truncated: true } : {}) };
        case "graph_set_node_mode":
          if (cmd.node_id === opts.failOnNode) throw new Error(`simulated refusal on ${String(cmd.node_id)}`);
          return { node_id: cmd.node_id, mode: cmd.mode, previous_mode: "active" };
        default:
          throw new Error(`unexpected ${String(cmd.cmd)}`);
      }
    },
    push: () => 1,
    canReach: (id: string) => id === TAB,
    isHeadless: () => false,
    tabs: () => [{ tab_id: TAB, title: "wf", connected_at: 0 }],
    resolveActiveTabId: () => TAB,
    workflowUuidFor: () => ({ known: true, uuid: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa" }),
    tabCanMutateGraph: () => true,
    tabGraphMutationCapability: () => ({ known: true, canMutate: true }),
  } as PanelToolCtx["bridge"];
  return { sent, ctx: makePanelToolCtx(bridge, TAB, new WorkflowTargetStore()) };
}

function tool() {
  const d = buildPanelToolDefs().find((d) => d.name === "panel_set_group_mode");
  if (!d) throw new Error("panel_set_group_mode is not registered");
  return d;
}

const modeCalls = (sent: Sent) => sent.filter((c) => c.cmd === "graph_set_node_mode").map((c) => [c.node_id, c.mode]);

describe("panel_set_group_mode", () => {
  it("bypasses every member of the group named by title, and reports each node", async () => {
    const { sent, ctx } = harness();
    const res = await tool().handler({ group: "refin", mode: "bypass" }, ctx);
    expect(res.isError, textOf(res)).toBeUndefined();
    expect(modeCalls(sent)).toEqual([
      [10, "bypass"],
      [11, "bypass"],
    ]);
    const reply = JSON.parse(textOf(res)) as { group_id: number; title: string; mode: string; nodes: unknown[] };
    expect(reply).toMatchObject({ group_id: 3, title: "Refiner", mode: "bypass" });
    expect(reply.nodes).toEqual([
      { node_id: 10, mode: "bypass", previous_mode: "active" },
      { node_id: 11, mode: "bypass", previous_mode: "active" },
    ]);
  });

  it("accepts the numeric group id and passes `force` through", async () => {
    const { sent, ctx } = harness();
    const res = await tool().handler({ group_id: 4, mode: "mute", force: true }, ctx);
    expect(res.isError, textOf(res)).toBeUndefined();
    expect(sent.filter((c) => c.cmd === "graph_set_node_mode")).toEqual([
      expect.objectContaining({ node_id: 20, mode: "mute", force: true }),
    ]);
  });

  it("refuses an unknown or ambiguous title without touching any node", async () => {
    const { sent, ctx } = harness({ groups: [{ id: 1, title: "Stage A", node_ids: [1] }, { id: 2, title: "Stage B", node_ids: [2] }] });
    const none = await tool().handler({ group: "nope", mode: "bypass" }, ctx);
    expect(none.isError).toBe(true);
    const many = await tool().handler({ group: "stage", mode: "bypass" }, ctx);
    expect(many.isError).toBe(true);
    expect(textOf(many)).toMatch(/ambiguous/);
    expect(modeCalls(sent)).toEqual([]);
  });

  it("refuses when the group's member list is capped, rather than switching half a group", async () => {
    const { sent, ctx } = harness({ groups: [{ id: 3, title: "Refiner", node_ids: [10], node_ids_truncated: "Showing 1 of 2" }] });
    const res = await tool().handler({ group_id: 3, mode: "bypass" }, ctx);
    expect(res.isError).toBe(true);
    expect(textOf(res)).toMatch(/truncated|capped/i);
    expect(modeCalls(sent)).toEqual([]);
  });

  it("says nothing to do for an empty group", async () => {
    const { sent, ctx } = harness({ groups: [{ id: 5, title: "Empty", node_ids: [] }] });
    const res = await tool().handler({ group_id: 5, mode: "bypass" }, ctx);
    expect(res.isError).toBeUndefined();
    expect(textOf(res)).toMatch(/no node/i);
    expect(modeCalls(sent)).toEqual([]);
  });

  it("stops at the first refused node and lists what was already switched", async () => {
    const { sent, ctx } = harness({ failOnNode: 11 });
    const res = await tool().handler({ group_id: 3, mode: "mute" }, ctx);
    expect(res.isError).toBe(true);
    expect(textOf(res)).toMatch(/simulated refusal/);
    expect(textOf(res)).toMatch(/10/);
    expect(modeCalls(sent)).toEqual([
      [10, "mute"],
      [11, "mute"],
    ]);
  });
});

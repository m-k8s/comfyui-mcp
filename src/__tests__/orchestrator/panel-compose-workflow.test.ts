// panel_compose_workflow — a fragment of the pseudo-Python `to_code` renders,
// applied to the user's LIVE graph: new targets create nodes, existing targets
// are modified, variables are links. The plan is computed in full first and
// nothing reaches the canvas while a problem stands; once applying, a failed
// step halts the rest and the reply says exactly what was applied.

import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../../comfyui/client.js", () => ({
  getObjectInfo: vi.fn(),
  backfillObjectInfo: vi.fn(),
  resetClient: vi.fn(),
  resetObjectInfoCache: vi.fn(),
}));

const { buildPanelToolDefs, makePanelToolCtx } = await import("../../orchestrator/panel-tools.js");
const { getObjectInfo } = await import("../../comfyui/client.js");
import { WorkflowTargetStore } from "../../services/workflow-target-store.js";
import type { PanelToolCtx, ToolResult } from "../../orchestrator/panel-tools.js";
import type { ComfyUINodeDef, ObjectInfo } from "../../comfyui/types.js";

const TAB = "11111111-2222-3333-4444-555555555555";

const textOf = (res: ToolResult): string =>
  res.content.map((c) => (c as { text?: string }).text ?? "").join(" ");

const def = (input: ComfyUINodeDef["input"], output: string[]): ComfyUINodeDef => ({
  input,
  output,
  output_name: output,
  output_is_list: output.map(() => false),
  name: "",
  display_name: "",
  description: "",
  category: "",
  output_node: false,
});

const OI: ObjectInfo = {
  CheckpointLoaderSimple: def({ required: { ckpt_name: [["a.safetensors"]] } }, ["MODEL", "CLIP", "VAE"]),
  LoraLoaderModelOnly: def(
    { required: { model: ["MODEL"], lora_name: [["x.safetensors"]], strength_model: ["FLOAT", { default: 1 }] } },
    ["MODEL"],
  ),
  ModelSamplingAuraFlow: def({ required: { model: ["MODEL"], shift: ["FLOAT", { default: 3 }] } }, ["MODEL"]),
  CFGNorm: def({ required: { model: ["MODEL"], strength: ["FLOAT", { default: 1 }] } }, ["patched_model"]),
};

const LIVE_NODES = [
  { id: 50, type: "CheckpointLoaderSimple", widgets_values: ["a.safetensors"] },
  { id: 71, type: "CFGNorm", widgets_values: [1] },
];

const CODE = `
model_a = LoraLoaderModelOnly(model=model_50, lora_name="x.safetensors", strength_model=0.8)
model_b = ModelSamplingAuraFlow(model=model_a, shift=3.1)
patched_model_71 = CFGNorm(model=model_b, strength=1.0)
`;

type Sent = Array<Record<string, unknown>>;

function harness(
  opts: { failOn?: (cmd: Record<string, unknown>, nth: number) => boolean; liveNodes?: typeof LIVE_NODES } = {},
) {
  const sent: Sent = [];
  const LIVE = opts.liveNodes ?? LIVE_NODES;
  let nextId = 101;
  const bridge = {
    send: async (cmd: Record<string, unknown>) => {
      sent.push(cmd);
      const nth = sent.filter((c) => c.cmd === cmd.cmd).length;
      if (opts.failOn?.(cmd, nth)) throw new Error(`simulated failure on ${String(cmd.cmd)}`);
      switch (cmd.cmd) {
        case "graph_serialize":
          return { workflow: { nodes: LIVE, links: [] }, node_count: LIVE.length };
        case "graph_get_state":
          return {
            viewing: { scope: "root" },
            nodes: LIVE.map((n) => ({ id: n.id, type: n.type, widgets: {} })),
          };
        case "graph_add_node":
          // The real panel reply: the created node under `added`, its id a STRING.
          return { added: { id: String(nextId++), type: cmd.class_type, mode: "active" }, viewing: { scope: "root" } };
        case "graph_set_widget":
          return { ok: true, node_id: cmd.node_id, widget: cmd.widget, previous: null, value: cmd.value };
        case "graph_connect":
          return { ok: true };
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
  const d = buildPanelToolDefs().find((d) => d.name === "panel_compose_workflow");
  if (!d) throw new Error("panel_compose_workflow is not registered");
  return d;
}

const of = (sent: Sent, cmd: string) => sent.filter((c) => c.cmd === cmd);

beforeEach(() => {
  vi.mocked(getObjectInfo).mockReset();
  vi.mocked(getObjectInfo).mockResolvedValue(OI);
});

describe("panel_compose_workflow", () => {
  it("creates the new nodes, sets their values, and wires the fragment onto the live graph", async () => {
    const { sent, ctx } = harness();
    const res = await tool().handler({ code: CODE }, ctx);
    expect(res.isError, textOf(res)).toBeUndefined();

    const adds = of(sent, "graph_add_node");
    expect(adds.map((c) => c.class_type)).toEqual(["LoraLoaderModelOnly", "ModelSamplingAuraFlow"]);

    // Values of a created node are set on the id the canvas handed back.
    expect(of(sent, "graph_set_widget")).toContainEqual(
      expect.objectContaining({ node_id: "101", widget: "lora_name", value: "x.safetensors" }),
    );
    expect(of(sent, "graph_set_widget")).toContainEqual(
      expect.objectContaining({ node_id: "101", widget: "strength_model", value: 0.8 }),
    );
    expect(of(sent, "graph_set_widget")).toContainEqual(
      expect.objectContaining({ node_id: 71, widget: "strength", value: 1 }),
    );

    // Links resolve provisional ids to the ids the canvas handed back; the last one rewires node 71.
    const links = of(sent, "graph_connect").map((c) => [c.from_node_id, c.from_output, c.to_node_id, c.to_input]);
    expect(links).toEqual([
      [50, 0, "101", "model"],
      ["101", 0, "102", "model"],
      ["102", 0, 71, "model"],
    ]);

    const reply = JSON.parse(textOf(res)) as { created: Array<{ node_id: string; class_type: string }> };
    expect(reply.created.map((c) => [c.node_id, c.class_type])).toEqual([
      ["101", "LoraLoaderModelOnly"],
      ["102", "ModelSamplingAuraFlow"],
    ]);
  });

  it("composes onto a BLANK canvas (a fresh tab has zero nodes, which is a graph, not a failed read)", async () => {
    const { sent, ctx } = harness({ liveNodes: [] });
    const res = await tool().handler(
      { code: `model_1, clip_1, vae_1 = CheckpointLoaderSimple(ckpt_name="a.safetensors")\nx = CFGNorm(model=model_1)` },
      ctx,
    );
    expect(res.isError, textOf(res)).toBeUndefined();
    expect(of(sent, "graph_add_node").map((c) => c.class_type)).toEqual(["CheckpointLoaderSimple", "CFGNorm"]);
    expect(of(sent, "graph_connect")).toEqual([
      expect.objectContaining({ from_node_id: "101", from_output: 0, to_node_id: "102", to_input: "model" }),
    ]);
  });

  it("touches nothing on the canvas while a problem stands, and names the line", async () => {
    const { sent, ctx } = harness();
    const res = await tool().handler({ code: `x = LoraLoaderModelOnly(model=model_99)\ny = Nope(a=1)` }, ctx);
    expect(res.isError).toBe(true);
    expect(textOf(res)).toMatch(/line 1/);
    expect(textOf(res)).toMatch(/line 2/);
    expect(of(sent, "graph_add_node")).toHaveLength(0);
    expect(of(sent, "graph_set_widget")).toHaveLength(0);
    expect(of(sent, "graph_connect")).toHaveLength(0);
  });

  it("dry_run returns the plan without applying it", async () => {
    const { sent, ctx } = harness();
    const res = await tool().handler({ code: CODE, dry_run: true }, ctx);
    expect(res.isError).toBeUndefined();
    expect(textOf(res)).toMatch(/create node .*LoraLoaderModelOnly/);
    expect(of(sent, "graph_add_node")).toHaveLength(0);
    expect(of(sent, "graph_connect")).toHaveLength(0);
  });

  it("halts at the first failed step and says what was applied before it", async () => {
    const { sent, ctx } = harness({ failOn: (cmd, nth) => cmd.cmd === "graph_connect" && nth === 2 });
    const res = await tool().handler({ code: CODE }, ctx);
    expect(res.isError).toBe(true);
    const text = textOf(res);
    expect(text).toMatch(/simulated failure/);
    expect(text).toMatch(/applied/i);
    // Nothing after the failure was attempted.
    expect(of(sent, "graph_connect")).toHaveLength(2);
    expect(of(sent, "graph_set_widget").some((c) => c.node_id === 71)).toBe(false);
  });

  it("reads the live graph even when object_info is unreachable, and still refuses an unknown output", async () => {
    vi.mocked(getObjectInfo).mockRejectedValue(new Error("ECONNREFUSED"));
    const { sent, ctx } = harness();
    const res = await tool().handler({ code: `x = CFGNorm(model=out0_50, strength=2)` }, ctx);
    expect(res.isError, textOf(res)).toBeUndefined();
    expect(of(sent, "graph_add_node")).toHaveLength(1);
    expect(of(sent, "graph_connect")).toEqual([
      expect.objectContaining({ from_node_id: 50, from_output: 0, to_node_id: "101", to_input: "model" }),
    ]);
  });
});

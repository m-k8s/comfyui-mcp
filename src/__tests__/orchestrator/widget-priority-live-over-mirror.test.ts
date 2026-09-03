// `widgets_values_named` is a MIRROR the frontend writes once and then carries
// along: nothing re-synchronises it when a widget changes. On two real
// workflows it announced one LoRA while `widgets_values` (what ComfyUI
// executes) loaded another. Reading the mirror FIRST therefore reports a graph
// that does not exist. The live name-keyed capture (`graph_get_state`, straight
// off the litegraph widget objects) is the frontend's own truth and must win;
// the mirror is a last resort, and a read that had nothing else must say so.

import { describe, expect, it } from "vitest";

import {
  readLiveUiGraphForContentDrift,
  uiGraphToApiGraph,
} from "../../orchestrator/root-shape-mismatch.js";

const LORA_EXECUTED = "bfs_head_v5_2511_merged_version_rank_32_fp32.safetensors";
const LORA_STALE = "bfs_head_v5_2511_merged_version_rank_64_bf16.safetensors";

const text = (payload: unknown) => ({
  content: [{ type: "text", text: JSON.stringify(payload) }],
});

describe("widget priority: what executes beats the widgets_values_named mirror", () => {
  it("prefers the live name-keyed `widgets` over the mirror when the two disagree", () => {
    const api = uiGraphToApiGraph([
      {
        id: 7,
        type: "LoraLoaderModelOnly",
        widgets: { lora_name: LORA_EXECUTED, strength_model: 1 },
        widgets_values_named: { lora_name: LORA_STALE, strength_model: 1 },
      },
    ]);
    expect(api["7"]?.inputs.lora_name).toBe(LORA_EXECUTED);
  });

  it("prefers the overlay from the live capture (`capturedWidgetValues`) over the mirror", () => {
    const api = uiGraphToApiGraph([
      {
        id: 7,
        type: "LoraLoaderModelOnly",
        capturedWidgetValues: { lora_name: LORA_EXECUTED, strength_model: 1 },
        widgets_values: [LORA_EXECUTED, 1],
        widgets_values_named: { lora_name: LORA_STALE, strength_model: 1 },
      },
    ]);
    expect(api["7"]?.inputs.lora_name).toBe(LORA_EXECUTED);
  });

  it("discloses a node whose ONLY named source was the mirror", () => {
    const api = uiGraphToApiGraph([
      {
        id: 7,
        type: "LoraLoaderModelOnly",
        widgets_values: [LORA_STALE, 1],
        widgets_values_named: { lora_name: LORA_STALE, strength_model: 1 },
      },
    ]);
    const meta = api["7"]?._meta as Record<string, unknown> | undefined;
    expect(meta?.widgets_source).toBe("widgets_values_named");
  });

  it("does not flag a node read from the live capture", () => {
    const api = uiGraphToApiGraph([
      { id: 7, type: "LoraLoaderModelOnly", widgets: { lora_name: LORA_EXECUTED } },
    ]);
    const meta = api["7"]?._meta as Record<string, unknown> | undefined;
    expect(meta?.widgets_source).toBeUndefined();
  });

  it("readLiveUiGraphForContentDrift lays the live capture over the serialized nodes", async () => {
    const sent: string[] = [];
    const live = await readLiveUiGraphForContentDrift(async (cmd) => {
      sent.push(String(cmd.cmd));
      if (cmd.cmd === "graph_serialize") {
        return text({
          nodes: [
            {
              id: 7,
              type: "LoraLoaderModelOnly",
              widgets_values: [LORA_EXECUTED, 1],
              widgets_values_named: { lora_name: LORA_STALE, strength_model: 1 },
            },
          ],
        });
      }
      if (cmd.cmd === "graph_get_state") {
        return text({
          viewing: { scope: "root" },
          nodes: [
            {
              id: 7,
              type: "LoraLoaderModelOnly",
              widgets: { lora_name: LORA_EXECUTED, strength_model: 1 },
            },
          ],
        });
      }
      throw new Error(`unexpected ${String(cmd.cmd)}`);
    });
    expect(live?.recovered_from).toBe("graph_serialize");
    expect(sent).toEqual(["graph_serialize", "graph_get_state"]);
    const api = uiGraphToApiGraph(live?.nodes);
    expect(api["7"]?.inputs.lora_name).toBe(LORA_EXECUTED);
  });

  it("keeps the serialized read when the live capture describes a subgraph, not the root", async () => {
    const live = await readLiveUiGraphForContentDrift(async (cmd) => {
      if (cmd.cmd === "graph_serialize") {
        return text({
          nodes: [{ id: 7, type: "LoraLoaderModelOnly", widgets_values_named: { lora_name: LORA_STALE } }],
        });
      }
      if (cmd.cmd === "graph_get_state") {
        return text({
          viewing: { scope: "subgraph" },
          nodes: [{ id: 7, type: "KSampler", widgets: { seed: 1 } }],
        });
      }
      throw new Error(`unexpected ${String(cmd.cmd)}`);
    });
    const api = uiGraphToApiGraph(live?.nodes);
    // The subgraph's node 7 must NOT overwrite the root's node 7.
    expect(api["7"]?.inputs.seed).toBeUndefined();
    expect(api["7"]?.inputs.lora_name).toBe(LORA_STALE);
  });
});

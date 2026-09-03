// `visualize_workflow (action:"to_code")` — the graph as pseudo-Python: one
// assignment per node in TOPOLOGICAL order, every link a variable reference,
// every literal a keyword argument, typed signatures of the classes used in a
// preamble. ComfyBench (CVPR 2025) measures this family at 62.0 pass / 41.0
// resolve against 51.0 / 30.0 for an element list (the shape of `to_dsl`) and
// 52.0 / 33.0 for raw JSON: the model reads dependencies as data flow instead
// of resolving ids by hand.

import { describe, expect, it } from "vitest";

import type { ComfyUINodeDef, ObjectInfo, WorkflowJSON } from "../../comfyui/types.js";
import { workflowToCode } from "../../services/workflow-code.js";

/** A node definition with only what the renderer reads filled in. */
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
  CLIPTextEncode: def(
    { required: { text: ["STRING", { multiline: true }], clip: ["CLIP"] } },
    ["CONDITIONING"],
  ),
  KSampler: def(
    {
      required: {
        model: ["MODEL"],
        seed: ["INT", { default: 0, control_after_generate: true }],
        positive: ["CONDITIONING"],
      },
      optional: { latent_image: ["LATENT"] },
    },
    ["LATENT"],
  ),
};

// Deliberately listed consumer-first: the renderer must sort by dependency,
// not by id.
const WF: WorkflowJSON = {
  "3": {
    class_type: "KSampler",
    inputs: { seed: 5, model: ["1", 0], positive: ["2", 0] },
  },
  "2": {
    class_type: "CLIPTextEncode",
    inputs: { text: "a cat", clip: ["1", 1] },
    _meta: { title: "Positive" },
  },
  "1": {
    class_type: "CheckpointLoaderSimple",
    inputs: { ckpt_name: "a.safetensors" },
  },
};

describe("workflowToCode", () => {
  it("emits one assignment per node, producers before consumers", () => {
    const code = workflowToCode(WF, OI);
    const at = (s: string) => {
      const i = code.indexOf(s);
      expect(i, `missing: ${s}`).toBeGreaterThanOrEqual(0);
      return i;
    };
    expect(at("CheckpointLoaderSimple(")).toBeLessThan(at("CLIPTextEncode("));
    expect(at("CLIPTextEncode(")).toBeLessThan(at("KSampler("));
  });

  it("names outputs after their type and node id, and passes links as variables", () => {
    const code = workflowToCode(WF, OI);
    expect(code).toContain('model_1, clip_1, vae_1 = CheckpointLoaderSimple(ckpt_name="a.safetensors")');
    expect(code).toContain('conditioning_2 = CLIPTextEncode(text="a cat", clip=clip_1)');
    expect(code).toContain("latent_3 = KSampler(seed=5, model=model_1, positive=conditioning_2)");
  });

  it("opens with a header and the typed signatures of the classes used", () => {
    const code = workflowToCode(WF, OI);
    expect(code.startsWith("# WORKFLOW: 3 nodes")).toBe(true);
    expect(code).toContain("#   CLIPTextEncode(text: STRING, clip: CLIP) -> CONDITIONING");
    expect(code).toContain("#   KSampler(model: MODEL, seed: INT, positive: CONDITIONING, latent_image?: LATENT) -> LATENT");
    expect(code).toContain("#   CheckpointLoaderSimple(ckpt_name: COMBO) -> MODEL, CLIP, VAE");
  });

  it("without object_info, names only the outputs something consumes and skips the preamble", () => {
    const code = workflowToCode(WF);
    expect(code).toContain('out0_1, out1_1 = CheckpointLoaderSimple(ckpt_name="a.safetensors")');
    expect(code).toContain("out0_2 = CLIPTextEncode(");
    // Nothing consumes the sampler: a bare call, no dangling variable.
    expect(code).toMatch(/^KSampler\(seed=5, model=out0_1, positive=out0_2\)/m);
    expect(code).not.toContain("Signatures");
  });

  it("carries the title, and the execution mode when a node is switched off", () => {
    const wf: WorkflowJSON = {
      ...WF,
      "2": { ...WF["2"], _meta: { title: "Positive", mode: "bypassed" } },
    };
    const code = workflowToCode(wf, OI);
    // The assignment, not the signature line of the preamble.
    const assignment = (s: string) => code.split("\n").find((l) => !l.startsWith("#") && l.includes(s));
    const line = assignment("CLIPTextEncode(");
    expect(line).toContain('"Positive"');
    expect(line).toContain("[bypass]");
    expect(assignment("CheckpointLoaderSimple(")).not.toContain("#");
  });

  it("keeps a class name that is not an identifier callable, and says what it was", () => {
    const wf: WorkflowJSON = {
      "9": { class_type: "Image Save", inputs: { images: ["3", 0] } },
      ...WF,
    };
    const code = workflowToCode(wf, OI);
    expect(code).toContain("Image_Save(images=latent_3)");
    expect(code).toContain('type "Image Save"');
  });

  it("renders literals as Python: booleans, None, escaped and clipped strings", () => {
    const long = "x".repeat(200);
    const wf: WorkflowJSON = {
      "1": {
        class_type: "CLIPTextEncode",
        inputs: { text: `line one\nline "two"`, flag: true, nothing: null, big: long },
      },
    };
    const code = workflowToCode(wf);
    expect(code).toContain('text="line one\\nline \'two\'"');
    expect(code).toContain("flag=True");
    expect(code).toContain("nothing=None");
    expect(code).not.toContain(long);
    expect(code).toContain("...");
  });

  it("survives a cycle instead of recursing forever", () => {
    const wf: WorkflowJSON = {
      "1": { class_type: "A", inputs: { x: ["2", 0] } },
      "2": { class_type: "B", inputs: { y: ["1", 0] } },
    };
    const code = workflowToCode(wf);
    expect(code).toContain("A(");
    expect(code).toContain("B(");
  });

  it("renders an empty workflow as a header only", () => {
    expect(workflowToCode({})).toBe("# WORKFLOW: 0 nodes\n");
  });
});

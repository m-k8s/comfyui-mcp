// The reverse of `to_code`: a fragment of the same pseudo-Python, read (never
// executed) and turned into a plan of graph operations. Two rules carry the
// whole thing:
//
//   - a variable named `<output>_<node id>` that matches an EXISTING node
//     refers to that node's output;
//   - a line whose first target names an existing node MODIFIES it (values,
//     rewiring); a line whose target is new CREATES a node.
//
// Nothing is applied here. The plan and the problems come back together, and a
// caller applies nothing while a problem stands: a half-applied fragment is a
// graph that is harder to repair than a refusal.

import { describe, expect, it } from "vitest";

import type { ComfyUINodeDef, ObjectInfo, WorkflowJSON } from "../../comfyui/types.js";
import { workflowToCode } from "../../services/workflow-code.js";
import { codeToWorkflow, composeFromCode } from "../../services/workflow-code-compose.js";

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
  CLIPTextEncode: def({ required: { text: ["STRING", { multiline: true }], clip: ["CLIP"] } }, ["CONDITIONING"]),
};

// The existing graph: a loader (50) feeding a CFGNorm (71) directly.
const EXISTING: WorkflowJSON = {
  "50": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "a.safetensors" } },
  "71": { class_type: "CFGNorm", inputs: { model: ["50", 0], strength: 1 } },
  "328:3": { class_type: "CLIPTextEncode", inputs: { text: "inside a subgraph", clip: ["50", 1] } },
};

describe("composeFromCode: the fragment from the docstring", () => {
  const CODE = `
# two LoRA-ish stages slid between the loader and the sampler's model input
model_a = LoraLoaderModelOnly(model=model_50, lora_name="x.safetensors", strength_model=0.8)
model_b = ModelSamplingAuraFlow(model=model_a, shift=3.1)
patched_model_71 = CFGNorm(model=model_b, strength=1.0)
`;

  it("creates the two new nodes, rewires the existing one, and reports no problem", () => {
    const r = composeFromCode(CODE, { existing: EXISTING, objectInfo: OI });
    expect(r.problems).toEqual([]);
    const ops = r.operations;
    expect(ops.filter((o) => o.op === "add_node").map((o) => o.op === "add_node" && o.class_type)).toEqual([
      "LoraLoaderModelOnly",
      "ModelSamplingAuraFlow",
    ]);
    // The existing CFGNorm is modified, not recreated.
    expect(ops.some((o) => o.op === "add_node" && o.class_type === "CFGNorm")).toBe(false);
    expect(ops).toContainEqual({ op: "set_widget", node: "71", name: "strength", value: 1 });
  });

  it("links through provisional ids, and the last link rewires node 71's model input", () => {
    const r = composeFromCode(CODE, { existing: EXISTING, objectInfo: OI });
    const adds = r.operations.filter((o) => o.op === "add_node");
    const [lora, aura] = adds.map((o) => (o.op === "add_node" ? o.node : ""));
    expect(r.operations).toContainEqual({ op: "connect", from: "50", from_slot: 0, to: lora, to_input: "model" });
    expect(r.operations).toContainEqual({ op: "connect", from: lora, from_slot: 0, to: aura, to_input: "model" });
    expect(r.operations).toContainEqual({ op: "connect", from: aura, from_slot: 0, to: "71", to_input: "model" });
    expect(r.created).toEqual([lora, aura]);
  });

  it("carries literal values into the created node", () => {
    const r = composeFromCode(CODE, { existing: EXISTING, objectInfo: OI });
    const lora = r.operations.find((o) => o.op === "add_node" && o.class_type === "LoraLoaderModelOnly");
    expect(lora && lora.op === "add_node" ? lora.widgets : null).toEqual({
      lora_name: "x.safetensors",
      strength_model: 0.8,
    });
  });

  it("writes a human plan, one line per effect", () => {
    const r = composeFromCode(CODE, { existing: EXISTING, objectInfo: OI });
    expect(r.plan.some((l) => /create .*LoraLoaderModelOnly/.test(l))).toBe(true);
    expect(r.plan.some((l) => /71/.test(l) && /model/.test(l))).toBe(true);
  });
});

describe("composeFromCode: resolving variables", () => {
  it("names an output by its type, case-insensitively, and by out<slot>", () => {
    const r = composeFromCode(
      `conditioning_9 = CLIPTextEncode(text="x", clip=CLIP_50)\nconditioning_10 = CLIPTextEncode(text="y", clip=out1_50)`,
      { existing: EXISTING, objectInfo: OI },
    );
    expect(r.problems).toEqual([]);
    const links = r.operations.filter((o) => o.op === "connect");
    expect(links.map((o) => (o.op === "connect" ? o.from_slot : -1))).toEqual([1, 1]);
  });

  it("reaches a node inside a subgraph through its underscore spelling", () => {
    const r = composeFromCode(`conditioning_328_3 = CLIPTextEncode(text="changed")`, {
      existing: EXISTING,
      objectInfo: OI,
    });
    expect(r.problems).toEqual([]);
    expect(r.operations).toEqual([{ op: "set_widget", node: "328:3", name: "text", value: "changed" }]);
  });

  it("refuses a variable that names no node, and says what a variable must look like", () => {
    const r = composeFromCode(`model_a = CFGNorm(model=model_99)`, { existing: EXISTING, objectInfo: OI });
    expect(r.problems).toHaveLength(1);
    expect(r.problems[0]).toMatch(/line 1/);
    expect(r.problems[0]).toMatch(/model_99/);
    expect(r.problems[0]).toMatch(/<output>_<node id>/);
    expect(r.operations).toEqual([]);
  });

  it("refuses an output the node does not have, listing the ones it has", () => {
    const r = composeFromCode(`model_a = CFGNorm(model=latent_50)`, { existing: EXISTING, objectInfo: OI });
    expect(r.problems[0]).toMatch(/node 50 has no output "latent"/);
    expect(r.problems[0]).toMatch(/MODEL, CLIP, VAE/);
  });

  it("refuses to modify an existing node under another class", () => {
    const r = composeFromCode(`patched_model_71 = LoraLoaderModelOnly(model=model_50)`, {
      existing: EXISTING,
      objectInfo: OI,
    });
    expect(r.problems[0]).toMatch(/node 71 is a CFGNorm, not a LoraLoaderModelOnly/);
  });

  it("refuses an uninstalled class and an unknown input, naming the real ones", () => {
    const r = composeFromCode(`x = NoSuchNode(a=1)\ny = CFGNorm(model=model_50, strengthh=2)`, {
      existing: EXISTING,
      objectInfo: OI,
    });
    expect(r.problems).toHaveLength(2);
    expect(r.problems[0]).toMatch(/line 1: NoSuchNode is not installed/);
    expect(r.problems[1]).toMatch(/line 2: CFGNorm has no input "strengthh"/);
    expect(r.problems[1]).toMatch(/model, strength/);
  });

  it("finds a class whose name is not an identifier under the spelling to_code gave it", () => {
    const oi: ObjectInfo = { ...OI, "Image Save": def({ required: { images: ["IMAGE"] } }, []) };
    const r = composeFromCode(`Image_Save(images=out0_50)`, { existing: EXISTING, objectInfo: oi });
    expect(r.problems).toEqual([]);
    expect(r.operations[0]).toMatchObject({ op: "add_node", class_type: "Image Save" });
  });

  it("modifies an existing node of such a class under the same spelling", () => {
    const oi: ObjectInfo = { ...OI, "Image Save": def({ required: { images: ["IMAGE"], quality: ["INT"] } }, []) };
    const existing: WorkflowJSON = { ...EXISTING, "9": { class_type: "Image Save", inputs: { quality: 80 } } };
    const r = composeFromCode(`out0_9 = Image_Save(quality=95)`, { existing, objectInfo: oi });
    expect(r.problems).toEqual([]);
    expect(r.operations).toEqual([{ op: "set_widget", node: "9", name: "quality", value: 95 }]);
  });

  it("round-trips a graph with such a class through to_code", () => {
    const oi: ObjectInfo = { ...OI, "Image Save": def({ required: { images: ["IMAGE"] } }, []) };
    const wf: WorkflowJSON = {
      "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "a.safetensors" } },
      "2": { class_type: "Image Save", inputs: { images: ["1", 0] } },
    };
    const back = codeToWorkflow(workflowToCode(wf, oi), oi);
    expect(back.problems).toEqual([]);
    expect(back.workflow).toEqual(wf);
  });

  it("without object_info, creates what it is told and checks nothing it cannot know", () => {
    const r = composeFromCode(`x = Anything(a=1, b=out0_50)`, { existing: EXISTING });
    expect(r.problems).toEqual([]);
    expect(r.operations[0]).toMatchObject({ op: "add_node", class_type: "Anything", widgets: { a: 1 } });
  });
});

describe("composeFromCode: what the reader refuses, with the line", () => {
  const refuse = (code: string) => composeFromCode(code, { existing: EXISTING, objectInfo: OI }).problems;

  it("positional arguments", () => {
    expect(refuse(`x = CFGNorm(model_50, strength=1)`)[0]).toMatch(/line 1.*named/);
  });

  it("statements that are not an assignment or a bare call", () => {
    expect(refuse(`import os`)[0]).toMatch(/line 1.*only .*assignments/i);
    expect(refuse(`for n in nodes:\n    pass`)[0]).toMatch(/line 1/);
    expect(refuse(`def f():\n    return 1`)[0]).toMatch(/line 1/);
  });

  it("calls through an attribute, and expressions in place of a value", () => {
    expect(refuse(`x = comfy.CFGNorm(model=model_50)`)[0]).toMatch(/line 1/);
    expect(refuse(`x = CFGNorm(model=model_50, strength=1 + 2)`)[0]).toMatch(/line 1.*constant/);
    expect(refuse(`x = CFGNorm(model=model_50, strength=f(2))`)[0]).toMatch(/line 1.*constant/);
  });

  it("star arguments", () => {
    expect(refuse(`x = CFGNorm(**opts)`)[0]).toMatch(/line 1.*\*\*/);
  });

  it("unbalanced code, with the line", () => {
    expect(refuse(`x = CFGNorm(model=model_50`)[0]).toMatch(/line 1/);
  });

  it("does not apply anything when a later line fails", () => {
    const r = composeFromCode(`a = CFGNorm(model=model_50)\nb = Nope(x=1)`, { existing: EXISTING, objectInfo: OI });
    expect(r.problems).toHaveLength(1);
    // The plan of the valid line is still reported, so the caller can show it
    // — but operations are only honoured problem-free, and the caller knows.
    expect(r.operations.length).toBeGreaterThan(0);
  });
});

describe("composeFromCode: literals and comments", () => {
  it("reads Python literals: strings with escapes, numbers, booleans, None, lists", () => {
    const r = composeFromCode(
      `x = Anything(s="a \\"q\\" \\n b", t='single', n=-3, f=2.5, b=True, c=False, z=None, l=[1, "two", 3.0])`,
      { existing: {} },
    );
    expect(r.problems).toEqual([]);
    expect(r.operations[0]).toMatchObject({
      op: "add_node",
      widgets: { s: 'a "q" \n b', t: "single", n: -3, f: 2.5, b: true, c: false, z: null, l: [1, "two", 3] },
    });
  });

  it("ignores comments, blank lines, a trailing comment on a statement, and the to_code header", () => {
    const code = `# WORKFLOW: 1 nodes\n#\n\nx = Anything(a=1)  # "a title" | [bypass]\n`;
    const r = composeFromCode(code, { existing: {} });
    expect(r.problems).toEqual([]);
    expect(r.operations).toHaveLength(1);
  });

  it("accepts a bare call for a node nothing consumes", () => {
    const r = composeFromCode(`SaveImage(images=out0_50, filename_prefix="out")`, { existing: EXISTING });
    expect(r.problems).toEqual([]);
    expect(r.operations[0]).toMatchObject({ op: "add_node", class_type: "SaveImage" });
  });
});

describe("codeToWorkflow: a whole graph from code", () => {
  const WF: WorkflowJSON = {
    "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "a.safetensors" } },
    "2": { class_type: "CLIPTextEncode", inputs: { text: "a cat", clip: ["1", 1] } },
    "3": { class_type: "CFGNorm", inputs: { model: ["1", 0], strength: 1 } },
  };

  it("round-trips what to_code rendered, keeping the node ids the variables carry", () => {
    const back = codeToWorkflow(workflowToCode(WF, OI), OI);
    expect(back.problems).toEqual([]);
    expect(back.workflow).toEqual(WF);
  });

  it("numbers nodes whose variables carry no id, after the ones that do", () => {
    const back = codeToWorkflow(
      `model_7, clip_7, vae_7 = CheckpointLoaderSimple(ckpt_name="a.safetensors")\ncond = CLIPTextEncode(text="x", clip=clip_7)`,
      OI,
    );
    expect(back.problems).toEqual([]);
    expect(Object.keys(back.workflow)).toEqual(["7", "8"]);
    expect(back.workflow["8"].inputs.clip).toEqual(["7", 1]);
  });

  it("returns the problems instead of a partial graph", () => {
    const back = codeToWorkflow(`model_1 = Nope(a=1)`, OI);
    expect(back.problems).toHaveLength(1);
    expect(back.workflow).toEqual({});
  });
});

import type { ComfyUINodeDef, ObjectInfo, WorkflowJSON } from "../comfyui/types.js";

/**
 * The graph as pseudo-Python: one assignment per node in TOPOLOGICAL order,
 * every link a variable reference, every literal a keyword argument, and the
 * typed signatures of the classes used in a preamble.
 *
 *   # WORKFLOW: 3 nodes
 *   #   CLIPTextEncode(text: STRING, clip: CLIP) -> CONDITIONING
 *   model_1, clip_1, vae_1 = CheckpointLoaderSimple(ckpt_name="a.safetensors")
 *   conditioning_2 = CLIPTextEncode(text="a cat", clip=clip_1)  # "Positive"
 *   latent_3 = KSampler(seed=5, model=model_1, positive=conditioning_2)
 *
 * Why a third rendering next to `to_dsl` and Mermaid: ComfyBench (CVPR 2025)
 * measures how well a model reads and writes a ComfyUI graph in each shape.
 * Code scores 62.0 pass / 41.0 resolve; an element list — `to_dsl`'s shape,
 * `key <- id.slot` under a node header — 51.0 / 30.0; raw JSON 52.0 / 33.0.
 * In code a dependency is a variable the reader has already met, not an id to
 * resolve by hand, and typing the ports in the preamble is worth about ten
 * points more in the same ablation.
 *
 * This is a READ rendering. It is not parsed back (yet): a variable name is
 * `<output type>_<node id>`, so the id survives for every panel_* tool.
 */

const NOT_IDENTIFIER = /[^0-9A-Za-z_]/g;

/**
 * The callable spelling of a class name: `Image Save` is rendered `Image_Save`.
 * Exported so the reader can find the class again under that spelling.
 */
export function callableName(classType: string): string {
  return identifier(classType, "Node");
}

/** A name that holds as an identifier, without losing what it named. */
function identifier(name: unknown, fallback: string): string {
  const clean = String(name ?? "")
    .replace(NOT_IDENTIFIER, "_")
    .replace(/^_+|_+$/g, "");
  if (!clean) return fallback;
  if (/^\d/.test(clean)) return `${fallback}_${clean}`;
  return clean;
}

/**
 * The variable that carries one output. The output's type rather than a slot
 * number — `latent_70` reads, `out0_70` does not — and the node id, which is
 * what every other tool addresses the node by. Subgraph paths carry a colon
 * that no language accepts.
 */
function variable(nodeId: string, outputName: string | undefined, slot: number): string {
  const base = identifier(outputName, `out${slot}`).toLowerCase();
  return `${base || `out${slot}`}_${nodeId.replace(/:/g, "_")}`;
}

function literal(value: unknown, max = 90): string {
  if (typeof value === "boolean") return value ? "True" : "False";
  if (value === null || value === undefined) return "None";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") {
    let s = value.replace(/\n/g, "\\n");
    if (s.length > max) s = `${s.slice(0, max)}...`;
    return `"${s.replace(/"/g, "'")}"`;
  }
  let json: string;
  try {
    json = JSON.stringify(value);
  } catch {
    json = String(value);
  }
  return json.length > max ? `${json.slice(0, max)}...` : json;
}

function isLink(value: unknown, workflow: WorkflowJSON): value is [string | number, number] {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    (typeof value[0] === "string" || typeof value[0] === "number") &&
    typeof value[1] === "number" &&
    Number.isInteger(value[1]) &&
    Object.prototype.hasOwnProperty.call(workflow, String(value[0]))
  );
}

/**
 * Each node after the nodes it depends on. Depth-first post-order; the
 * `inProgress` mark stops on a cycle. ComfyUI refuses cycles at queue time,
 * but a graph mid-edit can carry one, and an infinite recursion would be a
 * crash where an imperfect rendering is due.
 */
function topologicalOrder(workflow: WorkflowJSON): string[] {
  const done = new Set<string>();
  const inProgress = new Set<string>();
  const order: string[] = [];
  const descend = (id: string): void => {
    if (done.has(id) || inProgress.has(id) || !(id in workflow)) return;
    inProgress.add(id);
    for (const value of Object.values(workflow[id].inputs ?? {})) {
      if (isLink(value, workflow)) descend(String(value[0]));
    }
    inProgress.delete(id);
    done.add(id);
    order.push(id);
  };
  for (const id of Object.keys(workflow)) descend(id);
  return order;
}

function outputNames(def: ComfyUINodeDef | undefined): string[] | undefined {
  const names = def?.output_name ?? def?.output;
  return Array.isArray(names) && names.length > 0 ? names.map(String) : undefined;
}

/**
 * Typed inputs and outputs of every class used. Without this preamble the
 * reader has to ask create_workflow (action:"node_info") once per class before
 * it can tell a link from a literal it has never seen.
 */
function signatures(classTypes: string[], objectInfo: ObjectInfo): string[] {
  const lines: string[] = [];
  for (const classType of classTypes) {
    const def = objectInfo[classType];
    if (!def) continue;
    const inputs: string[] = [];
    for (const group of ["required", "optional"] as const) {
      for (const [name, spec] of Object.entries(def.input?.[group] ?? {})) {
        if (!Array.isArray(spec)) continue;
        const type = Array.isArray(spec[0]) ? "COMBO" : String(spec[0]);
        inputs.push(`${name}${group === "optional" ? "?" : ""}: ${type}`);
      }
    }
    const outputs = outputNames(def)?.join(", ") ?? "none";
    lines.push(`#   ${classType}(${inputs.join(", ")}) -> ${outputs}`);
  }
  return lines;
}

function modeTag(mode: unknown): string | undefined {
  if (mode === "bypassed" || mode === "bypass" || mode === 4) return "[bypass]";
  if (mode === "muted" || mode === "mute" || mode === 2) return "[mute]";
  return undefined;
}

export function workflowToCode(workflow: WorkflowJSON, objectInfo?: ObjectInfo): string {
  const order = topologicalOrder(workflow);
  const header = [`# WORKFLOW: ${order.length} nodes`];
  if (order.length === 0) return `${header[0]}\n`;

  // Without object_info a node's outputs are unknown; the links tell which
  // slots something consumes, and only those get a variable.
  const consumedSlots = new Map<string, number>();
  for (const id of order) {
    for (const value of Object.values(workflow[id].inputs ?? {})) {
      if (!isLink(value, workflow)) continue;
      const src = String(value[0]);
      consumedSlots.set(src, Math.max(consumedSlots.get(src) ?? -1, value[1]));
    }
  }

  const classTypes: string[] = [];
  for (const id of order) {
    const t = workflow[id].class_type;
    if (t && !classTypes.includes(t)) classTypes.push(t);
  }

  header.push("#");
  header.push("# Each line assigns a node's outputs to variables. An argument that is a");
  header.push("# variable is a LINK, any other argument is a literal value. The order");
  header.push("# follows the dependencies. A variable nothing consumes is a dead branch.");
  const preamble = objectInfo ? signatures(classTypes, objectInfo) : [];
  if (preamble.length > 0) {
    header.push("#");
    header.push("# Signatures of the classes used, `?` marks an optional input:");
    header.push(...preamble);
  }

  const lines = [...header, ""];
  for (const id of order) {
    const node = workflow[id];
    const classType = String(node.class_type || "Unknown");
    const callable = callableName(classType);
    const def = objectInfo?.[classType];

    const names = outputNames(def);
    let targets = "";
    if (names) {
      targets = names.map((name, slot) => variable(id, name, slot)).join(", ");
    } else if (consumedSlots.has(id)) {
      const last = consumedSlots.get(id) ?? 0;
      targets = Array.from({ length: last + 1 }, (_, slot) => variable(id, undefined, slot)).join(", ");
    }

    const args: string[] = [];
    for (const [key, value] of Object.entries(node.inputs ?? {})) {
      if (isLink(value, workflow)) {
        const src = String(value[0]);
        const srcNames = outputNames(objectInfo?.[workflow[src].class_type]);
        args.push(`${identifier(key, "input")}=${variable(src, srcNames?.[value[1]], value[1])}`);
      } else {
        args.push(`${identifier(key, "arg")}=${literal(value)}`);
      }
    }

    const notes: string[] = [];
    if (callable !== classType) notes.push(`type ${JSON.stringify(classType)}`);
    const tag = modeTag(node._meta?.mode);
    if (tag) notes.push(`${tag} does not execute`);
    if (node._meta?.title) notes.push(JSON.stringify(node._meta.title));

    const call = `${callable}(${args.join(", ")})`;
    lines.push(`${targets ? `${targets} = ${call}` : call}${notes.length ? `  # ${notes.join(" | ")}` : ""}`);
  }
  return `${lines.join("\n").trimEnd()}\n`;
}

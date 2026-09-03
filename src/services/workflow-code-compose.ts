import type { ObjectInfo, WorkflowJSON } from "../comfyui/types.js";
import { callableName } from "./workflow-code.js";
import { parseCode, type CodeStatement, type CodeValue } from "./workflow-code-parse.js";

/**
 * Compose a fragment of graph by writing code.
 *
 * Why: chaining five nodes with the granular tools is five add_node calls and
 * six connect calls, each a round trip to the browser, with the agent holding
 * the ids handed out along the way. The same thing is five lines:
 *
 *   model_a = LoraLoaderModelOnly(model=model_50, lora_name="x.safetensors")
 *   model_b = ModelSamplingAuraFlow(model=model_a, shift=3.1)
 *   patched_model_71 = CFGNorm(model=model_b, strength=1.0)
 *
 * It is the shape `to_code` already renders the graph in: read and write in
 * the same language, so the agent never translates.
 *
 * Two rules, and they suffice:
 *
 *   - a variable whose name matches an EXISTING node, `model_50` for the
 *     `MODEL` output of node 50, refers to it;
 *   - a line whose target already exists MODIFIES that node, a line whose
 *     target is new CREATES one. The third line above therefore rewires the
 *     `model` input of node 71, without replacing it.
 *
 * Nothing here is applied. The plan and the problems come back together; a
 * caller applies nothing while a problem stands, because a half-applied
 * fragment is a graph that is harder to repair than a refusal.
 */

export type Operation =
  | { op: "add_node"; node: string; class_type: string; widgets: Record<string, unknown> }
  | { op: "set_widget"; node: string; name: string; value: unknown }
  | { op: "connect"; from: string; from_slot: number; to: string; to_input: string };

export interface ComposeResult {
  operations: Operation[];
  problems: string[];
  /** One human line per effect, in order. */
  plan: string[];
  /** Provisional ids of the nodes the fragment creates, in order. */
  created: string[];
}

export interface ComposeContext {
  /** The graph the fragment is applied to; `{}` for a graph built from scratch. */
  existing: WorkflowJSON;
  objectInfo?: ObjectInfo;
}

// `model_50`, `out0_328_3`: a suffix naming a node, a prefix naming its output.
// Subgraph ids carry an underscore where the API format has a colon.
const VARIABLE = /^([A-Za-z][A-Za-z0-9_]*?)_(\d+(?:_\d+)*)$/;

/** (node id, output name) when the variable names an existing node. */
function decompose(name: string, existing: WorkflowJSON): [string, string] | null {
  const m = VARIABLE.exec(name);
  if (!m) return null;
  // Try the colon spelling first: `328_3` must not be read as node 3 of a
  // prefix `out0_328`.
  for (const candidate of [m[2].replace(/_/g, ":"), m[2]]) {
    if (Object.prototype.hasOwnProperty.call(existing, candidate)) return [candidate, m[1]];
  }
  return null;
}

function outputNames(classType: string | undefined, objectInfo?: ObjectInfo): string[] {
  const def = classType ? objectInfo?.[classType] : undefined;
  const names = def?.output_name ?? def?.output;
  return Array.isArray(names) ? names.map(String) : [];
}

function outputIndex(outputName: string, classType: string | undefined, objectInfo?: ObjectInfo): number | null {
  const names = outputNames(classType, objectInfo);
  const i = names.findIndex((n) => n.toLowerCase() === outputName.toLowerCase());
  if (i >= 0) return i;
  const m = /^out(\d+)$/.exec(outputName);
  return m ? Number(m[1]) : null;
}

function declaredInputs(classType: string, objectInfo?: ObjectInfo): string[] | null {
  const def = objectInfo?.[classType];
  if (!def) return null;
  return [...Object.keys(def.input?.required ?? {}), ...Object.keys(def.input?.optional ?? {})];
}

function show(value: unknown): string {
  const s = JSON.stringify(value);
  return s.length > 60 ? `${s.slice(0, 60)}...` : s;
}

export function composeFromCode(code: string, ctx: ComposeContext): ComposeResult {
  const { existing } = ctx;
  // An EMPTY object_info knows nothing, which is not the same as "no class is
  // installed": treat it as absent rather than refusing every class.
  const objectInfo = ctx.objectInfo && Object.keys(ctx.objectInfo).length > 0 ? ctx.objectInfo : undefined;
  const parsed = parseCode(code);
  const problems = [...parsed.problems];
  const operations: Operation[] = [];
  const plan: string[] = [];
  const created: string[] = [];

  // Variables the fragment defines itself, bound to (node id, output slot).
  const fresh = new Map<string, [string, number]>();
  const usedIds = new Set(Object.keys(existing));
  const numeric = [...usedIds].filter((id) => /^\d+$/.test(id)).map(Number);
  let next = (numeric.length ? Math.max(...numeric) : 0) + 1;
  const allocate = (wanted: string | null): string => {
    if (wanted && !usedIds.has(wanted)) {
      usedIds.add(wanted);
      // Ids handed out afterwards continue past the one the variable carried.
      next = Math.max(next, Number(wanted) + 1);
      return wanted;
    }
    while (usedIds.has(String(next))) next++;
    const id = String(next++);
    usedIds.add(id);
    return id;
  };

  // The graph as the fragment sees it: what exists, plus what the fragment has
  // created so far, so `out0_7` can name a node created two lines above.
  const view: WorkflowJSON = { ...existing };

  // `to_code` renders a class whose name is not an identifier, `Image Save`,
  // as `Image_Save(...)`. Read that spelling back to the real class, looking
  // among the installed classes first, then among the classes already on the
  // graph (the only source when object_info is unreachable).
  const realClass = (written: string): string => {
    if (objectInfo?.[written] || Object.values(existing).some((n) => n.class_type === written)) return written;
    const installed = objectInfo ? Object.keys(objectInfo).find((c) => callableName(c) === written) : undefined;
    if (installed) return installed;
    const onGraph = Object.values(existing).map((n) => n.class_type).find((c) => callableName(c) === written);
    return onGraph ?? written;
  };

  const resolve = (name: string, line: number): [string, number] | null => {
    const bound = fresh.get(name);
    if (bound) return bound;
    const found = decompose(name, view);
    if (!found) {
      problems.push(
        `line ${line}: "${name}" names no node. A variable must read \`<output>_<node id>\`, for example \`model_50\`, or have been assigned earlier in this fragment.`,
      );
      return null;
    }
    const [nodeId, outputName] = found;
    const classType = view[nodeId]?.class_type;
    const index = outputIndex(outputName, classType, objectInfo);
    if (index === null) {
      const names = outputNames(classType, objectInfo);
      problems.push(
        `line ${line}: node ${nodeId} has no output "${outputName}". Outputs: ${names.join(", ") || "none known"}.`,
      );
      return null;
    }
    return [nodeId, index];
  };

  for (const st of parsed.statements) {
    const outcome = composeStatement(st);
    if (outcome) problems.push(outcome);
  }

  function composeStatement(st: CodeStatement): string | null {
    const { line, targets } = st;
    const classType = realClass(st.classType);

    // Does the first target name an existing node? Then this line modifies it.
    const first = targets[0];
    const existingTarget = first && !fresh.has(first) ? decompose(first, existing) : null;
    let nodeId: string;
    let isNew: boolean;
    if (existingTarget) {
      nodeId = existingTarget[0];
      const real = existing[nodeId]?.class_type;
      if (real !== classType) {
        return `line ${line}: node ${nodeId} is a ${real}, not a ${classType}. To replace it, remove it first.`;
      }
      isNew = false;
    } else {
      if (objectInfo && !objectInfo[classType]) {
        return `line ${line}: ${classType} is not installed. Search the class by name before creating it.`;
      }
      const carried = first ? VARIABLE.exec(first)?.[2].replace(/_/g, ":") ?? null : null;
      nodeId = allocate(carried && /^\d+$/.test(carried) ? carried : null);
      view[nodeId] = { class_type: classType, inputs: {} };
      isNew = true;
    }

    // Each target names one output; register them for the following lines.
    targets.forEach((name, i) => {
      const m = VARIABLE.exec(name);
      const index = m ? outputIndex(m[1], classType, objectInfo) : null;
      fresh.set(name, [nodeId, index ?? i]);
    });

    const declared = declaredInputs(classType, objectInfo);
    const widgets: Record<string, unknown> = {};
    const links: Array<[string, string, number]> = [];
    let failed: string | null = null;
    for (const { name, value } of st.args) {
      if (declared && !declared.includes(name)) {
        failed = `line ${line}: ${classType} has no input "${name}". Inputs: ${[...declared].sort().join(", ")}.`;
        break;
      }
      const bound = bindValue(value, line);
      if (bound === undefined) {
        failed = ""; // resolve() already recorded the problem
        break;
      }
      if (bound.kind === "link") links.push([name, bound.node, bound.slot]);
      else widgets[name] = bound.value;
    }
    if (failed !== null) return failed || null;

    if (isNew) {
      operations.push({ op: "add_node", node: nodeId, class_type: classType, widgets });
      created.push(nodeId);
      plan.push(`create node ${nodeId}: ${classType}${Object.keys(widgets).length ? ` (${Object.entries(widgets).map(([k, v]) => `${k}=${show(v)}`).join(", ")})` : ""}`);
    } else {
      for (const [name, value] of Object.entries(widgets)) {
        operations.push({ op: "set_widget", node: nodeId, name, value });
      }
      if (Object.keys(widgets).length) {
        plan.push(`node ${nodeId}: ${Object.entries(widgets).map(([k, v]) => `${k}=${show(v)}`).join(", ")}`);
      }
    }
    for (const [input, from, slot] of links) {
      operations.push({ op: "connect", from, from_slot: slot, to: nodeId, to_input: input });
      plan.push(`connect ${from}[${slot}] -> ${nodeId}.${input}`);
    }
    return null;
  }

  function bindValue(
    value: CodeValue,
    line: number,
  ): { kind: "link"; node: string; slot: number } | { kind: "value"; value: unknown } | undefined {
    if (value.kind === "literal") return { kind: "value", value: value.value };
    const source = resolve(value.name, line);
    if (!source) return undefined;
    return { kind: "link", node: source[0], slot: source[1] };
  }

  return { operations, problems, plan, created };
}

/**
 * A whole API-format workflow from code, for `visualize_workflow
 * (action:"from_code")`. Every target is new; a variable's id suffix is kept
 * as the node id when it is free, so what `to_code` rendered comes back with
 * the ids it carried.
 */
export function codeToWorkflow(code: string, objectInfo?: ObjectInfo): { workflow: WorkflowJSON; problems: string[] } {
  const r = composeFromCode(code, { existing: {}, objectInfo });
  if (r.problems.length > 0) return { workflow: {}, problems: r.problems };
  const workflow: WorkflowJSON = {};
  for (const op of r.operations) {
    if (op.op === "add_node") {
      workflow[op.node] = { class_type: op.class_type, inputs: { ...op.widgets } };
    } else if (op.op === "connect") {
      workflow[op.to].inputs[op.to_input] = [op.from, op.from_slot];
    }
  }
  return { workflow, problems: [] };
}

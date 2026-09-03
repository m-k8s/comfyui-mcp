// Custom skills for the lanes that cannot load skills natively (Ollama, the
// custom OpenAI-compatible endpoint, OpenRouter): those lanes reach the
// BUNDLED skills through list_packs action:"skill_list" / "skill_read", and
// nothing else. A user's own skills — the Claude lane loads them from
// ~/.claude/skills without asking — were out of reach. COMFYUI_MCP_SKILLS_DIRS
// names extra skill directories (path-delimited); their skills are listed and
// readable through the same two actions, with their origin disclosed. A name
// clash is settled in favour of the bundled skill, which the prompts refer to.

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { z } from "zod";

import { registerSkillsAccessTools } from "../../tools/skills-access.js";

type ToolResult = { isError?: boolean; content: Array<{ type: string; text?: string }> };
type Handler = (args: Record<string, unknown>) => Promise<ToolResult>;

function listPacks(): Handler {
  const tools: Array<{ name: string; handler: Handler }> = [];
  const server = {
    tool: (name: string, _desc: string, _shape: z.ZodRawShape, handler: Handler) => {
      tools.push({ name, handler });
    },
  };
  registerSkillsAccessTools(server as never);
  const t = tools.find((x) => x.name === "list_packs");
  if (!t) throw new Error("list_packs not registered");
  return t.handler;
}

const text = (r: ToolResult) => r.content.map((c) => c.text ?? "").join("\n");

function skill(dir: string, folder: string, name: string, description: string, body: string): void {
  mkdirSync(join(dir, folder), { recursive: true });
  writeFileSync(join(dir, folder, "SKILL.md"), `---\nname: ${name}\ndescription: ${description}\n---\n${body}\n`);
}

let extraA: string;
let extraB: string;
const OLD = process.env.COMFYUI_MCP_SKILLS_DIRS;

beforeEach(() => {
  extraA = mkdtempSync(join(tmpdir(), "cmcp-skills-a-"));
  extraB = mkdtempSync(join(tmpdir(), "cmcp-skills-b-"));
  skill(extraA, "mon-skill", "mon-skill", "Édition Qwen 2511, recette maison", "# Ma recette\n\nUtiliser le LoRA rank 32.");
  skill(extraB, "autre", "autre", "Un second dossier", "corps B");
  // A folder that shadows a bundled skill by name: the bundled one must win.
  skill(extraB, "civitai", "civitai", "usurpation", "PAS la skill bundlée");
  process.env.COMFYUI_MCP_SKILLS_DIRS = [extraA, extraB, join(extraA, "nexistepas")].join(delimiter);
});

afterEach(() => {
  if (OLD === undefined) delete process.env.COMFYUI_MCP_SKILLS_DIRS;
  else process.env.COMFYUI_MCP_SKILLS_DIRS = OLD;
  rmSync(extraA, { recursive: true, force: true });
  rmSync(extraB, { recursive: true, force: true });
});

describe("list_packs skills from COMFYUI_MCP_SKILLS_DIRS", () => {
  it("lists the custom skills after the bundled ones, each with its origin", async () => {
    const res = await listPacks()({ action: "skill_list" });
    expect(res.isError).toBeUndefined();
    const parsed = JSON.parse(text(res)) as { count: number; skills: Array<{ name: string; description: string; source?: string }> };
    const names = parsed.skills.map((s) => s.name);
    expect(names).toContain("mon-skill");
    expect(names).toContain("autre");
    expect(names.indexOf("civitai")).toBeLessThan(names.indexOf("mon-skill"));
    const mine = parsed.skills.find((s) => s.name === "mon-skill");
    expect(mine).toMatchObject({ description: "Édition Qwen 2511, recette maison", source: extraA });
    expect(parsed.skills.find((s) => s.name === "civitai")?.source).toBe("bundled");
    // The shadowing folder is not listed twice.
    expect(names.filter((n) => n === "civitai")).toHaveLength(1);
    expect(parsed.count).toBe(parsed.skills.length);
  });

  it("reads a custom skill's body, and keeps the bundled one on a name clash", async () => {
    const mine = await listPacks()({ action: "skill_read", name: "mon-skill" });
    expect(mine.isError).toBeUndefined();
    expect(text(mine)).toContain("Utiliser le LoRA rank 32.");
    const bundled = await listPacks()({ action: "skill_read", name: "civitai" });
    expect(bundled.isError).toBeUndefined();
    expect(text(bundled)).not.toContain("PAS la skill bundlée");
  });

  it("still refuses a name that is not a plain directory name", async () => {
    const res = await listPacks()({ action: "skill_read", name: "../mon-skill" });
    expect(res.isError).toBe(true);
  });

  it("ignores the variable when it is unset or names only missing directories", async () => {
    process.env.COMFYUI_MCP_SKILLS_DIRS = join(extraA, "nexistepas");
    const res = await listPacks()({ action: "skill_list" });
    const parsed = JSON.parse(text(res)) as { skills: Array<{ name: string }> };
    expect(parsed.skills.map((s) => s.name)).not.toContain("mon-skill");
  });
});

// An override is frozen at the moment it was edited; the default it replaced
// keeps evolving with releases (the skills rules of 2c7fb22 reached no override).
// The editor can only say so if the list carries the fact: `stale` is true when
// an override exists and differs from the CURRENT default.

import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  clearPromptOverride,
  listPrompts,
  registerPrompt,
  setPromptOverride,
} from "../../services/prompt-overrides.js";

const ID = "test.stale-override";
let dir: string;
const OLD = process.env.COMFYUI_MCP_PANEL_PROMPTS;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "cmcp-stale-"));
  process.env.COMFYUI_MCP_PANEL_PROMPTS = join(dir, "panel-prompts.json");
  registerPrompt(ID, "Stale probe", "DEFAULT, second edition", "help");
});

afterEach(() => {
  clearPromptOverride(ID);
  if (OLD === undefined) delete process.env.COMFYUI_MCP_PANEL_PROMPTS;
  else process.env.COMFYUI_MCP_PANEL_PROMPTS = OLD;
  rmSync(dir, { recursive: true, force: true });
});

const item = () => listPrompts().find((p) => p.id === ID);

describe("listPrompts marks an override that no longer matches the current default", () => {
  it("is not stale without an override", () => {
    expect(item()).toMatchObject({ overridden: false, stale: false });
  });

  it("is stale when the override text differs from the default", () => {
    setPromptOverride(ID, "DEFAULT, first edition, as the user saved it");
    expect(item()).toMatchObject({ overridden: true, stale: true });
  });

  it("is not stale when the override equals the current default, whitespace aside", () => {
    setPromptOverride(ID, "  DEFAULT, second edition\n");
    expect(item()).toMatchObject({ overridden: true, stale: false });
  });
});

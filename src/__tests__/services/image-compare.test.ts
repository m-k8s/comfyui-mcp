// get_image (action:"compare") — did an edit HAPPEN, by the pixels, not by eye.
//
// The failure it exists for: an edit workflow renders its input unchanged (an
// empty mask, a strength at zero, a switch left on the wrong side). A model
// that looks at the output sees exactly the image that was asked for, because
// it is the source image, and calls the edit a success. Sight cannot settle
// that question, so the question is taken away from it: a verdict, a rate, and
// a picture of where the change is.

import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { compareImages, compareRaw, renderChangeMap } from "../../services/image-compare.js";
import type { RawPixels } from "../../services/color-analysis.js";

/** A flat mid-grey W×H RGB image. */
function grey(width: number, height: number, value = 128): RawPixels {
  return { data: Buffer.alloc(width * height * 3, value), width, height, channels: 3 };
}

function clone(raw: RawPixels): RawPixels {
  return { ...raw, data: Buffer.from(raw.data) };
}

function paint(raw: RawPixels, x: number, y: number, rgb: [number, number, number]): void {
  const i = (y * raw.width + x) * 3;
  raw.data[i] = rgb[0];
  raw.data[i + 1] = rgb[1];
  raw.data[i + 2] = rgb[2];
}

describe("compareRaw", () => {
  it("calls two identical images unchanged, with certainty", () => {
    const a = grey(100, 100);
    const r = compareRaw(a, clone(a));
    expect(r.verdict).toBe("unchanged");
    expect(r.certainty).toBe("certain");
    expect(r.pixels_changed).toBe(0);
    expect(r.reason).toMatch(/identical/);
  });

  it("treats deviations within the tolerance as encoding noise, and warns against trusting the picture", () => {
    const a = grey(100, 100);
    const b = grey(100, 100, 130); // every pixel off by 2, under the default tolerance of 3
    const r = compareRaw(a, b);
    expect(r.verdict).toBe("unchanged");
    expect(r.certainty).toBe("very likely");
    expect(r.pixels_changed).toBe(0);
    expect(r.max_deviation).toBe(2);
    expect(r.caution).toMatch(/looks like what was asked/);
  });

  it("locates a concentrated edit: verdict, region, share of the image", () => {
    const a = grey(100, 100);
    const b = clone(a);
    for (let y = 30; y < 40; y++) for (let x = 20; x < 30; x++) paint(b, x, y, [255, 0, 0]);
    const r = compareRaw(a, b);
    expect(r.verdict).toBe("modified");
    expect(r.certainty).toBe("certain");
    expect(r.pixels_changed).toBe(100);
    expect(r.pixels_changed_pct).toBeCloseTo(1, 5);
    expect(r.region).toEqual({ x: 20, y: 30, width: 10, height: 10, share_of_image_pct: 1 });
    expect(r.spread).toBe("concentrated");
  });

  it("reads scattered single pixels as a diffuse change, the mark of a global re-encode", () => {
    const a = grey(100, 100);
    const b = clone(a);
    for (let y = 0; y < 100; y += 5) for (let x = 0; x < 100; x += 5) paint(b, x, y, [200, 200, 200]);
    const r = compareRaw(a, b);
    expect(r.verdict).toBe("modified");
    expect(r.spread).toBe("diffuse");
    expect(r.spread_note).toMatch(/re-encod/);
  });

  it("calls different dimensions a modification without comparing pixels", () => {
    const r = compareRaw(grey(100, 100), grey(120, 100));
    expect(r.verdict).toBe("modified");
    expect(r.certainty).toBe("certain");
    expect(r.reason).toMatch(/dimensions/);
    expect(r.pixels_changed).toBeUndefined();
  });

  it("honours a custom tolerance", () => {
    const a = grey(10, 10);
    const b = grey(10, 10, 140); // off by 12
    expect(compareRaw(a, b, 3).verdict).toBe("modified");
    expect(compareRaw(a, b, 20).verdict).toBe("unchanged");
  });
});

describe("renderChangeMap", () => {
  it("paints the changed pixels red over the dimmed edited image", async () => {
    const after = grey(40, 40);
    const mask = new Uint8Array(40 * 40);
    mask[5 * 40 + 7] = 1;
    const png = await renderChangeMap(after, mask);
    expect(png.mimeType).toBe("image/png");
    const bytes = Buffer.from(png.data, "base64");
    expect(bytes.subarray(0, 8)).toEqual(Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));

    const sharp = (await import("sharp")).default;
    const { data } = await sharp(bytes).raw().toBuffer({ resolveWithObject: true });
    const at = (x: number, y: number) => [data[(y * 40 + x) * 3], data[(y * 40 + x) * 3 + 1], data[(y * 40 + x) * 3 + 2]];
    expect(at(7, 5)[0]).toBe(255);
    expect(at(7, 5)[1]).toBeLessThan(80);
    // An unchanged pixel is a lightened grey, not red.
    const [r, g, b] = at(0, 0);
    expect(r).toBe(g);
    expect(g).toBe(b);
    expect(r).toBeGreaterThan(90);
  });
});

describe("compareImages", () => {
  it("compares two files by path and returns the verdict as JSON plus the change map", async () => {
    const sharp = (await import("sharp")).default;
    const dir = await mkdtemp(join(tmpdir(), "compare-"));
    const before = join(dir, "before.png");
    const after = join(dir, "after.png");
    const base = Buffer.alloc(32 * 32 * 3, 100);
    await writeFile(before, await sharp(base, { raw: { width: 32, height: 32, channels: 3 } }).png().toBuffer());
    const edited = Buffer.from(base);
    for (let y = 8; y < 16; y++) for (let x = 8; x < 16; x++) edited.fill(250, (y * 32 + x) * 3, (y * 32 + x) * 3 + 3);
    await writeFile(after, await sharp(edited, { raw: { width: 32, height: 32, channels: 3 } }).png().toBuffer());

    const res = await compareImages({ path: after, reference_path: before });
    const text = res.content.find((b) => b.type === "text");
    expect(text && text.type === "text" ? JSON.parse(text.text.slice(text.text.indexOf("{"))) : null).toMatchObject({
      verdict: "modified",
      region: { x: 8, y: 8, width: 8, height: 8 },
    });
    expect(res.content.some((b) => b.type === "image")).toBe(true);
  });

  it("requires a reference", async () => {
    await expect(compareImages({ path: "/nowhere/after.png" })).rejects.toThrow(/reference/);
  });
});

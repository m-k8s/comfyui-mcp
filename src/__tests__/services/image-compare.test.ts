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

import { compareImages, compareRaw, renderChangeMap, DEFAULT_TOLERANCE } from "../../services/image-compare.js";
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

/** A rectangular zone mask: 1 inside [x0,x1)×[y0,y1), 0 elsewhere. */
function zoneRect(width: number, height: number, x0: number, y0: number, x1: number, y1: number): Uint8Array {
  const z = new Uint8Array(width * height);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) z[y * width + x] = 1;
  return z;
}

function paintBlock(raw: RawPixels, x0: number, y0: number, x1: number, y1: number, rgb: [number, number, number]): void {
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) paint(raw, x, y, rgb);
}

describe("compareRaw with a zone", () => {
  it("confines the verdict to the zone and reports in-zone and out-of-zone stats", () => {
    const a = grey(100, 100);
    const b = clone(a);
    paintBlock(b, 20, 30, 30, 40, [255, 0, 0]); // 10×10 change inside the zone
    const zone = zoneRect(100, 100, 15, 25, 45, 45); // 30×20 = 600 px covering the change
    const r = compareRaw(a, b, DEFAULT_TOLERANCE, zone);
    expect(r.verdict).toBe("modified");
    expect(r.in_zone).toBeDefined();
    expect(r.in_zone?.pixels).toBe(600);
    expect(r.in_zone?.pixels_changed).toBe(100);
    expect(r.in_zone?.max_deviation).toBe(128);
    expect(r.out_of_zone?.pixels_changed).toBe(0);
    expect(r.leak).toBe(false);
  });

  it("flags a leak: change OUTSIDE the zone, nothing inside", () => {
    const a = grey(100, 100);
    const b = clone(a);
    paintBlock(b, 60, 60, 70, 70, [255, 0, 0]); // change well outside the zone
    const zone = zoneRect(100, 100, 0, 0, 20, 20);
    const r = compareRaw(a, b, DEFAULT_TOLERANCE, zone);
    expect(r.verdict).toBe("unchanged");
    expect(r.in_zone?.pixels_changed).toBe(0);
    expect(r.out_of_zone?.pixels_changed).toBe(100);
    expect(r.leak).toBe(true);
  });

  it("leaves the stats untouched when no zone is given (no regression)", () => {
    const r = compareRaw(grey(10, 10), clone(grey(10, 10)));
    expect(r.in_zone).toBeUndefined();
    expect(r.out_of_zone).toBeUndefined();
    expect(r.leak).toBeUndefined();
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

describe("compareImages with a zone", () => {
  const sharpP = import("sharp").then((m) => m.default);

  async function fixture(): Promise<{ before: string; after: string; dir: string }> {
    const sharp = await sharpP;
    const dir = await mkdtemp(join(tmpdir(), "compare-zone-"));
    const before = join(dir, "before.png");
    const after = join(dir, "after.png");
    const base = Buffer.alloc(32 * 32 * 3, 100);
    await writeFile(before, await sharp(base, { raw: { width: 32, height: 32, channels: 3 } }).png().toBuffer());
    const edited = Buffer.from(base);
    for (let y = 8; y < 16; y++) for (let x = 8; x < 16; x++) edited.fill(250, (y * 32 + x) * 3, (y * 32 + x) * 3 + 3);
    await writeFile(after, await sharp(edited, { raw: { width: 32, height: 32, channels: 3 } }).png().toBuffer());
    return { before, after, dir };
  }

  function statsOf(res: Awaited<ReturnType<typeof compareImages>>): Record<string, unknown> {
    const text = res.content.find((b) => b.type === "text");
    if (!text || text.type !== "text") throw new Error("no text block");
    return JSON.parse(text.text.slice(text.text.indexOf("{")));
  }

  it("takes a mask file, confines the stats to it, and still returns a map image", async () => {
    const sharp = await sharpP;
    const { before, after, dir } = await fixture();
    const maskPath = join(dir, "mask.png");
    const mask = Buffer.alloc(32 * 32, 0);
    for (let y = 8; y < 16; y++) for (let x = 8; x < 16; x++) mask[y * 32 + x] = 255;
    await writeFile(maskPath, await sharp(mask, { raw: { width: 32, height: 32, channels: 1 } }).png().toBuffer());

    const res = await compareImages({ path: after, reference_path: before, mask_path: maskPath });
    const stats = statsOf(res);
    expect(stats).toMatchObject({ verdict: "modified", leak: false });
    expect((stats.in_zone as { pixels: number }).pixels).toBe(64);
    expect((stats.in_zone as { pixels_changed: number }).pixels_changed).toBe(64);
    expect((stats.out_of_zone as { pixels_changed: number }).pixels_changed).toBe(0);
    expect(res.content.some((b) => b.type === "image")).toBe(true);
  });

  it("takes a bbox and confines the stats to it", async () => {
    const { before, after } = await fixture();
    const res = await compareImages({ path: after, reference_path: before, bbox: "8,8,8,8" });
    const stats = statsOf(res);
    expect((stats.in_zone as { pixels: number }).pixels).toBe(64);
    expect((stats.in_zone as { pixels_changed: number }).pixels_changed).toBe(64);
  });

  it("rejects a mask and a bbox given together", async () => {
    const { before, after, dir } = await fixture();
    const maskPath = join(dir, "mask.png");
    const sharp = await sharpP;
    await writeFile(maskPath, await sharp(Buffer.alloc(32 * 32, 255), { raw: { width: 32, height: 32, channels: 1 } }).png().toBuffer());
    await expect(
      compareImages({ path: after, reference_path: before, bbox: "0,0,4,4", mask_path: maskPath }),
    ).rejects.toThrow(/mutually exclusive/i);
  });

  it("rejects a malformed bbox", async () => {
    const { before, after } = await fixture();
    await expect(compareImages({ path: after, reference_path: before, bbox: "nope" })).rejects.toThrow(/bbox/i);
  });

  it("resizes a mask that does not match the image size", async () => {
    const sharp = await sharpP;
    const { before, after, dir } = await fixture();
    const maskPath = join(dir, "mask-small.png");
    // Half-resolution mask: white over 4..8 maps to 8..16 once scaled to 32×32.
    const mask = Buffer.alloc(16 * 16, 0);
    for (let y = 4; y < 8; y++) for (let x = 4; x < 8; x++) mask[y * 16 + x] = 255;
    await writeFile(maskPath, await sharp(mask, { raw: { width: 16, height: 16, channels: 1 } }).png().toBuffer());

    const res = await compareImages({ path: after, reference_path: before, mask_path: maskPath });
    const stats = statsOf(res);
    expect(stats).toMatchObject({ verdict: "modified" });
    expect((stats.in_zone as { pixels_changed: number }).pixels_changed).toBeGreaterThan(0);
  });
});

import { readFile } from "node:fs/promises";
import { isAbsolute, resolve } from "node:path";

import { ValidationError } from "../utils/errors.js";
import { resolveBytes, toRaw, type AnalyzeColorOptions, type RawPixels } from "./color-analysis.js";
import { requireSharp } from "./sharp-loader.js";

// ---------------------------------------------------------------------------
// get_image (action:"compare") — did an edit HAPPEN, by the pixels, not by eye.
//
// The failure this exists for is precise. An edit workflow sometimes renders
// its input unchanged: an empty mask, a strength at zero, a switch left on the
// wrong side, and nothing was edited. A model that looks at the result sees the
// image that was asked for, because it is the same image, and concludes the
// edit succeeded. Sight cannot settle that question, so it is taken away from
// it. Three answers, in order of importance:
//
//   a VERDICT, "unchanged" or "modified", founded on pixels, not an impression;
//   a RATE, the share of pixels that differ and by how much;
//   a MAP, where the change is, drawn over the edited image.
//
// The tolerance exists because two images can differ without an edit: a JPEG
// round trip, a VAE encode/decode, a resize. Under the threshold the difference
// is noise, and saying so prevents a success that did not happen.
// ---------------------------------------------------------------------------

/** Below this per-channel deviation (of 255) a pixel counts as unchanged. Two
 *  passes through a VAE move values by one or two levels with no edit at all. */
export const DEFAULT_TOLERANCE = 3;

/** Above this share of touched pixels it is an edit; below, encoding noise. */
const EDIT_THRESHOLD = 0.001; // one pixel in a thousand

const MAP_MAX_WIDTH = 900;

/** Measures over one part of the image: the zone, or everything outside it. */
export interface ZoneStats {
  pixels: number;
  pixels_changed: number;
  /** Fraction of the part's pixels that changed beyond the tolerance, 0..1. */
  changed_frac: number;
  mean_deviation: number;
  /** 95th percentile of the per-pixel deviation over the part. */
  p95_deviation: number;
  max_deviation: number;
}

export interface CompareStats {
  verdict: "unchanged" | "modified";
  certainty: "certain" | "very likely";
  reason: string;
  /** Present when the verdict is "unchanged" but the files are not byte-identical. */
  caution?: string;
  before: { width: number; height: number };
  after: { width: number; height: number };
  tolerance: number;
  pixels_changed?: number;
  pixels_changed_pct?: number;
  mean_deviation?: number;
  max_deviation?: number;
  /** Bounding box of the changed pixels, when there is an edit. */
  region?: { x: number; y: number; width: number; height: number; share_of_image_pct: number };
  spread?: "concentrated" | "diffuse" | "mixed";
  spread_note?: string;
  /** Measures inside the zone, when a zone (mask or bbox) is given. */
  in_zone?: ZoneStats;
  /** Measures outside the zone, the background noise, when a zone is given. */
  out_of_zone?: ZoneStats;
  /** True when a change spilled outside the zone: an inpaint that did not stay put. */
  leak?: boolean;
}

export interface CompareRawResult extends CompareStats {
  /** 1 where the pixel changed beyond the tolerance; empty when sizes differ. */
  mask: Uint8Array;
}

function round(n: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

/** Per-part accumulators, filled during the single comparison pass. */
interface ZoneAccumulator {
  pixels: number;
  changed: number;
  sum: number;
  max: number;
  /** Histogram of the per-pixel deviation (0..255), for the percentile. */
  hist: Float64Array;
}

function newAccumulator(): ZoneAccumulator {
  return { pixels: 0, changed: 0, sum: 0, max: 0, hist: new Float64Array(256) };
}

/** The p-th percentile of the deviations counted in a histogram. */
function histPercentile(hist: Float64Array, total: number, p: number): number {
  if (total === 0) return 0;
  const target = p * total;
  let cum = 0;
  for (let v = 0; v < 256; v++) {
    cum += hist[v];
    if (cum >= target) return v;
  }
  return 255;
}

function zoneStatsOf(acc: ZoneAccumulator): ZoneStats {
  return {
    pixels: acc.pixels,
    pixels_changed: acc.changed,
    changed_frac: round(acc.pixels ? acc.changed / acc.pixels : 0, 4),
    mean_deviation: round(acc.pixels ? acc.sum / acc.pixels : 0, 3),
    p95_deviation: histPercentile(acc.hist, acc.pixels, 0.95),
    max_deviation: acc.max,
  };
}

/**
 * Are these two decoded images different, by how much, and where?
 *
 * When a `zone` is given (1 = inside the zone, one entry per pixel), the same
 * pass also breaks the measures down into IN the zone and OUT of it, and the
 * verdict is decided on the zone alone: did the inpaint change what it was
 * meant to, and did anything leak outside. Without a zone the behaviour is
 * exactly as before, no breakdown, no leak.
 */
export function compareRaw(
  before: RawPixels,
  after: RawPixels,
  tolerance = DEFAULT_TOLERANCE,
  zone?: Uint8Array,
): CompareRawResult {
  const size = { before: { width: before.width, height: before.height }, after: { width: after.width, height: after.height } };
  if (before.width !== after.width || before.height !== after.height) {
    // A different size IS a modification, and comparing pixels after a resize
    // would mix the size change with the rest. Settled here, no further.
    return {
      verdict: "modified",
      certainty: "certain",
      reason: "the two images do not have the same dimensions.",
      ...size,
      tolerance,
      mask: new Uint8Array(0),
    };
  }

  const total = before.width * before.height;
  const ca = before.channels;
  const cb = after.channels;
  const channels = Math.min(ca, cb, 3);
  const mask = new Uint8Array(total);
  let changed = 0;
  let identical = true;
  let sum = 0;
  let max = 0;
  const inside = zone ? newAccumulator() : null;
  const outside = zone ? newAccumulator() : null;
  for (let p = 0; p < total; p++) {
    let worst = 0;
    for (let c = 0; c < channels; c++) {
      const d = Math.abs(before.data[p * ca + c] - after.data[p * cb + c]);
      if (d > worst) worst = d;
    }
    if (worst > 0) identical = false;
    sum += worst;
    if (worst > max) max = worst;
    const isChanged = worst > tolerance;
    if (isChanged) {
      mask[p] = 1;
      changed++;
    }
    if (zone) {
      const acc = zone[p] ? inside! : outside!;
      acc.pixels++;
      acc.sum += worst;
      if (worst > acc.max) acc.max = worst;
      acc.hist[worst]++;
      if (isChanged) acc.changed++;
    }
  }
  const share = total ? changed / total : 0;

  const base: CompareRawResult = {
    verdict: "unchanged",
    certainty: "certain",
    reason: "",
    ...size,
    tolerance,
    pixels_changed: changed,
    pixels_changed_pct: round(share * 100, 4),
    mean_deviation: round(total ? sum / total : 0, 3),
    max_deviation: max,
    mask,
  };

  if (zone) return zoneVerdict(base, inside!, outside!, before.width, before.height, tolerance);

  if (identical) {
    base.reason = "the two images are identical to the pixel. The workflow returned its input as it was.";
    return base;
  }
  if (share < EDIT_THRESHOLD) {
    base.certainty = "very likely";
    base.reason =
      `${changed} of ${total} pixels exceed the tolerance of ${tolerance}, i.e. ${(share * 100).toFixed(4)} %. ` +
      "A deviation that small and that scattered is encoding noise, not an edit.";
    base.caution =
      "Do not conclude success by looking at the image: it looks like what was asked for because it is the source image.";
    return base;
  }

  locateChanges(base, mask, before.width, before.height, changed);
  base.verdict = "modified";
  base.certainty = "certain";
  base.reason = `${changed} pixels (${(share * 100).toFixed(2)} %) changed beyond the tolerance of ${tolerance}.`;
  return base;
}

/** Where do the changed pixels sit? Fills region, spread, spread_note in place. */
function locateChanges(stats: CompareStats, mask: Uint8Array, width: number, height: number, changed: number): void {
  const total = width * height;
  let x0 = width;
  let x1 = -1;
  let y0 = height;
  let y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return;
  const boxArea = (x1 - x0 + 1) * (y1 - y0 + 1);
  const density = boxArea ? changed / boxArea : 0;
  stats.region = { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1, share_of_image_pct: round((boxArea / total) * 100, 2) };
  stats.spread = density > 0.5 ? "concentrated" : density < 0.15 ? "diffuse" : "mixed";
  if (stats.spread === "diffuse") {
    stats.spread_note =
      "The changed pixels are scattered across the region: the mark of a global re-encode rather than a local retouch.";
  }
}

/**
 * The verdict when a zone is in play. The zone decides "modified": did the
 * inpaint change what it was meant to. Everything outside is background noise,
 * and a change there is a leak, reported apart from the verdict.
 */
function zoneVerdict(
  base: CompareRawResult,
  inside: ZoneAccumulator,
  outside: ZoneAccumulator,
  width: number,
  height: number,
  tolerance: number,
): CompareRawResult {
  base.in_zone = zoneStatsOf(inside);
  base.out_of_zone = zoneStatsOf(outside);
  const inShare = inside.pixels ? inside.changed / inside.pixels : 0;
  const outShare = outside.pixels ? outside.changed / outside.pixels : 0;
  base.leak = outShare >= EDIT_THRESHOLD;
  locateChanges(base, base.mask, width, height, base.pixels_changed ?? 0);

  const leakNote = base.leak
    ? ` A leak: ${outside.changed} pixel(s) outside the zone changed too.`
    : "";
  if (inShare >= EDIT_THRESHOLD) {
    base.verdict = "modified";
    base.certainty = "certain";
    base.reason =
      `${inside.changed} of ${inside.pixels} zone pixels (${round(inShare * 100, 2)} %) ` +
      `changed beyond the tolerance of ${tolerance}.${leakNote}`;
  } else {
    base.verdict = "unchanged";
    base.certainty = "certain";
    base.reason =
      `the zone is unchanged: ${inside.changed} of ${inside.pixels} zone pixels ` +
      `exceed the tolerance of ${tolerance}.${leakNote}`;
    if (base.leak) {
      base.caution =
        "Nothing changed inside the zone, yet pixels changed outside it: the edit missed its target or spilled.";
    }
  }
  return base;
}

/**
 * The edited image, lightened and desaturated, with the changed pixels in red.
 * A black-and-white mask would say where, not on what; keeping the picture
 * underneath is what lets a reader say "the change is on the head" rather than
 * "there is white at the top left".
 */
export async function renderChangeMap(
  after: RawPixels,
  mask: Uint8Array,
  zone?: Uint8Array,
): Promise<{ data: string; mimeType: string }> {
  const sharp = await requireSharp("Image comparison");
  const total = after.width * after.height;
  const c = after.channels;
  const rgb = Buffer.alloc(total * 3);
  for (let p = 0; p < total; p++) {
    const r = after.data[p * c];
    const g = c > 1 ? after.data[p * c + 1] : r;
    const b = c > 2 ? after.data[p * c + 2] : r;
    let luma = Math.min(255, Math.round((0.299 * r + 0.587 * g + 0.114 * b) * 0.45 + 90));
    // Outside the zone, dim the background so the zone reads at a glance. The
    // changed pixels stay red everywhere, so a leak outside the zone is visible.
    if (zone && !zone[p]) luma = Math.round(luma * 0.45);
    if (mask[p]) {
      rgb[p * 3] = 255;
      rgb[p * 3 + 1] = Math.round(luma * 0.25);
      rgb[p * 3 + 2] = Math.round(luma * 0.25);
    } else {
      rgb[p * 3] = luma;
      rgb[p * 3 + 1] = luma;
      rgb[p * 3 + 2] = luma;
    }
  }
  let image = sharp(rgb, { raw: { width: after.width, height: after.height, channels: 3 } });
  if (after.width > MAP_MAX_WIDTH) image = image.resize({ width: MAP_MAX_WIDTH });
  const png = await image.png().toBuffer();
  return { data: png.toString("base64"), mimeType: "image/png" };
}

export interface CompareImagesOptions extends AnalyzeColorOptions {
  /** The BEFORE image: an absolute path, or a path under the ComfyUI output dir. */
  reference_path?: string;
  reference_asset_id?: string;
  reference_filename?: string;
  reference_subfolder?: string;
  reference_type?: "output" | "input" | "temp";
  tolerance?: number;
  /** Draw the change map (default true). */
  locate?: boolean;
  /** Restrict the comparison to a ZONE, by mask: luminance >= 128 marks it. */
  mask_path?: string;
  mask_asset_id?: string;
  mask_filename?: string;
  mask_subfolder?: string;
  mask_type?: "output" | "input" | "temp";
  /** Restrict the comparison to a ZONE, by box: "x,y,w,h" in image pixels. */
  bbox?: string;
}

export interface CompareImagesResult {
  content: Array<{ type: "text"; text: string } | { type: "image"; data: string; mimeType: string }>;
}

async function referenceBytes(opts: CompareImagesOptions): Promise<Buffer> {
  if (opts.reference_asset_id || opts.reference_filename) {
    return resolveBytes(
      {
        asset_id: opts.reference_asset_id,
        filename: opts.reference_filename,
        subfolder: opts.reference_subfolder,
        type: opts.reference_type,
      },
      'get_image (action:"compare") reference',
    );
  }
  if (opts.reference_path) {
    const p = opts.reference_path.trim();
    if (!p) throw new ValidationError("reference_path must be a non-empty string.");
    return isAbsolute(p) ? readFile(resolve(p)) : resolveBytes({ path: p }, 'get_image (action:"compare") reference');
  }
  throw new ValidationError(
    'get_image (action:"compare") needs the BEFORE image as a reference: reference_path, reference_filename (+reference_subfolder/reference_type), or reference_asset_id.',
  );
}

async function maskBytes(opts: CompareImagesOptions): Promise<Buffer> {
  if (opts.mask_asset_id || opts.mask_filename) {
    return resolveBytes(
      {
        asset_id: opts.mask_asset_id,
        filename: opts.mask_filename,
        subfolder: opts.mask_subfolder,
        type: opts.mask_type ?? "input",
      },
      'get_image (action:"compare") mask',
    );
  }
  const p = (opts.mask_path ?? "").trim();
  if (!p) throw new ValidationError("mask_path must be a non-empty string.");
  return isAbsolute(p) ? readFile(resolve(p)) : resolveBytes({ path: p }, 'get_image (action:"compare") mask');
}

/** Does the caller ask for a mask zone? */
function hasMask(opts: CompareImagesOptions): boolean {
  return Boolean(opts.mask_path || opts.mask_asset_id || opts.mask_filename);
}

/** A zone from a mask image: luminance >= 128 marks the zone, resized to fit. */
async function maskToZone(opts: CompareImagesOptions, width: number, height: number): Promise<Uint8Array> {
  const sharp = await requireSharp("Image comparison");
  const bytes = await maskBytes(opts);
  const { data, info } = await sharp(bytes, { limitInputPixels: 100_000_000 })
    .resize(width, height, { kernel: "lanczos3", fit: "fill" })
    .greyscale()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const stride = info.channels;
  const zone = new Uint8Array(width * height);
  for (let p = 0; p < zone.length; p++) zone[p] = data[p * stride] >= 128 ? 1 : 0;
  return zone;
}

/** A zone from a "x,y,w,h" box in image coordinates, clamped to the image. */
function bboxToZone(bbox: string, width: number, height: number): Uint8Array {
  const parts = bbox.split(",").map((s) => s.trim());
  const nums = parts.map((s) => Number(s));
  if (parts.length !== 4 || nums.some((n) => !Number.isInteger(n))) {
    throw new ValidationError('bbox must be four integers "x,y,w,h" in image pixels.');
  }
  const [x, y, w, h] = nums;
  const x0 = Math.max(0, x);
  const y0 = Math.max(0, y);
  const x1 = Math.min(width, x + w);
  const y1 = Math.min(height, y + h);
  const zone = new Uint8Array(width * height);
  for (let yy = y0; yy < y1; yy++) for (let xx = x0; xx < x1; xx++) zone[yy * width + xx] = 1;
  return zone;
}

/** Compare the edited image (the usual source triple) against its reference (the before). */
export async function compareImages(opts: CompareImagesOptions): Promise<CompareImagesResult> {
  const feature = 'get_image (action:"compare")';
  if (hasMask(opts) && opts.bbox) {
    throw new ValidationError('get_image (action:"compare"): mask and bbox are mutually exclusive, pass only one.');
  }
  const [beforeBytes, afterBytes] = await Promise.all([referenceBytes(opts), resolveBytes(opts, feature)]);
  const [before, after] = await Promise.all([toRaw(beforeBytes, "Image comparison"), toRaw(afterBytes, "Image comparison")]);

  let zone: Uint8Array | undefined;
  if (hasMask(opts)) {
    zone = await maskToZone(opts, after.width, after.height);
  } else if (opts.bbox) {
    zone = bboxToZone(opts.bbox, after.width, after.height);
  }
  if (zone && !zone.some((v) => v === 1)) {
    throw new ValidationError('get_image (action:"compare"): the zone is empty (mask all black, or bbox outside the image).');
  }

  const { mask, ...stats } = compareRaw(before, after, opts.tolerance ?? DEFAULT_TOLERANCE, zone);

  const content: CompareImagesResult["content"] = [
    {
      type: "text",
      text:
        `Image comparison — ${stats.verdict.toUpperCase()} (${stats.certainty})\n${stats.reason}\n\n` +
        JSON.stringify(stats, null, 2),
    },
  ];
  if (stats.region && opts.locate !== false) {
    const map = await renderChangeMap(after, mask, zone);
    content.push({
      type: "text",
      text: zone
        ? "Change map: in red, the changed pixels; the zone is bright, the rest dimmed to place it."
        : "Change map: in red, the pixels that changed; in dimmed grey, the edited image, to place the region.",
    });
    content.push({ type: "image", data: map.data, mimeType: map.mimeType });
  }
  return { content };
}

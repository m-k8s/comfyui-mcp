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
}

export interface CompareRawResult extends CompareStats {
  /** 1 where the pixel changed beyond the tolerance; empty when sizes differ. */
  mask: Uint8Array;
}

function round(n: number, digits: number): number {
  const f = 10 ** digits;
  return Math.round(n * f) / f;
}

/** Are these two decoded images different, by how much, and where? */
export function compareRaw(before: RawPixels, after: RawPixels, tolerance = DEFAULT_TOLERANCE): CompareRawResult {
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
  for (let p = 0; p < total; p++) {
    let worst = 0;
    for (let c = 0; c < channels; c++) {
      const d = Math.abs(before.data[p * ca + c] - after.data[p * cb + c]);
      if (d > worst) worst = d;
    }
    if (worst > 0) identical = false;
    sum += worst;
    if (worst > max) max = worst;
    if (worst > tolerance) {
      mask[p] = 1;
      changed++;
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

  // Where does the change sit?
  let x0 = before.width;
  let x1 = -1;
  let y0 = before.height;
  let y1 = -1;
  for (let y = 0; y < before.height; y++) {
    for (let x = 0; x < before.width; x++) {
      if (!mask[y * before.width + x]) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  const boxArea = (x1 - x0 + 1) * (y1 - y0 + 1);
  const density = boxArea ? changed / boxArea : 0;

  base.verdict = "modified";
  base.certainty = "certain";
  base.reason = `${changed} pixels (${(share * 100).toFixed(2)} %) changed beyond the tolerance of ${tolerance}.`;
  base.region = { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1, share_of_image_pct: round((boxArea / total) * 100, 2) };
  base.spread = density > 0.5 ? "concentrated" : density < 0.15 ? "diffuse" : "mixed";
  if (base.spread === "diffuse") {
    base.spread_note =
      "The changed pixels are scattered across the region: the mark of a global re-encode rather than a local retouch.";
  }
  return base;
}

/**
 * The edited image, lightened and desaturated, with the changed pixels in red.
 * A black-and-white mask would say where, not on what; keeping the picture
 * underneath is what lets a reader say "the change is on the head" rather than
 * "there is white at the top left".
 */
export async function renderChangeMap(after: RawPixels, mask: Uint8Array): Promise<{ data: string; mimeType: string }> {
  const sharp = await requireSharp("Image comparison");
  const total = after.width * after.height;
  const c = after.channels;
  const rgb = Buffer.alloc(total * 3);
  for (let p = 0; p < total; p++) {
    const r = after.data[p * c];
    const g = c > 1 ? after.data[p * c + 1] : r;
    const b = c > 2 ? after.data[p * c + 2] : r;
    const luma = Math.min(255, Math.round((0.299 * r + 0.587 * g + 0.114 * b) * 0.45 + 90));
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

/** Compare the edited image (the usual source triple) against its reference (the before). */
export async function compareImages(opts: CompareImagesOptions): Promise<CompareImagesResult> {
  const feature = 'get_image (action:"compare")';
  const [beforeBytes, afterBytes] = await Promise.all([referenceBytes(opts), resolveBytes(opts, feature)]);
  const [before, after] = await Promise.all([toRaw(beforeBytes, "Image comparison"), toRaw(afterBytes, "Image comparison")]);
  const { mask, ...stats } = compareRaw(before, after, opts.tolerance ?? DEFAULT_TOLERANCE);

  const content: CompareImagesResult["content"] = [
    {
      type: "text",
      text:
        `Image comparison — ${stats.verdict.toUpperCase()} (${stats.certainty})\n${stats.reason}\n\n` +
        JSON.stringify(stats, null, 2),
    },
  ];
  if (stats.verdict === "modified" && stats.region && opts.locate !== false) {
    const map = await renderChangeMap(after, mask);
    content.push({
      type: "text",
      text: "Change map: in red, the pixels that changed; in dimmed grey, the edited image, to place the region.",
    });
    content.push({ type: "image", data: map.data, mimeType: map.mimeType });
  }
  return { content };
}

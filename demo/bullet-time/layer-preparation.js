const CHUNK = 16_384;
const NEIGHBORS = [[-1, 0], [1, 0], [0, -1], [0, 1]];
const NEIGHBORS_8 = [
  [-1, 0], [1, 0], [0, -1], [0, 1],
  [-1, -1], [1, -1], [-1, 1], [1, 1],
];
const PATCH_OFFSETS = [
  [0, 0], [-1, 0], [1, 0], [0, -1], [0, 1],
  [-2, 0], [2, 0], [0, -2], [0, 2], [-2, -2], [2, -2], [-2, 2], [2, 2],
  [-4, 0], [4, 0], [0, -4], [0, 4], [-4, -4], [4, -4], [-4, 4], [4, 4],
];

const now = () => globalThis.performance?.now?.() ?? Date.now();
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));

function checkDimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2
      || width * height > 4_000_000) {
    throw new RangeError('Les dimensions de l’image sont invalides ou dépassent 4 millions de pixels.');
  }
}

function checkBuffers({ rgba, depth, width, height }) {
  checkDimensions(width, height);
  const count = width * height;
  if (!rgba || rgba.length < count * 4) throw new RangeError('L’image RGB est incomplète.');
  if (!depth || depth.length < count) throw new RangeError('La carte de profondeur est incomplète.');
  return count;
}

async function yieldToWorker() {
  await new Promise(resolve => setTimeout(resolve, 0));
}

function progress(onProgress, phase, done, total) {
  const ratio = total ? done / total : 1;
  onProgress?.({ phase, done, total, ratio, progress: ratio });
}

function sampledDepthRange(depth, count) {
  const stride = Math.max(1, Math.ceil(count / 65_536));
  const samples = [];
  for (let i = 0; i < count; i += stride) {
    const value = depth[i];
    if (Number.isFinite(value) && value > 0) samples.push(value);
  }
  if (!samples.length) return null;
  samples.sort((a, b) => a - b);
  const low = samples[Math.floor((samples.length - 1) * 0.05)];
  const high = samples[Math.floor((samples.length - 1) * 0.95)];
  const median = samples[Math.floor((samples.length - 1) * 0.5)];
  return { low, high, median };
}

/**
 * Grow a four-connected region from the selected point. Depth thresholds are
 * relative so this works with either relative inverse-depth or scene distance.
 */
export async function autoMaskFromDepth({ depth, width, height, pivot }, onProgress) {
  const started = now();
  checkDimensions(width, height);
  const count = width * height;
  if (!depth || depth.length < count) throw new RangeError('La carte de profondeur est incomplète.');
  const mask = new Uint8Array(count);
  const range = sampledDepthRange(depth, count);
  if (!range || range.high - range.low < Math.max(1e-5, range.median * 0.002)) {
    return { mask, detected: false, selectedPixels: 0, seconds: (now() - started) / 1000 };
  }

  const u = clamp(Number.isFinite(pivot?.u) ? pivot.u : 0.5, 0, 1);
  const v = clamp(Number.isFinite(pivot?.v) ? pivot.v : 0.5, 0, 1);
  const seedX = Math.min(width - 1, Math.floor(u * width));
  const seedY = Math.min(height - 1, Math.floor(v * height));
  const seed = seedY * width + seedX;
  const seedDepth = depth[seed];
  if (!Number.isFinite(seedDepth) || seedDepth <= 0) {
    return { mask, detected: false, selectedPixels: 0, seconds: (now() - started) / 1000 };
  }

  const queue = new Uint32Array(count);
  let head = 0, tail = 0;
  mask[seed] = 1; queue[tail++] = seed;
  const maxPixels = Math.floor(count * 0.55);
  const seedTolerance = Math.max(seedDepth * 0.12, (range.high - range.low) * 0.025);
  const stepFloor = Math.max(seedDepth * 0.006, range.median * 0.004);
  let processed = 0;
  progress(onProgress, 'mask', 0, count);

  while (head < tail && tail < maxPixels) {
    const index = queue[head++];
    const x = index % width, y = Math.floor(index / width);
    const here = depth[index];
    for (const [dx, dy] of NEIGHBORS) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const next = ny * width + nx;
      if (mask[next]) continue;
      const there = depth[next];
      if (!Number.isFinite(there) || there <= 0) continue;
      // Reject a strong local depth edge and prevent slow gradients from
      // carrying the flood far away from the clicked subject.
      if (Math.abs(there - here) > Math.max(Math.max(here, there) * 0.10, stepFloor)) continue;
      if (Math.abs(there - seedDepth) > seedTolerance) continue;
      mask[next] = 1;
      queue[tail++] = next;
    }
    processed++;
    if (processed % CHUNK === 0) {
      progress(onProgress, 'mask', head, count);
      await yieldToWorker();
    }
  }
  progress(onProgress, 'mask', tail, tail);
  // A flat plane or a seed on the background is too ambiguous to present as
  // a subject mask. Avoid returning a nearly full-frame selection.
  if (tail >= maxPixels) {
    mask.fill(0);
    return { mask, detected: false, selectedPixels: 0, seconds: (now() - started) / 1000 };
  }
  return { mask, detected: tail >= Math.max(4, Math.floor(count * 0.00002)),
    selectedPixels: tail, seconds: (now() - started) / 1000 };
}

async function buildSourceMap(mask, depth, width, height, distance, source, queue, onProgress) {
  const count = width * height;
  let head = 0, tail = 0;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      const index = row + x;
      if (!mask[index]) continue;
      let best = -1, bestDepth = -Infinity;
      for (const [dx, dy] of NEIGHBORS_8) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const neighbor = ny * width + nx;
        const value = depth[neighbor];
        if (!mask[neighbor] && Number.isFinite(value) && value > 0 && value > bestDepth) {
          best = neighbor; bestDepth = value;
        }
      }
      if (best >= 0) {
        distance[index] = 1;
        source[index] = best;
        queue[tail++] = index;
      }
    }
    if ((y + 1) % 64 === 0) {
      progress(onProgress, 'background-depth', ((y + 1) / height) * 0.2, 1);
      await yieldToWorker();
    }
  }

  head = 0;
  while (head < tail) {
    const index = queue[head++];
    const x = index % width, y = Math.floor(index / width);
    for (const [dx, dy] of NEIGHBORS_8) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const next = ny * width + nx;
      if (!mask[next] || distance[next]) continue;
      distance[next] = Math.min(65_535, distance[index] + 1);
      source[next] = source[index];
      queue[tail++] = next;
    }
    if (head % CHUNK === 0) {
      progress(onProgress, 'background-depth', 0.2 + 0.8 * head / Math.max(1, tail), 1);
      await yieldToWorker();
    }
  }
  return tail;
}

function donorPatchIsBackground(center, mask, width, height, pyramids) {
  const cx = center % width, cy = Math.floor(center / width);
  for (const [dx, dy] of PATCH_OFFSETS) {
    const x = cx + dx, y = cy + dy;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    if (mask[y * width + x]) return false;
  }
  for (const pyramid of pyramids) {
    const px = Math.floor(cx / pyramid.factor), py = Math.floor(cy / pyramid.factor);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const x = px + dx, y = py + dy;
      if (x < 0 || x >= pyramid.width || y < 0 || y >= pyramid.height) continue;
      if (!pyramid.clean[y * pyramid.width + x]) return false;
    }
  }
  return true;
}

function nearbyDonors(sourceIndex, mask, width, height, radius, pyramids, limit = 56) {
  const sx = sourceIndex % width, sy = Math.floor(sourceIndex / width);
  const step = radius >= 24 ? 4 : 3;
  const donors = [];
  // Search outward in square rings. Every center and every sampled patch
  // location is checked against the subject mask before it can be a donor.
  for (let ring = 0; ring <= radius && donors.length < limit; ring += step) {
    for (let dy = -ring; dy <= ring && donors.length < limit; dy += step) {
      for (let dx = -ring; dx <= ring && donors.length < limit; dx += step) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) < Math.max(0, ring - step)) continue;
        const x = sx + dx, y = sy + dy;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        const candidate = y * width + x;
        if (!mask[candidate] && donorPatchIsBackground(candidate, mask, width, height, pyramids)) donors.push(candidate);
      }
    }
  }
  if (!donors.length && donorPatchIsBackground(sourceIndex, mask, width, height, pyramids)) donors.push(sourceIndex);
  return donors;
}

function candidateScore(target, candidate, sourceCenter, layer, mask, distance, contextRGBA, donorRGBA, pyramids, width, height) {
  const tx = target % width, ty = Math.floor(target / width);
  const cx = candidate % width, cy = Math.floor(candidate / width);
  let score = 0, weightTotal = 0;
  // Fine native-resolution patch score. Known target context is matched
  // against original background or earlier, already filled layers.
  for (let n = 0; n < PATCH_OFFSETS.length; n++) {
    const [dx, dy] = PATCH_OFFSETS[n];
    const x = tx + dx, y = ty + dy;
    if (x < 0 || x >= width || y < 0 || y >= height) continue;
    const context = y * width + x;
    if (mask[context] && distance[context] >= layer) continue;
    const donorX = cx + dx, donorY = cy + dy;
    if (donorX < 0 || donorX >= width || donorY < 0 || donorY >= height) continue;
    const targetColor = context * 4, donorColor = (donorY * width + donorX) * 4;
    const dr = contextRGBA[targetColor] - donorRGBA[donorColor];
    const dg = contextRGBA[targetColor + 1] - donorRGBA[donorColor + 1];
    const db = contextRGBA[targetColor + 2] - donorRGBA[donorColor + 2];
    const scale = Math.max(Math.abs(dx), Math.abs(dy));
    const weight = scale <= 1 ? 1.25 : scale <= 2 ? 1 : 0.7;
    score += weight * (dr * dr + dg * dg + db * db) / (3 * 255 * 255);
    weightTotal += weight;
  }
  let fineScore = weightTotal ? score / weightTotal : 0;

  // Coarse patch scores compare real RGB pyramids at factors 2 and 4. Donor
  // cells are available only when their whole native block is outside mask;
  // target cells average known background and completed earlier fill pixels.
  let coarseSum = 0, coarseWeight = 0;
  for (const pyramid of pyramids) {
    const targetCellX = Math.floor(tx / pyramid.factor);
    const targetCellY = Math.floor(ty / pyramid.factor);
    const donorCellX = Math.floor(cx / pyramid.factor);
    const donorCellY = Math.floor(cy / pyramid.factor);
    let coarseSum = 0, coarseWeight = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const targetX = targetCellX + dx, targetY = targetCellY + dy;
      const donorX = donorCellX + dx, donorY = donorCellY + dy;
      if (targetX < 0 || targetX >= pyramid.width || targetY < 0 || targetY >= pyramid.height
          || donorX < 0 || donorX >= pyramid.width || donorY < 0 || donorY >= pyramid.height) continue;
      const targetCell = targetY * pyramid.width + targetX;
      const donorCell = donorY * pyramid.width + donorX;
      const targetCount = pyramid.count[targetCell];
      if (!targetCount || !pyramid.clean[donorCell]) continue;
      const dr = (pyramid.sumR[targetCell] / targetCount - pyramid.sumR[donorCell] / pyramid.fullCount[donorCell]) / 255;
      const dg = (pyramid.sumG[targetCell] / targetCount - pyramid.sumG[donorCell] / pyramid.fullCount[donorCell]) / 255;
      const db = (pyramid.sumB[targetCell] / targetCount - pyramid.sumB[donorCell] / pyramid.fullCount[donorCell]) / 255;
      const factorWeight = pyramid.factor === 2 ? 0.45 : 0.20;
      coarseSum += (dr * dr + dg * dg + db * db) * factorWeight;
      coarseWeight += factorWeight;
    }
  }
  const multiscaleScore = coarseWeight
    ? fineScore * 0.35 + coarseSum / (coarseWeight * 3) * 0.65
    : fineScore;
  const spatial = Math.hypot(cx - sxFor(sourceCenter, width), cy - syFor(sourceCenter, width));
  return multiscaleScore + spatial * 1e-7;
}

function sxFor(index, width) { return index % width; }
function syFor(index, width) { return Math.floor(index / width); }

function buildRgbPyramids(rgba, mask, width, height) {
  const pyramids = [];
  for (const factor of [2, 4]) {
    const pyramidWidth = Math.ceil(width / factor), pyramidHeight = Math.ceil(height / factor);
    const cells = pyramidWidth * pyramidHeight;
    const sumR = new Uint32Array(cells), sumG = new Uint32Array(cells), sumB = new Uint32Array(cells);
    const count = new Uint8Array(cells), fullCount = new Uint8Array(cells), clean = new Uint8Array(cells);
    clean.fill(1);
    const pyramid = { factor, width: pyramidWidth, height: pyramidHeight, sumR, sumG, sumB, count, fullCount, clean };
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const index = y * width + x, cell = Math.floor(y / factor) * pyramidWidth + Math.floor(x / factor);
      fullCount[cell]++;
      if (mask[index]) { clean[cell] = 0; continue; }
      count[cell]++;
      const offset = index * 4;
      sumR[cell] += rgba[offset]; sumG[cell] += rgba[offset + 1]; sumB[cell] += rgba[offset + 2];
    }
    pyramids.push(pyramid);
  }
  return pyramids;
}

function updatePyramids(pyramids, index, color, width) {
  const x = index % width, y = Math.floor(index / width);
  for (const pyramid of pyramids) {
    const cell = Math.floor(y / pyramid.factor) * pyramid.width + Math.floor(x / pyramid.factor);
    if (pyramid.count[cell] >= pyramid.fullCount[cell]) continue;
    pyramid.sumR[cell] += color[0]; pyramid.sumG[cell] += color[1]; pyramid.sumB[cell] += color[2];
    pyramid.count[cell]++;
  }
}

/**
 * Remove masked pixels from a background plate and fill only a narrow inner
 * band with clean, deterministic RGB patches sampled from outside the mask.
 */
export async function prepareLayerBuffers({ rgba, depth, width, height, mask: inputMask, amplitude = 1 }, onProgress) {
  const started = now();
  const count = checkBuffers({ rgba, depth, width, height });
  if (!inputMask || inputMask.length < count) throw new RangeError('Le masque du sujet est incomplet.');
  const mask = new Uint8Array(count);
  for (let i = 0; i < count; i++) mask[i] = inputMask[i] ? 1 : 0;
  if (!mask.some(Boolean)) throw new RangeError('Le masque du sujet est vide.');
  const output = new Uint8ClampedArray(count * 4);
  const backgroundDepth = new Float32Array(count);
  const valid = new Uint8Array(count);
  const estimated = new Uint8Array(count);
  const distance = new Uint16Array(count);
  const source = new Int32Array(count); source.fill(-1);
  const queue = new Uint32Array(count);

  // Copy known background pixels byte-for-byte. Masked RGB starts transparent
  // so even an accidental use of the invalid core cannot show the foreground.
  for (let i0 = 0; i0 < count; i0 += CHUNK) {
    const end = Math.min(count, i0 + CHUNK);
    for (let i = i0; i < end; i++) {
      if (mask[i]) continue;
      const rgb = i * 4;
      output[rgb] = rgba[rgb]; output[rgb + 1] = rgba[rgb + 1];
      output[rgb + 2] = rgba[rgb + 2]; output[rgb + 3] = rgba[rgb + 3];
      backgroundDepth[i] = depth[i];
      valid[i] = Number.isFinite(depth[i]) && depth[i] > 0 ? 1 : 0;
    }
    if (end < count) await yieldToWorker();
  }

  const sourceStarted = now();
  progress(onProgress, 'background-depth', 0, 1);
  await buildSourceMap(mask, depth, width, height, distance, source, queue, onProgress);
  const sourceMapSeconds = (now() - sourceStarted) / 1000;
  await yieldToWorker();
  const fillStarted = now();
  progress(onProgress, 'background', 0, 1);
  const pyramids = buildRgbPyramids(rgba, mask, width, height);

  // Prepare the maximum plate once. The amplitude can then change without
  // rebuilding the background; the Euclidean source distance keeps every
  // synthesized pixel inside the strict 5% long-side limit.
  const maxBand = Math.floor(Math.max(width, height) * 0.05);
  const requested = clamp(Number.isFinite(amplitude) ? amplitude : 1, 1, 2);
  const bandWidth = maxBand;
  const layerCounts = new Uint32Array(bandWidth + 1);
  for (let i0 = 0; i0 < count; i0 += CHUNK) {
    const end = Math.min(count, i0 + CHUNK);
    for (let i = i0; i < end; i++) {
      if (!mask[i]) continue;
      const nearest = source[i];
      const own = depth[i];
      const behind = Number.isFinite(own) && own > 0 ? own * 1.02 + 1e-5 : 0;
      if (nearest >= 0) backgroundDepth[i] = Math.max(depth[nearest], behind);
      else backgroundDepth[i] = behind;
      if (nearest < 0) continue;
      const euclidean = Math.hypot((i % width) - (nearest % width), Math.floor(i / width) - Math.floor(nearest / width));
      if (bandWidth > 0 && distance[i] <= bandWidth && euclidean <= bandWidth) layerCounts[distance[i]]++;
    }
    if (end < count) await yieldToWorker();
  }

  const fillCount = layerCounts.reduce((sum, n) => sum + n, 0);
  let done = 0;
  const layerOffsets = new Uint32Array(bandWidth + 2);
  for (let layer = 1; layer <= bandWidth; layer++) layerOffsets[layer + 1] = layerOffsets[layer] + layerCounts[layer];
  const layerIndices = new Uint32Array(fillCount);
  const layerCursors = layerOffsets.slice(0, bandWidth + 1);
  for (let i0 = 0; i0 < count; i0 += CHUNK) {
    const end = Math.min(count, i0 + CHUNK);
    for (let i = i0; i < end; i++) {
      if (mask[i] && distance[i] > 0 && distance[i] <= bandWidth && source[i] >= 0
          && Math.hypot((i % width) - (source[i] % width), Math.floor(i / width) - Math.floor(source[i] / width)) <= bandWidth) {
        layerIndices[layerCursors[distance[i]]++] = i;
      }
    }
    if (end < count) await yieldToWorker();
  }
  progress(onProgress, 'background', 0, Math.max(1, fillCount));
  const donorRadius = Math.min(40, Math.max(12, bandWidth + 8));
  for (let layer = 1; layer <= bandWidth; layer++) {
    for (let item = layerOffsets[layer]; item < layerOffsets[layer + 1]; item++) {
      const index = layerIndices[item];
      const donorList = nearbyDonors(source[index], mask, width, height, donorRadius, pyramids);
      let best = source[index], bestScore = Number.POSITIVE_INFINITY;
      for (const donor of donorList) {
        const score = candidateScore(index, donor, source[index], layer, mask, distance, output, rgba, pyramids, width, height);
        if (score < bestScore) { bestScore = score; best = donor; }
      }
      const targetOffset = index * 4, sourceOffset = best * 4;
      output[targetOffset] = rgba[sourceOffset]; output[targetOffset + 1] = rgba[sourceOffset + 1];
      output[targetOffset + 2] = rgba[sourceOffset + 2]; output[targetOffset + 3] = rgba[sourceOffset + 3];
      updatePyramids(pyramids, index, [rgba[sourceOffset], rgba[sourceOffset + 1], rgba[sourceOffset + 2]], width);
      valid[index] = Number.isFinite(backgroundDepth[index]) && backgroundDepth[index] > 0 ? 1 : 0;
      estimated[index] = 1;
      done++;
      if (done % CHUNK === 0) {
        progress(onProgress, 'background', done, Math.max(1, fillCount));
        await yieldToWorker();
      }
    }
    progress(onProgress, 'background', done, Math.max(1, fillCount));
    if (done < fillCount) await yieldToWorker();
  }
  const fillSeconds = (now() - fillStarted) / 1000;
  progress(onProgress, 'background', done, Math.max(done, 1));
  return { rgba: output, depth: backgroundDepth, valid, estimated, mask,
    bandWidth, requestedAmplitude: requested, sourceMapSeconds, fillSeconds,
    seconds: (now() - started) / 1000 };
}

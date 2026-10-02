const CHUNK_SIZE = 32_768;
const EDGE_THRESHOLD = 0.10;
const EDGE_BAND = 0.55;
const EDGE_REFINEMENT_RADIUS = 1.25;
const COLOR_SIGMA = 0.18;

const colorWeights = Float32Array.from({ length: 256 }, (_, i) => {
  const distanceSquared = i / 255;
  return Math.exp(-distanceSquared / (2 * COLOR_SIGMA * COLOR_SIGMA));
});

const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const now = () => globalThis.performance?.now?.() ?? Date.now();

function checkAbort(signal) {
  if (typeof signal?.throwIfAborted === 'function') signal.throwIfAborted();
  else if (signal?.aborted) throw signal.reason ?? new Error('Depth refinement was aborted.');
}

async function yieldIfNeeded(signal) {
  checkAbort(signal);
  await new Promise(resolve => setTimeout(resolve, 0));
  checkAbort(signal);
}

function colorDistance(guideR, guideG, guideB, a, b) {
  const dr = (guideR[a] - guideR[b]) / 255;
  const dg = (guideG[a] - guideG[b]) / 255;
  const db = (guideB[a] - guideB[b]) / 255;
  return Math.sqrt((dr * dr + dg * dg + db * db) / 3);
}

function refinementStrength(relativeDepthJump, rgbContrast) {
  const depthConfidence = clamp((relativeDepthJump - EDGE_THRESHOLD) / 0.45, 0, 1);
  const colorConfidence = clamp((rgbContrast - 0.06) / 0.45, 0, 1);
  return 0.16 + 0.74 * depthConfidence * colorConfidence;
}

/**
 * Upsample inverse depth to the native RGB grid, using RGB only around clear
 * low-resolution depth edges. edgeMask bits are left=1, right=2, up=4, down=8.
 */
export async function refineDepth(depth, dw, dh, rgba, w, h, signal) {
  const started = now();
  if (!Number.isInteger(dw) || !Number.isInteger(dh) || dw < 1 || dh < 1
      || !Number.isInteger(w) || !Number.isInteger(h) || w < 1 || h < 1) {
    throw new RangeError('Depth and RGB dimensions must be positive integers.');
  }
  const sourceCount = dw * dh;
  const outputCount = w * h;
  if (!depth || depth.length < sourceCount) throw new RangeError('Depth data is smaller than its dimensions.');
  if (!rgba || rgba.length < outputCount * 4) throw new RangeError('RGBA data is smaller than its dimensions.');
  checkAbort(signal);

  // Average native RGB into the depth-grid cells. This retains a stable guide
  // while giving the edge-aware pass an RGB value for every depth sample.
  const guideR = new Float32Array(sourceCount);
  const guideG = new Float32Array(sourceCount);
  const guideB = new Float32Array(sourceCount);
  const guideCount = new Uint32Array(sourceCount);
  const xScale = dw / w;
  const yScale = dh / h;
  let guidePixelsSinceYield = 0;
  for (let y = 0; y < h; y++) {
    const gy = Math.min(dh - 1, Math.floor((y + 0.5) * yScale));
    const rowOffset = y * w;
    for (let x0 = 0; x0 < w; x0 += CHUNK_SIZE) {
      const xEnd = Math.min(w, x0 + CHUNK_SIZE);
      for (let x = x0; x < xEnd; x++) {
        const gx = Math.min(dw - 1, Math.floor((x + 0.5) * xScale));
        const gridIndex = gy * dw + gx;
        const rgbIndex = (rowOffset + x) * 4;
        guideR[gridIndex] += rgba[rgbIndex];
        guideG[gridIndex] += rgba[rgbIndex + 1];
        guideB[gridIndex] += rgba[rgbIndex + 2];
        guideCount[gridIndex]++;
      }
      guidePixelsSinceYield += xEnd - x0;
      if (guidePixelsSinceYield >= CHUNK_SIZE) {
        guidePixelsSinceYield = 0;
        await yieldIfNeeded(signal);
      }
    }
  }
  for (let i0 = 0; i0 < sourceCount; i0 += CHUNK_SIZE) {
    const end = Math.min(sourceCount, i0 + CHUNK_SIZE);
    for (let i = i0; i < end; i++) {
      const count = guideCount[i];
      if (count) {
        guideR[i] /= count;
        guideG[i] /= count;
        guideB[i] /= count;
      }
    }
    if (end < sourceCount) await yieldIfNeeded(signal);
  }

  // Store the strength of each horizontal or vertical low-resolution edge.
  // Weak gradients stay on ordinary bilinear interpolation.
  const edgeRight = new Float32Array(sourceCount);
  const edgeDown = new Float32Array(sourceCount);
  let edgePixelsSinceYield = 0;
  for (let y = 0; y < dh; y++) {
    for (let x0 = 0; x0 < dw; x0 += CHUNK_SIZE) {
      const xEnd = Math.min(dw, x0 + CHUNK_SIZE);
      for (let x = x0; x < xEnd; x++) {
        const i = y * dw + x;
        const a = depth[i];
        if (!Number.isFinite(a)) continue;
        if (x + 1 < dw) {
          const j = i + 1, b = depth[j];
          if (Number.isFinite(b)) {
            const jump = Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-6);
            if (jump > EDGE_THRESHOLD) {
              const contrast = guideCount[i] && guideCount[j]
                ? colorDistance(guideR, guideG, guideB, i, j) : 0;
              edgeRight[i] = refinementStrength(jump, contrast);
            }
          }
        }
        if (y + 1 < dh) {
          const j = i + dw, b = depth[j];
          if (Number.isFinite(b)) {
            const jump = Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-6);
            if (jump > EDGE_THRESHOLD) {
              const contrast = guideCount[i] && guideCount[j]
                ? colorDistance(guideR, guideG, guideB, i, j) : 0;
              edgeDown[i] = refinementStrength(jump, contrast);
            }
          }
        }
      }
      edgePixelsSinceYield += xEnd - x0;
      if (edgePixelsSinceYield >= CHUNK_SIZE) {
        edgePixelsSinceYield = 0;
        await yieldIfNeeded(signal);
      }
    }
  }

  const values = new Float32Array(outputCount);
  const edgeMask = new Uint8Array(outputCount);
  let outputPixelsSinceYield = 0;
  for (let y = 0; y < h; y++) {
    const cellCoordinateY = (y + 0.5) * yScale;
    const cellY = clamp(Math.floor(cellCoordinateY), 0, dh - 1);
    const sourceY = clamp(cellCoordinateY - 0.5, 0, dh - 1);
    const y0 = Math.floor(sourceY), y1 = Math.min(dh - 1, y0 + 1);
    const fy = sourceY - y0;
    for (let x0 = 0; x0 < w; x0 += CHUNK_SIZE) {
      const xEnd = Math.min(w, x0 + CHUNK_SIZE);
      for (let x = x0; x < xEnd; x++) {
        const outIndex = y * w + x;
        const cellCoordinateX = (x + 0.5) * xScale;
        const cellX = clamp(Math.floor(cellCoordinateX), 0, dw - 1);
        const sourceX = clamp(cellCoordinateX - 0.5, 0, dw - 1);
        const x0Depth = Math.floor(sourceX), x1Depth = Math.min(dw - 1, x0Depth + 1);
        const fx = sourceX - x0Depth;
        const i00 = y0 * dw + x0Depth, i10 = y0 * dw + x1Depth;
        const i01 = y1 * dw + x0Depth, i11 = y1 * dw + x1Depth;
        const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
        const w01 = (1 - fx) * fy, w11 = fx * fy;
        let baseSum = 0, baseWeight = 0, localMin = Infinity, localMax = -Infinity;
        const v00 = depth[i00], v10 = depth[i10], v01 = depth[i01], v11 = depth[i11];
        if (Number.isFinite(v00)) {
          localMin = Math.min(localMin, v00); localMax = Math.max(localMax, v00);
          baseSum += v00 * w00; baseWeight += w00;
        }
        if (Number.isFinite(v10)) {
          localMin = Math.min(localMin, v10); localMax = Math.max(localMax, v10);
          baseSum += v10 * w10; baseWeight += w10;
        }
        if (Number.isFinite(v01)) {
          localMin = Math.min(localMin, v01); localMax = Math.max(localMax, v01);
          baseSum += v01 * w01; baseWeight += w01;
        }
        if (Number.isFinite(v11)) {
          localMin = Math.min(localMin, v11); localMax = Math.max(localMax, v11);
          baseSum += v11 * w11; baseWeight += w11;
        }
        let base = baseWeight > 0 ? baseSum / baseWeight : NaN;

        // Mark only the native pixels close to an actual four-direction depth
        // edge. The same nearby edges gate the RGB-guided refinement below.
        let mask = 0, refineMix = 0;
        const leftDistance = cellCoordinateX - cellX;
        const rightDistance = cellX + 1 - cellCoordinateX;
        const upDistance = cellCoordinateY - cellY;
        const downDistance = cellY + 1 - cellCoordinateY;
        if (cellX > 0) {
          const strength = edgeRight[cellY * dw + cellX - 1];
          if (strength > 0) {
            if (leftDistance <= EDGE_BAND) mask |= 1;
            if (leftDistance < EDGE_REFINEMENT_RADIUS) refineMix = Math.max(refineMix,
              strength * (1 - leftDistance / EDGE_REFINEMENT_RADIUS));
          }
        }
        if (cellX + 1 < dw) {
          const strength = edgeRight[cellY * dw + cellX];
          if (strength > 0) {
            if (rightDistance <= EDGE_BAND) mask |= 2;
            if (rightDistance < EDGE_REFINEMENT_RADIUS) refineMix = Math.max(refineMix,
              strength * (1 - rightDistance / EDGE_REFINEMENT_RADIUS));
          }
        }
        if (cellY > 0) {
          const strength = edgeDown[(cellY - 1) * dw + cellX];
          if (strength > 0) {
            if (upDistance <= EDGE_BAND) mask |= 4;
            if (upDistance < EDGE_REFINEMENT_RADIUS) refineMix = Math.max(refineMix,
              strength * (1 - upDistance / EDGE_REFINEMENT_RADIUS));
          }
        }
        if (cellY + 1 < dh) {
          const strength = edgeDown[cellY * dw + cellX];
          if (strength > 0) {
            if (downDistance <= EDGE_BAND) mask |= 8;
            if (downDistance < EDGE_REFINEMENT_RADIUS) refineMix = Math.max(refineMix,
              strength * (1 - downDistance / EDGE_REFINEMENT_RADIUS));
          }
        }
        edgeMask[outIndex] = mask;

        if (refineMix > 0) {
          const sampleX = Math.round(sourceX), sampleY = Math.round(sourceY);
          const rgbIndex = outIndex * 4;
          const targetR = rgba[rgbIndex], targetG = rgba[rgbIndex + 1], targetB = rgba[rgbIndex + 2];
          let guidedSum = 0, guidedWeight = 0, guidedMin = Infinity, guidedMax = -Infinity;
          for (let gy = sampleY - 1; gy <= sampleY + 1; gy++) {
            if (gy < 0 || gy >= dh) continue;
            const dy = Math.abs(sourceY - gy);
            const spatialY = Math.max(0, 1 - dy / 1.5);
            if (!spatialY) continue;
            for (let gx = sampleX - 1; gx <= sampleX + 1; gx++) {
              if (gx < 0 || gx >= dw) continue;
              const dx = Math.abs(sourceX - gx);
              const spatial = spatialY * Math.max(0, 1 - dx / 1.5);
              if (!spatial) continue;
              const i = gy * dw + gx, value = depth[i];
              if (!Number.isFinite(value)) continue;
              guidedMin = Math.min(guidedMin, value); guidedMax = Math.max(guidedMax, value);
              let colorWeight = 1;
              if (guideCount[i]) {
                const dr = (targetR - guideR[i]) / 255;
                const dg = (targetG - guideG[i]) / 255;
                const db = (targetB - guideB[i]) / 255;
                const distanceSquared = (dr * dr + dg * dg + db * db) / 3;
                const colorIndex = Math.min(255, Math.round(distanceSquared * 255));
                colorWeight = colorWeights[colorIndex];
              }
              const weight = spatial * colorWeight;
              guidedSum += value * weight;
              guidedWeight += weight;
            }
          }
          if (guidedWeight > 0) {
            const guided = guidedSum / guidedWeight;
            if (!Number.isFinite(base)) base = guided;
            else base += (guided - base) * Math.min(0.9, refineMix);
            localMin = Math.min(localMin, guidedMin);
            localMax = Math.max(localMax, guidedMax);
          }
        }

        if (!Number.isFinite(base)) {
          // Invalid-only bilinear support: use the closest finite value in the
          // local stencil, and leave wholly invalid areas at finite zero.
          let nearestDistance = Infinity;
          const nearX = Math.round(sourceX), nearY = Math.round(sourceY);
          for (let gy = nearY - 1; gy <= nearY + 1; gy++) {
            if (gy < 0 || gy >= dh) continue;
            for (let gx = nearX - 1; gx <= nearX + 1; gx++) {
              if (gx < 0 || gx >= dw) continue;
              const value = depth[gy * dw + gx];
              if (!Number.isFinite(value)) continue;
              localMin = Math.min(localMin, value);
              localMax = Math.max(localMax, value);
              const distance = (sourceX - gx) ** 2 + (sourceY - gy) ** 2;
              if (distance < nearestDistance) {
                nearestDistance = distance; base = value;
              }
            }
          }
        }
        if (!Number.isFinite(base)) base = 0;
        if (localMin <= localMax) base = clamp(base, localMin, localMax);
        values[outIndex] = base;
      }
      outputPixelsSinceYield += xEnd - x0;
      if (outputPixelsSinceYield >= CHUNK_SIZE) {
        outputPixelsSinceYield = 0;
        await yieldIfNeeded(signal);
      }
    }
  }

  checkAbort(signal);
  return { values, edgeMask, seconds: (now() - started) / 1000 };
}

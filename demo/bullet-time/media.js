import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  CanvasSink,
  CanvasSource,
  Input,
  Mp4OutputFormat,
  Output,
  Quality,
  WebMOutputFormat,
  canEncodeVideo,
} from "mediabunny";

const MAX_FILE_BYTES = 100 * 1024 * 1024;
const MAX_DURATION_SECONDS = 60;
const OUTPUT_FPS = 30;
const EFFECT_SECONDS = 2;
const EFFECT_FRAMES = OUTPUT_FPS * EFFECT_SECONDS;
const MAX_OUTPUT_SIDE = 1280;

function throwIfAborted(signal) {
  if (!signal?.aborted) return;
  if (signal.reason instanceof Error) throw signal.reason;
  throw new DOMException("Opération annulée.", "AbortError");
}

function waitForAbort(promise, signal) {
  if (!signal) return promise;
  throwIfAborted(signal);

  return new Promise((resolve, reject) => {
    const onAbort = () => reject(signal.reason instanceof Error
      ? signal.reason
      : new DOMException("Opération annulée.", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", onAbort));
  });
}

async function closeIterator(iterator, signal) {
  try {
    await iterator.return?.();
  } catch (error) {
    if (!signal?.aborted) throw error;
  }
}

function evenSize(width, height, maxSide) {
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const even = (value) => Math.max(2, Math.floor((value * scale) / 2) * 2);
  return { width: even(width), height: even(height) };
}

function report(onProgress, phase, progress, detail) {
  onProgress?.({ phase, progress: Math.max(0, Math.min(1, progress)), detail });
}

function selectedSourceIndex(duration, time) {
  const frameCount = Math.max(1, Math.ceil(duration * OUTPUT_FPS));
  const requested = Number.isFinite(time) ? Math.round(time * OUTPUT_FPS) : 0;
  return Math.max(0, Math.min(frameCount - 1, requested));
}

function createCanvas(width, height) {
  const canvas = typeof OffscreenCanvas !== "undefined"
    ? new OffscreenCanvas(width, height)
    : document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

async function encodeConfig(codec, width, height) {
  return canEncodeVideo(codec, {
    width,
    height,
    frameRate: OUTPUT_FPS,
    quality: new Quality("medium"),
  });
}

export async function chooseEncoding(width, height) {
  if (typeof VideoEncoder === "undefined") {
    throw new Error("Ce navigateur ne propose pas l’encodeur WebCodecs requis pour l’export.");
  }
  const quality = new Quality("medium");
  if (await encodeConfig("avc", width, height)) {
    return {
      codec: "avc",
      format: new Mp4OutputFormat(),
      extension: ".mp4",
      mime: "video/mp4",
      quality,
    };
  }

  if (await encodeConfig("vp9", width, height)) {
    return {
      codec: "vp9",
      format: new WebMOutputFormat(),
      extension: ".webm",
      mime: "video/webm",
      quality,
    };
  }

  if (await encodeConfig("vp8", width, height)) {
    return {
      codec: "vp8",
      format: new WebMOutputFormat(),
      extension: ".webm",
      mime: "video/webm",
      quality,
    };
  }

  throw new Error("Ce navigateur ne sait encoder ni en H.264, ni en VP9/VP8 via WebCodecs.");
}

export async function openVideo(file, signal) {
  throwIfAborted(signal);
  if (!file || typeof file.size !== "number") throw new Error("Fichier vidéo invalide.");
  if (file.size > MAX_FILE_BYTES) throw new Error("La vidéo dépasse la limite de 100 Mo.");
  if (typeof VideoDecoder === "undefined") {
    throw new Error("Ce navigateur ne propose pas WebCodecs pour décoder les vidéos.");
  }
  if (typeof VideoEncoder === "undefined") {
    throw new Error("Ce navigateur ne propose pas WebCodecs pour encoder une vidéo.");
  }

  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  let disposed = false;
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    input.dispose();
  };
  const cancelOpen = () => dispose();
  signal?.addEventListener("abort", cancelOpen, { once: true });

  try {
    if (!(await input.canRead())) {
      throw new Error("Ce format vidéo n’est pas pris en charge.");
    }
    const track = await input.getPrimaryVideoTrack();
    throwIfAborted(signal);
    if (!track) throw new Error("Aucune piste vidéo n’a été trouvée dans ce fichier.");

    const [rawStart, metadataEnd] = await Promise.all([
      track.getFirstTimestamp(),
      track.getDurationFromMetadata({ skipLiveWait: true }),
    ]);
    const start = Math.max(0, rawStart);
    if (metadataEnd !== null && metadataEnd - start > MAX_DURATION_SECONDS) {
      throw new Error("La vidéo dépasse la limite de 60 secondes.");
    }

    const end = await track.computeDuration({ skipLiveWait: true });
    const duration = end - start;
    throwIfAborted(signal);
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("La durée de cette vidéo est invalide.");
    if (duration > MAX_DURATION_SECONDS) throw new Error("La vidéo dépasse la limite de 60 secondes.");

    if (typeof VideoDecoder === "undefined" || !(await track.canDecode())) {
      throw new Error("Ce navigateur ne peut pas décoder le codec vidéo de ce fichier avec WebCodecs.");
    }

    const displayWidth = await track.getDisplayWidth();
    const displayHeight = await track.getDisplayHeight();
    if (!displayWidth || !displayHeight) throw new Error("Les dimensions de la vidéo sont invalides.");
    const { width, height } = evenSize(displayWidth, displayHeight, MAX_OUTPUT_SIDE);
    const encoding = await chooseEncoding(width, height);
    throwIfAborted(signal);

    const sink = new CanvasSink(track, {
      width,
      height,
      fit: "fill",
      poolSize: 2,
      alpha: false,
    });

    return { input, track, duration, start, width, height, sink, encoding, file, dispose };
  } catch (error) {
    dispose();
    if (signal?.aborted) throwIfAborted(signal);
    throw error;
  } finally {
    signal?.removeEventListener("abort", cancelOpen);
  }
}

export async function frameAt(media, time, signal) {
  throwIfAborted(signal);
  const relativeTime = selectedSourceIndex(media.duration, time) / OUTPUT_FPS;
  const wrapped = await waitForAbort(media.sink.getCanvas(media.start + relativeTime), signal);
  throwIfAborted(signal);
  if (!wrapped) throw new Error("Impossible de lire cette frame de la vidéo.");

  const canvas = createCanvas(media.width, media.height);
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Le navigateur ne peut pas créer un canvas 2D.");
  context.drawImage(wrapped.canvas, 0, 0, media.width, media.height);
  return canvas;
}

function median(values) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * fraction))];
}

function measureThumbnail(canvas, previousGray) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Le navigateur ne peut pas analyser les images vidéo.");
  const { width, height } = canvas;
  const rgba = context.getImageData(0, 0, width, height).data;
  const gray = new Float32Array(width * height);
  let motion = 0;

  for (let i = 0; i < gray.length; i++) {
    const pixel = i * 4;
    const value = 0.2126 * rgba[pixel] + 0.7152 * rgba[pixel + 1] + 0.0722 * rgba[pixel + 2];
    gray[i] = value;
    if (previousGray) motion += Math.abs(value - previousGray[i]);
  }
  if (previousGray) motion /= gray.length;

  let sum = 0;
  let squared = 0;
  let count = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const laplacian = 4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - width] - gray[i + width];
      sum += laplacian;
      squared += laplacian * laplacian;
      count++;
    }
  }
  const mean = count ? sum / count : 0;
  const sharpness = count ? Math.max(0, squared / count - mean * mean) : 0;
  return { gray, motion, sharpness };
}

export async function suggestMoment(media, signal, onProgress) {
  throwIfAborted(signal);
  const thumb = evenSize(media.width, media.height, 160);
  const sink = new CanvasSink(media.track, {
    width: thumb.width,
    height: thumb.height,
    fit: "fill",
    poolSize: 1,
    alpha: false,
  });
  const timestamps = [];
  // A coarse scan can rank the first frame of a new shot as an activity peak.
  for (let time = 0; time < media.duration; time += 0.125) timestamps.push(media.start + time);
  if (!timestamps.length) timestamps.push(media.start);

  const samples = [];
  let previousGray = null;
  let iteratorIndex = 0;
  const iterator = sink.canvasesAtTimestamps(timestamps)[Symbol.asyncIterator]();
  try {
    while (true) {
      const next = await waitForAbort(iterator.next(), signal);
      if (next.done) break;
      throwIfAborted(signal);
      const metrics = measureThumbnail(next.value.canvas, previousGray);
      previousGray = metrics.gray;
      samples.push({
        time: Math.max(0, Math.min(media.duration, next.value.timestamp - media.start)),
        motion: metrics.motion,
        sharpness: metrics.sharpness,
      });
      iteratorIndex++;
      report(onProgress, "suggest", iteratorIndex / timestamps.length, "Analyse du mouvement et de la netteté");
    }
  } finally {
    await closeIterator(iterator, signal);
  }

  if (!samples.length) throw new Error("Aucune frame exploitable n’a été trouvée.");
  const motionValues = samples.slice(1).map((sample) => sample.motion);
  const typicalMotion = median(motionValues);
  const motionDeviation = median(motionValues.map((value) => Math.abs(value - typicalMotion)));
  const cutThreshold = Math.max(20, typicalMotion + 4 * motionDeviation);
  const nearCut = new Set();
  samples.forEach((sample, index) => {
    if (sample.motion > cutThreshold) {
      for (let neighbor = Math.max(0, index - 4); neighbor <= Math.min(samples.length - 1, index + 4); neighbor++) {
        nearCut.add(neighbor);
      }
    }
  });

  let candidateIndices = samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample, index }) => !nearCut.has(index)
      && sample.time >= Math.min(0.5, media.duration / 4)
      && sample.time <= media.duration - Math.min(0.5, media.duration / 4));
  if (!candidateIndices.length) {
    candidateIndices = samples.map((sample, index) => ({ sample, index })).filter(({ index }) => !nearCut.has(index));
  }
  if (!candidateIndices.length) throw new Error('Aucun instant stable trouvé entre les changements de plan. Choisis un instant avec le curseur.');

  const motionScale = Math.max(1, percentile(candidateIndices.map(({ sample }) => sample.motion), 0.9));
  const sharpnessScale = Math.max(1, percentile(candidateIndices.map(({ sample }) => sample.sharpness), 0.9));
  const best = candidateIndices.reduce((winner, candidate) => {
    const score = 0.58 * Math.min(1, candidate.sample.motion / motionScale)
      + 0.42 * Math.min(1, candidate.sample.sharpness / sharpnessScale);
    return !winner || score > winner.score ? { ...candidate, score } : winner;
  }, null);

  report(onProgress, "suggest", 1, "Moment conseillé");
  return best.sample.time;
}

export async function exportVideo(media, { time, effect, signal, onProgress } = {}) {
  throwIfAborted(signal);
  if (!effect || typeof effect.render !== "function") {
    throw new Error("L’effet de profondeur n’est pas prêt pour l’export.");
  }
  const encoding = media.encoding ?? await chooseEncoding(media.width, media.height);
  const sourceFrameCount = Math.ceil(media.duration * OUTPUT_FPS);
  const selectedIndex = selectedSourceIndex(media.duration, time);
  const totalFrames = sourceFrameCount + EFFECT_FRAMES;
  const duration = totalFrames / OUTPUT_FPS;
  const timestamps = Array.from({ length: sourceFrameCount }, (_, index) => (
    media.start + Math.min(index / OUTPUT_FPS, Math.max(0, media.duration - 1e-6))
  ));
  const target = new BufferTarget();
  const output = new Output({ format: encoding.format, target });
  const startedAt = performance.now();
  const abortOutput = () => {
    if (output.state !== "canceled" && output.state !== "finalized") void output.cancel().catch(() => {});
  };
  signal?.addEventListener("abort", abortOutput, { once: true });

  let videoSource;
  let outputCanvas;
  let outputContext;
  let sourceIndex = 0;
  let encodedFrames = 0;
  const addFrame = async (canvas, timestamp) => {
    throwIfAborted(signal);
    outputContext.clearRect(0, 0, media.width, media.height);
    outputContext.drawImage(canvas, 0, 0, media.width, media.height);
    await waitForAbort(videoSource.add(timestamp, 1 / OUTPUT_FPS), signal);
    encodedFrames++;
    report(onProgress, "export", encodedFrames / totalFrames, `Encodage ${encodedFrames}/${totalFrames} frames`);
  };

  try {
    outputCanvas = createCanvas(media.width, media.height);
    outputContext = outputCanvas.getContext("2d");
    if (!outputContext) throw new Error("Le navigateur ne peut pas créer le canvas d’export.");
    videoSource = new CanvasSource(outputCanvas, {
      codec: encoding.codec,
      quality: encoding.quality ?? new Quality("medium"),
      latencyMode: "quality",
      keyFrameInterval: 2,
    });
    output.addVideoTrack(videoSource);
    await waitForAbort(output.start(), signal);

    const iterator = media.sink.canvasesAtTimestamps(timestamps)[Symbol.asyncIterator]();
    try {
      while (true) {
        const next = await waitForAbort(iterator.next(), signal);
        if (next.done) break;
        throwIfAborted(signal);
        if (sourceIndex >= sourceFrameCount) break;

        const outputTimestamp = sourceIndex / OUTPUT_FPS + (sourceIndex > selectedIndex ? EFFECT_SECONDS : 0);
        await addFrame(next.value.canvas, outputTimestamp);

        if (sourceIndex === selectedIndex) {
          for (let frame = 0; frame < EFFECT_FRAMES; frame++) {
            throwIfAborted(signal);
            const progress = frame / (EFFECT_FRAMES - 1);
            const generated = await effect.render(progress, signal);
            if (!generated) throw new Error("Le rendu de profondeur n’a pas produit de frame.");
            await addFrame(generated, (selectedIndex + 1 + frame) / OUTPUT_FPS);
          }
        }

        sourceIndex++;
      }
    } finally {
      await closeIterator(iterator, signal);
    }

    if (sourceIndex !== sourceFrameCount) {
      throw new Error("Le décodage s’est arrêté avant la fin de la vidéo.");
    }
    videoSource.close();
    throwIfAborted(signal);
    await waitForAbort(output.finalize(), signal);
    throwIfAborted(signal);
    const buffer = target.buffer;
    if (!buffer) throw new Error("Le fichier vidéo final est vide.");

    report(onProgress, "export", 1, "Vidéo prête");
    return {
      blob: new Blob([buffer], { type: encoding.mime }),
      extension: encoding.extension,
      mime: encoding.mime,
      duration,
      seconds: (performance.now() - startedAt) / 1000,
    };
  } catch (error) {
    if (output.state !== "canceled" && output.state !== "finalized") {
      try { await output.cancel(); } catch { /* The encoder may already have been canceled. */ }
    }
    throw error;
  } finally {
    signal?.removeEventListener("abort", abortOutput);
  }
}

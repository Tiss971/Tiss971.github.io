import { pipeline, RawImage, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.min.js';

env.allowLocalModels = false;
// Works without cross-origin isolation, including GitHub Pages.
env.backends.onnx.wasm.numThreads = 1;
const MODEL = 'onnx-community/depth-anything-v2-small';
let estimator;
let backend;

async function load(device) {
  const start = performance.now();
  const result = await pipeline('depth-estimation', MODEL, {
    device, dtype: device === 'webgpu' ? 'fp32' : 'q8',
    progress_callback: event => self.postMessage({ type: 'progress', event }),
  });
  backend = device;
  return { estimator: result, seconds: (performance.now() - start) / 1000 };
}

self.onmessage = async ({ data: { pixels, width, height } }) => {
  try {
    let loadSeconds = 0;
    if (!estimator) {
      let gpu = false;
      try { gpu = !!(await navigator.gpu?.requestAdapter()); } catch {}
      try {
        const loaded = await load(gpu ? 'webgpu' : 'wasm');
        estimator = loaded.estimator; loadSeconds = loaded.seconds;
      } catch (error) {
        if (!gpu) throw error;
        self.postMessage({ type: 'fallback' });
        const loaded = await load('wasm');
        estimator = loaded.estimator; loadSeconds = loaded.seconds;
      }
    }
    const image = new RawImage(new Uint8ClampedArray(pixels), width, height, 4);
    let start = performance.now();
    let result;
    try { result = await estimator(image); }
    catch (error) {
      if (backend !== 'webgpu') throw error;
      await estimator.dispose(); estimator = null;
      self.postMessage({ type: 'fallback' });
      const loaded = await load('wasm');
      estimator = loaded.estimator; loadSeconds += loaded.seconds;
      start = performance.now(); result = await estimator(image);
    }
    const depth = Float32Array.from(result.predicted_depth.data);
    const dims = result.predicted_depth.dims;
    self.postMessage({ type: 'result', depth, width: dims.at(-1), height: dims.at(-2),
      backend, loadSeconds, inferenceSeconds: (performance.now() - start) / 1000 }, [depth.buffer]);
  } catch (error) { self.postMessage({ type: 'error', message: error.message }); }
};

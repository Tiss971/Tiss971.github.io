let worker = null;
let active = false;

export function disposeDepth() { worker?.terminate(); worker = null; }

export async function estimateDepth(canvas, signal, onProgress) {
  signal.throwIfAborted();
  if (active) throw new Error('Une estimation de profondeur est déjà en cours.');
  active = true;
  const scale = Math.min(1, 384 / Math.max(canvas.width, canvas.height));
  const grid = document.createElement('canvas');
  grid.width = Math.max(2, Math.round(canvas.width * scale));
  grid.height = Math.max(2, Math.round(canvas.height * scale));
  const context = grid.getContext('2d', { willReadFrequently: true });
  context.drawImage(canvas, 0, 0, grid.width, grid.height);
  const pixels = context.getImageData(0, 0, grid.width, grid.height);
  try {
    worker ??= new Worker(new URL('./depth-worker.js', import.meta.url), { type: 'module' });
    const result = await new Promise((resolve, reject) => {
      const abort = () => { disposeDepth(); reject(signal.reason); };
      signal.addEventListener('abort', abort, { once: true });
      const finish = (callback, value) => {
        signal.removeEventListener('abort', abort);
        if (worker) { worker.onmessage = null; worker.onerror = null; }
        callback(value);
      };
      worker.onerror = event => { finish(reject, new Error(event.message || 'Impossible de charger le moteur de profondeur.')); disposeDepth(); };
      worker.onmessage = ({ data }) => {
        if (data.type === 'result') finish(resolve, data);
        else if (data.type === 'error') { finish(reject, new Error(data.message)); disposeDepth(); }
        else onProgress(data);
      };
      worker.postMessage({ pixels: pixels.data.buffer, width: grid.width, height: grid.height }, [pixels.data.buffer]);
    });
    signal.throwIfAborted();
    return { ...result, grid };
  } finally { active = false; }
}

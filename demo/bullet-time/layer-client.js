let nextId = 1;

function abortError(reason) {
  if (reason instanceof Error) return reason;
  try { return new DOMException(reason ? String(reason) : 'Annulé', 'AbortError'); }
  catch { const error = new Error(reason ? String(reason) : 'Annulé'); error.name = 'AbortError'; return error; }
}

function copyDepth(depth) {
  return depth instanceof Float32Array ? depth.slice() : Float32Array.from(depth ?? []);
}

function copyMask(mask) {
  return mask instanceof Uint8Array ? mask.slice() : Uint8Array.from(mask ?? []);
}

function copyRgba(rgba) {
  return rgba instanceof Uint8ClampedArray ? rgba.slice() : new Uint8ClampedArray(rgba ?? []);
}

/**
 * Create a local worker client for subject masks and background plates.
 * Pass the same AbortSignal used by the rest of the demo to terminate work.
 */
export function createLayerClient() {
  const active = new Map();
  let disposed = false;

  function run(action, payload, signal, onProgress) {
    if (disposed) return Promise.reject(new Error('Le client de reconstruction est fermé.'));
    if (signal?.aborted) return Promise.reject(abortError(signal.reason));
    if (typeof Worker === 'undefined') return Promise.reject(new Error('Les Web Workers sont indisponibles dans ce navigateur.'));

    let worker;
    try { worker = new Worker(new URL('./layer-worker.js', import.meta.url), { type: 'module' }); }
    catch (error) { return Promise.reject(new Error(`Impossible de démarrer le worker de reconstruction : ${error.message}`)); }
    const id = nextId++;

    return new Promise((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onAbort);
        worker.terminate();
        active.delete(worker);
      };
      const fail = error => { cleanup(); reject(error); };
      const onAbort = () => fail(abortError(signal?.reason));
      active.set(worker, () => fail(abortError()));
      signal?.addEventListener('abort', onAbort, { once: true });
      worker.onmessage = ({ data }) => {
        if (data?.id !== id) return;
        if (data.type === 'progress') {
          try { onProgress?.(data.event); } catch (error) { fail(error); }
        } else if (data.type === 'result') {
          cleanup(); resolve(data.result);
        } else if (data.type === 'error') {
          const error = new Error(data.error?.message ?? 'Échec de préparation des couches.');
          error.name = data.error?.name ?? 'Error';
          fail(error);
        }
      };
      worker.onerror = event => fail(new Error(event.message || 'Le worker de reconstruction a échoué.'));
      worker.onmessageerror = () => fail(new Error('La réponse du worker de reconstruction est illisible.'));
      try {
        const transfer = [];
        if (action === 'autoMask') {
          payload = { ...payload, depth: copyDepth(payload.depth) };
          transfer.push(payload.depth.buffer);
        } else if (action === 'prepareLayers') {
          payload = { ...payload, rgba: copyRgba(payload.rgba), depth: copyDepth(payload.depth), mask: copyMask(payload.mask) };
          transfer.push(payload.rgba.buffer, payload.depth.buffer, payload.mask.buffer);
        }
        worker.postMessage({ id, action, payload }, transfer);
      } catch (error) { fail(error); }
    });
  }

  return {
    autoMask(options, signal, onProgress) {
      return run('autoMask', options, signal, onProgress);
    },
    prepareLayers(options, signal, onProgress) {
      return run('prepareLayers', options, signal, onProgress);
    },
    dispose() {
      disposed = true;
      for (const cancel of active.values()) cancel();
    },
  };
}

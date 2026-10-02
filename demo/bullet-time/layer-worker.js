import { autoMaskFromDepth, prepareLayerBuffers } from './layer-preparation.js';

self.onmessage = async ({ data }) => {
  const { id, action, payload } = data ?? {};
  const report = event => self.postMessage({ id, type: 'progress', event });
  try {
    let result;
    if (action === 'autoMask') result = await autoMaskFromDepth(payload, report);
    else if (action === 'prepareLayers') result = await prepareLayerBuffers(payload, report);
    else throw new Error(`Action worker inconnue : ${action}`);

    const transfer = [];
    for (const key of ['mask', 'rgba', 'depth', 'valid', 'estimated']) {
      if (result[key]?.buffer instanceof ArrayBuffer) transfer.push(result[key].buffer);
    }
    self.postMessage({ id, type: 'result', result }, transfer);
  } catch (error) {
    self.postMessage({ id, type: 'error', error: {
      name: error?.name ?? 'Error', message: error?.message ?? 'Échec de préparation des couches.',
    } });
  }
};

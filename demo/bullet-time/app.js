import { openVideo, frameAt, suggestMoment, exportVideo } from './media.js';
import { estimateDepth, disposeDepth } from './depth.js';
import { createEffect, checkRendering } from './effect.js';

const el = Object.fromEntries([...document.querySelectorAll('[id]')].map(node => [node.id, node]));
let media = null, effect = null, effectTime = null;
let sourceURL = null, resultURL = null;
let controller = null, task = null, busy = false;
let previewToken = 0, previewTask = null, loadVersion = 0;
let trajectory = 'lateral', pivot = { u: .5, v: .5 }, picking = false, pivotPending = false;
let effectUpdates = Promise.resolve();
let effectUpdateRevision = 0;
let mode = 'v1', v2Ready = false, maskEditorOpen = false, maskHasSubject = false;
let maskBaseline = null, maskData = null, maskImage = null, maskWidth = 0, maskHeight = 0;
let maskStrokes = [], activeStroke = null, maskContext = null;
let motionController = null, motionPending = false;
const timings = {};
const seconds = n => `${n.toFixed(2).replace('.', ',')} s`;

function status(text, progress) {
  el.status.textContent = text;
  el.progress.hidden = progress === undefined;
  if (progress !== undefined) {
    if (progress === null) el.progress.removeAttribute('value');
    else el.progress.value = Math.min(100, Math.max(0, progress));
  }
}

function controls() {
  for (const id of ['timestamp']) el[id].disabled = busy || !media;
  for (const id of ['suggest', 'prepare']) el[id].disabled = busy || motionPending || !media || picking || pivotPending || maskEditorOpen;
  for (const id of ['intensity', 'trajectory']) el[id].disabled = busy || !media || picking || pivotPending || maskEditorOpen;
  el.generate.disabled = busy || motionPending || !effect || picking || pivotPending || maskEditorOpen;
  el.replay.disabled = busy || motionPending || !effect || picking || pivotPending || maskEditorOpen;
  el['pivot-pick'].disabled = busy || motionPending || !effect || pivotPending || maskEditorOpen;
  el['pivot-pick'].textContent = picking ? 'Annuler le choix' : 'Choisir le sujet';
  el['pivot-reset'].disabled = busy || motionPending || !effect || pivotPending || maskEditorOpen;
  el['pivot-overlay'].hidden = !picking;
  el['pivot-overlay'].disabled = busy || !picking || pivotPending;
  el['pivot-instructions'].hidden = !picking;
  el.cancel.hidden = !busy && !motionPending;
  el['v2-tools'].hidden = !effect;
  el['v2-enable'].hidden = !effect || v2Ready || maskEditorOpen;
  el['v2-enable'].disabled = busy || motionPending || !effect || picking || pivotPending;
  el['v2-editor'].hidden = !maskEditorOpen;
  el['mask-overlay'].hidden = !maskEditorOpen;
  el['mask-overlay'].style.pointerEvents = busy ? 'none' : 'auto';
  el['mask-undo'].disabled = busy || motionPending || !maskStrokes.length;
  el['mask-reset'].disabled = busy || motionPending || !maskStrokes.length;
  el['mask-brush'].disabled = busy || motionPending || !maskEditorOpen;
  el['mask-size'].disabled = busy || motionPending || !maskEditorOpen;
  el['v2-build'].disabled = busy || motionPending || !maskData || !maskHasSubject || !maskEditorOpen;
  el['v2-abandon'].disabled = busy || motionPending || !effect;
  el['v2-motion'].hidden = !v2Ready;
  el.amplitude.disabled = busy || !v2Ready || maskEditorOpen;
  el['v2-edit'].disabled = busy || motionPending || !v2Ready;
  el['v2-use'].hidden = !v2Ready || mode === 'v2';
  el['v2-use'].disabled = busy || motionPending || !effect;
  el['v2-v1'].hidden = !v2Ready || mode !== 'v2';
  el['v2-v1'].disabled = busy || motionPending || !effect;
  el['v2-fallback'].hidden = !v2Ready || !el['v2-fallback'].textContent;
  el.download.hidden = busy || !resultURL;
  // A new file or drop cancels and cleans up the current operation.
  el['drop-zone'].setAttribute('aria-busy', String(busy));
}

function showTimings() {
  el['timing-wrap'].hidden = !Object.keys(timings).length;
  el.timings.textContent = Object.entries(timings).map(([name, value]) => `${name} : ${typeof value === 'number' ? seconds(value) : value}`).join('\n');
}

function clearResult() {
  el.result.pause(); el.result.removeAttribute('src'); el.result.load();
  if (resultURL) URL.revokeObjectURL(resultURL);
  resultURL = null;
  el.download.removeAttribute('href'); el.download.hidden = true;
  el['result-wrap'].hidden = true;
}

async function stopPreview() {
  previewToken++;
  if (previewTask) await previewTask;
  previewTask = null;
}

async function playPreview() {
  await stopPreview();
  if (!effect || busy || picking) return;
  const token = ++previewToken, currentEffect = effect;
  el['preview-wrap'].hidden = false;
  el['source-wrap'].hidden = true;
  const context = el.preview.getContext('2d');
  previewTask = (async () => {
    const start = performance.now();
    while (token === previewToken) {
      const p = Math.min(1, (performance.now() - start) / 2000);
      const canvas = await currentEffect.render(p);
      if (token !== previewToken) break;
      context.drawImage(canvas, 0, 0);
      if (p >= 1) break;
      await new Promise(resolve => requestAnimationFrame(resolve));
    }
  })().catch(error => { if (token === previewToken) el.error.textContent = error.message; });
  await previewTask;
}

async function cancelWork() {
  if (motionController) {
    motionController.abort(new DOMException('Validation annulée', 'AbortError'));
    motionController = null;
    effectUpdateRevision++;
    motionPending = false;
    mode = 'v1';
    if (effect) effect.mode = 'v1';
  }
  controller?.abort(new DOMException('Traitement annulé', 'AbortError'));
  if (task) await task;
  await stopPreview();
  disposeDepth();
  controls();
  if (effect && !busy) void playPreview();
}

function updatePivotLabel() {
  const centered = pivot.u === .5 && pivot.v === .5;
  el['pivot-label'].value = centered
    ? 'Centre'
    : `Sujet · ${Math.round(pivot.u * 100)} %, ${Math.round(pivot.v * 100)} %`;
  el['pivot-label'].dataset.u = String(pivot.u);
  el['pivot-label'].dataset.v = String(pivot.v);
  positionPivotMarker();
}

function canvasContentRect() {
  const box = el.preview.getBoundingClientRect();
  if (!box.width || !box.height || !el.preview.width || !el.preview.height) return null;
  const aspect = el.preview.width / el.preview.height;
  let width = box.width, height = box.height;
  if (width / height > aspect) width = height * aspect;
  else height = width / aspect;
  return {
    left: box.left + (box.width - width) / 2,
    top: box.top + (box.height - height) / 2,
    width,
    height,
    boxLeft: box.left,
    boxTop: box.top
  };
}

function positionPivotMarker() {
  if (!el['pivot-marker']) return;
  const rect = canvasContentRect();
  if (!rect) return;
  el['pivot-marker'].style.left = `${rect.left - rect.boxLeft + pivot.u * rect.width}px`;
  el['pivot-marker'].style.top = `${rect.top - rect.boxTop + pivot.v * rect.height}px`;
}

function resetPickerState() {
  effectUpdateRevision++;
  picking = false;
  pivotPending = false;
  pivot = { u: .5, v: .5 };
  updatePivotLabel();
  controls();
}

function resetV2State() {
  maskEditorOpen = false;
  v2Ready = false;
  maskHasSubject = false;
  mode = 'v1';
  maskBaseline = maskData = maskImage = null;
  maskWidth = maskHeight = 0;
  maskStrokes = [];
  activeStroke = null;
  maskContext = null;
  if (effect) effect.mode = 'v1';
  el['v2-fallback'].textContent = '';
  if (el['mask-overlay']) {
    el['mask-overlay'].width = 1;
    el['mask-overlay'].height = 1;
    el['mask-overlay'].style.width = '';
    el['mask-overlay'].style.height = '';
  }
  controls();
}

function maskStatus() {
  maskHasSubject = Boolean(maskData?.some(Boolean));
  el['mask-undo'].disabled = busy || motionPending || !maskStrokes.length;
  el['mask-reset'].disabled = busy || motionPending || !maskStrokes.length;
  el['v2-build'].disabled = busy || motionPending || !maskData || !maskHasSubject || !maskEditorOpen;
}

function maskOverlayGeometry() {
  if (!maskEditorOpen || !maskWidth || !maskHeight) return;
  const rect = canvasContentRect();
  const parent = el['preview'].parentElement.getBoundingClientRect();
  if (!rect || !parent.width || !parent.height) return;
  const canvas = el['mask-overlay'];
  canvas.style.left = `${rect.left - parent.left}px`;
  canvas.style.top = `${rect.top - parent.top}px`;
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
}

function redrawMaskOverlay(dirty = null) {
  if (!maskData || !maskContext || !maskImage) return;
  const pixels = maskImage.data;
  const x0 = dirty ? Math.max(0, dirty.x0) : 0, x1 = dirty ? Math.min(maskWidth - 1, dirty.x1) : maskWidth - 1;
  const y0 = dirty ? Math.max(0, dirty.y0) : 0, y1 = dirty ? Math.min(maskHeight - 1, dirty.y1) : maskHeight - 1;
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const i = y * maskWidth + x, p = i * 4;
    if (maskData[i]) {
      pixels[p] = 35; pixels[p + 1] = 190; pixels[p + 2] = 255; pixels[p + 3] = 128;
    } else pixels[p + 3] = 0;
  }
  if (x1 >= x0 && y1 >= y0) maskContext.putImageData(maskImage, 0, 0, x0, y0, x1 - x0 + 1, y1 - y0 + 1);
  maskOverlayGeometry();
}

function setupMask(mask) {
  const source = effect?.source;
  maskWidth = source?.width || media?.width || 0;
  maskHeight = source?.height || media?.height || 0;
  const count = maskWidth * maskHeight;
  if (!mask || !maskWidth || !maskHeight || mask.length !== count) {
    throw new Error('Le masque du sujet ne correspond pas à la frame préparée. Recommence avec cette frame.');
  }
  maskBaseline = new Uint8Array(count);
  for (let i = 0; i < count; i++) maskBaseline[i] = mask[i] ? 1 : 0;
  maskData = maskBaseline.slice();
  maskHasSubject = maskData.some(Boolean);
  maskStrokes = [];
  const canvas = el['mask-overlay'];
  canvas.width = maskWidth; canvas.height = maskHeight;
  maskContext = canvas.getContext('2d', { willReadFrequently: true });
  if (!maskContext) throw new Error('Impossible de créer le calque de prévisualisation du masque.');
  maskImage = maskContext.createImageData(maskWidth, maskHeight);
  redrawMaskOverlay();
}

function brushRadius() {
  return Math.max(1, Math.round(Math.max(maskWidth, maskHeight) * Number(el['mask-size'].value) / 200));
}

function paintDisc(mask, x, y, radius, add) {
  const x0 = Math.max(0, Math.floor(x - radius)), x1 = Math.min(maskWidth - 1, Math.ceil(x + radius));
  const y0 = Math.max(0, Math.floor(y - radius)), y1 = Math.min(maskHeight - 1, Math.ceil(y + radius));
  const radiusSq = radius * radius;
  for (let py = y0; py <= y1; py++) for (let px = x0; px <= x1; px++) {
    if ((px - x) ** 2 + (py - y) ** 2 <= radiusSq) mask[py * maskWidth + px] = add ? 1 : 0;
  }
}

function applyStroke(mask, stroke) {
  const minX = Math.min(...stroke.points.map(point => point.x));
  const maxX = Math.max(...stroke.points.map(point => point.x));
  const minY = Math.min(...stroke.points.map(point => point.y));
  const maxY = Math.max(...stroke.points.map(point => point.y));
  const dirty = {
    x0: Math.max(0, Math.floor(minX - stroke.radius)),
    x1: Math.min(maskWidth - 1, Math.ceil(maxX + stroke.radius)),
    y0: Math.max(0, Math.floor(minY - stroke.radius)),
    y1: Math.min(maskHeight - 1, Math.ceil(maxY + stroke.radius)),
  };
  const step = Math.max(1, stroke.radius * .35);
  for (let i = 0; i < stroke.points.length; i++) {
    const point = stroke.points[i];
    if (i === 0) {
      paintDisc(mask, point.x, point.y, stroke.radius, stroke.add);
      continue;
    }
    const previous = stroke.points[i - 1];
    const distance = Math.hypot(point.x - previous.x, point.y - previous.y);
    const samples = Math.max(1, Math.ceil(distance / step));
    for (let sample = 1; sample <= samples; sample++) {
      const t = sample / samples;
      paintDisc(mask, previous.x + (point.x - previous.x) * t,
        previous.y + (point.y - previous.y) * t, stroke.radius, stroke.add);
    }
  }
  return dirty;
}

function rebuildMask() {
  if (!maskBaseline) return;
  maskData = maskBaseline.slice();
  for (const stroke of maskStrokes) applyStroke(maskData, stroke);
  redrawMaskOverlay();
  maskStatus();
}

function maskPoint(clientX, clientY) {
  const rect = canvasContentRect();
  if (!rect || !rect.width || !rect.height) return null;
  return {
    x: Math.max(0, Math.min(maskWidth - 1, (clientX - rect.left) / rect.width * maskWidth)),
    y: Math.max(0, Math.min(maskHeight - 1, (clientY - rect.top) / rect.height * maskHeight)),
  };
}

function setV2Fallback(message) {
  mode = 'v1';
  if (effect) effect.mode = 'v1';
  el['v2-fallback'].textContent = message || 'Aucune position sûre trouvée pour ces réglages. Le mouvement initial reste prêt ; réduis l’amplitude, change la trajectoire ou modifie le masque.';
  controls();
}

function updateAmplitudeLabel() {
  el['amplitude-label'].value = `${(Number(el.amplitude.value) / 100).toFixed(1).replace('.', ',')}×`;
  const retained = Number.isFinite(effect?.effectiveAmplitude) ? effect.effectiveAmplitude : Number(el.amplitude.value) / 100;
  const zoom = Number.isFinite(effect?.motionValidation?.zoom) ? ` · recadrage ${effect.motionValidation.zoom.toFixed(2).replace('.', ',')}×` : '';
  el['effective-amplitude'].textContent = `Amplitude retenue : ${retained.toFixed(1).replace('.', ',')}×${zoom}`;
}

function motionProgress(event) {
  const progress = Number(event?.progress);
  status('Vérification des positions du trajet…', Number.isFinite(progress) ? progress * 100 : null);
}

async function validateCurrentMotion(targetEffect, signal) {
  if (typeof targetEffect.validateMotion !== 'function') throw new Error('La validation du mouvement amplifié n’est pas disponible dans cette version.');
  const result = await targetEffect.validateMotion(signal, motionProgress);
  signal?.throwIfAborted();
  if (Number.isFinite(targetEffect.motionSeconds)) timings['Validation du trajet'] = targetEffect.motionSeconds;
  else delete timings['Validation du trajet'];
  updateAmplitudeLabel();
  showTimings();
  return result;
}

function queueEffectChange(change, { allowPicking = false, afterChange, validateV2 = false } = {}) {
  clearResult();
  const targetEffect = effect;
  motionController?.abort(new DOMException('Réglages modifiés', 'AbortError'));
  const updateController = new AbortController();
  motionController = updateController;
  const revision = ++effectUpdateRevision;
  if (!targetEffect) {
    motionController = null;
    return Promise.resolve(false);
  }
  motionPending = validateV2 || targetEffect.mode === 'v2';
  controls();
  const operation = effectUpdates.then(async () => {
    if (revision !== effectUpdateRevision || updateController.signal.aborted || busy || (!allowPicking && picking) || targetEffect !== effect) return false;
    await stopPreview();
    if (revision !== effectUpdateRevision || updateController.signal.aborted || busy || (!allowPicking && picking) || targetEffect !== effect) return false;
    change(targetEffect);
    if (validateV2 || targetEffect.mode === 'v2') {
      const result = await validateCurrentMotion(targetEffect, updateController.signal);
      if (revision !== effectUpdateRevision || targetEffect !== effect) return false;
      if (!result?.ok) {
        setV2Fallback(result?.message);
      } else {
        mode = 'v2';
        targetEffect.mode = 'v2';
        el['v2-fallback'].textContent = '';
        updateAmplitudeLabel();
      }
    }
    afterChange?.();
    controls();
    void playPreview();
    return true;
  });
  const handled = operation.catch(error => {
    if (revision === effectUpdateRevision && error.name !== 'AbortError') el.error.textContent = error.message || 'L’aperçu n’a pas pu être mis à jour.';
    if (revision === effectUpdateRevision && updateController.signal.aborted && targetEffect === effect) setV2Fallback('La vérification a été annulée. Le mouvement initial est disponible.');
    return false;
  }).finally(() => {
    if (motionController === updateController) {
      motionController = null;
      motionPending = false;
      controls();
    }
  });
  effectUpdates = handled.then(() => undefined);
  return handled;
}

function applyEffectSettings(currentEffect) {
  currentEffect.intensity = Number(el.intensity.value) / 100;
  currentEffect.trajectory = trajectory;
  currentEffect.pivot = { ...pivot };
  currentEffect.amplitude = Number(el.amplitude.value) / 100;
}

function progressMessage(label, event) {
  const progress = Number(event?.progress);
  status(label, Number.isFinite(progress) ? progress * 100 : null);
}

async function enableAmplifiedMode() {
  const targetEffect = effect;
  if (!targetEffect || busy || motionPending) return;
  clearResult();
  targetEffect.mode = 'v1';
  mode = 'v1';
  el['v2-fallback'].textContent = '';
  await runTask(async signal => {
    const started = performance.now();
    status('Détection du sujet à partir de la profondeur…', 0);
    const mask = await targetEffect.autoMask({ ...pivot }, signal,
      event => progressMessage('Détection du sujet à partir de la profondeur…', event));
    signal.throwIfAborted();
    if (targetEffect !== effect) return;
    setupMask(mask);
    const original = await targetEffect.render(0);
    signal.throwIfAborted();
    el.preview.width = media.width; el.preview.height = media.height;
    el.preview.getContext('2d').drawImage(original, 0, 0, media.width, media.height);
    el['preview-wrap'].hidden = false;
    el['source-wrap'].hidden = true;
    maskEditorOpen = true;
    v2Ready = false;
    mode = 'v1'; targetEffect.mode = 'v1';
    timings['Masque du sujet'] = Number.isFinite(targetEffect.maskSeconds)
      ? targetEffect.maskSeconds : (performance.now() - started) / 1000;
    redrawMaskOverlay();
    controls();
    status(maskData.some(Boolean)
      ? 'Masque prêt. Corrige les contours au pinceau, puis construis les couches.'
      : 'Aucun sujet isolé automatiquement. Peins la zone du sujet pour créer le masque.');
  });
}

async function buildAmplifiedLayers() {
  const targetEffect = effect;
  if (!targetEffect || !maskData || !maskEditorOpen || busy) return;
  const selectedMask = maskData.slice();
  let layersBuilt = false;
  clearResult();
  await runTask(async signal => {
    try {
      targetEffect.mode = 'v1'; mode = 'v1';
      const started = performance.now();
      status('Complétion du fond derrière le sujet…', 0);
      await targetEffect.prepareLayers(selectedMask, signal, event => {
        const phase = String(event?.phase || event?.stage || '').toLowerCase();
        const label = phase.includes('layer') || phase.includes('splat') || phase.includes('construction')
          ? 'Construction des couches de profondeur…' : 'Complétion du fond derrière le sujet…';
        progressMessage(label, event);
      });
      signal.throwIfAborted();
      if (targetEffect !== effect) return;
      layersBuilt = true;
      if (Number.isFinite(targetEffect.backgroundSeconds)) timings['Complétion du fond'] = targetEffect.backgroundSeconds;
      else timings['Complétion du fond'] = (performance.now() - started) / 1000;
      if (Number.isFinite(targetEffect.layerSeconds)) timings['Construction des couches'] = targetEffect.layerSeconds;
      targetEffect.mode = 'v2'; mode = 'v2';
      applyEffectSettings(targetEffect);
      status('Vérification des 60 positions du mouvement…', 0);
      const validation = await validateCurrentMotion(targetEffect, signal);
      signal.throwIfAborted();
      if (targetEffect !== effect) return;
      v2Ready = true;
      maskEditorOpen = false;
      if (!validation?.ok) {
        setV2Fallback(validation?.message);
        status('Aucune amplitude sûre pour ces réglages. Le mouvement initial reste disponible.');
      } else {
        mode = 'v2'; targetEffect.mode = 'v2';
        el['v2-fallback'].textContent = '';
        updateAmplitudeLabel();
        status('Couches prêtes. L’aperçu amplifié est activé.');
      }
      controls();
    } catch (error) {
      if (targetEffect === effect) {
        targetEffect.mode = 'v1'; mode = 'v1';
        if (layersBuilt) {
          v2Ready = true; maskEditorOpen = false;
          setV2Fallback('La validation a été annulée ou a échoué. Le mouvement initial reste disponible ; tu peux réessayer ou modifier le masque.');
        }
        controls();
      }
      throw error;
    }
  });
  if (effect && !busy && !maskEditorOpen) await playPreview();
}

async function editAmplifiedMask() {
  const targetEffect = effect;
  if (!targetEffect || !v2Ready || busy || motionPending || !maskData) return;
  clearResult();
  targetEffect.mode = 'v1'; mode = 'v1';
  v2Ready = false;
  el['v2-fallback'].textContent = '';
  await runTask(async signal => {
    const original = await targetEffect.render(0);
    signal.throwIfAborted();
    el.preview.getContext('2d').drawImage(original, 0, 0, media.width, media.height);
    el['preview-wrap'].hidden = false;
    el['source-wrap'].hidden = true;
    maskEditorOpen = true;
    redrawMaskOverlay();
    status('Modifie le masque, puis reconstruis les couches. La profondeur ne sera pas recalculée.');
  });
}

async function abandonAmplifiedMode() {
  const targetEffect = effect;
  if (!targetEffect || busy || motionPending) return;
  resetV2State();
  clearResult();
  targetEffect.mode = 'v1'; mode = 'v1';
  maskEditorOpen = false;
  controls();
  await playPreview();
  status('Mouvement initial activé.');
}

async function switchToV1() {
  if (!effect || busy || motionPending) return;
  await queueEffectChange(currentEffect => {
    currentEffect.mode = 'v1'; mode = 'v1';
  });
  status('Mouvement initial activé.');
}

async function switchToV2() {
  if (!effect || !v2Ready || busy || motionPending) return;
  await queueEffectChange(currentEffect => {
    applyEffectSettings(currentEffect);
    currentEffect.mode = 'v2'; mode = 'v2';
  }, { validateV2: true });
}

function contentPoint(clientX, clientY) {
  const rect = canvasContentRect();
  if (!rect || clientX < rect.left || clientX > rect.left + rect.width || clientY < rect.top || clientY > rect.top + rect.height) return null;
  return {
    u: Math.max(.01, Math.min(.99, (clientX - rect.left) / rect.width)),
    v: Math.max(.01, Math.min(.99, (clientY - rect.top) / rect.height))
  };
}

async function enterPicking() {
  if (!effect || busy || picking) return;
  effectUpdateRevision++;
  await runTask(async signal => {
    const currentEffect = effect;
    const original = await currentEffect.render(0);
    signal.throwIfAborted();
    if (currentEffect !== effect) return;
    el.preview.getContext('2d').drawImage(original, 0, 0);
    clearResult();
    el['preview-wrap'].hidden = false;
    el['source-wrap'].hidden = true;
    picking = true;
    controls();
    positionPivotMarker();
    status('Clique ou touche le sujet dans l’image. Entrée ou Espace choisit le centre de l’image ; Échap annule.');
  });
  if (picking) el['pivot-overlay'].focus();
}

async function cancelPicking() {
  if (!picking || busy || pivotPending) return;
  await stopPreview();
  picking = false;
  controls();
  el['pivot-pick'].focus();
  status('Choix annulé. L’aperçu précédent reprend.');
  await playPreview();
}

async function choosePivot(nextPivot) {
  if (!picking || busy || pivotPending || !effect) return;
  const previous = pivot;
  resetV2State();
  pivot = { u: nextPivot.u, v: nextPivot.v };
  updatePivotLabel();
  pivotPending = true;
  controls();
  const update = queueEffectChange(applyEffectSettings, {
    allowPicking: true,
    afterChange: () => {
      picking = false;
      pivotPending = false;
      controls();
      el['pivot-pick'].focus();
      status('Sujet choisi. L’aperçu se rejoue autour de ce point.');
    }
  });
  const revision = effectUpdateRevision;
  if (!await update && revision === effectUpdateRevision) {
    pivot = previous;
    pivotPending = false;
    updatePivotLabel();
    controls();
  }
}

async function fitPivotToCenter() {
  if (!effect || busy || pivotPending) return;
  if (!picking && pivot.u === .5 && pivot.v === .5) {
    status('Le centre de l’image est déjà sélectionné.');
    return;
  }
  const previous = pivot;
  resetV2State();
  pivot = { u: .5, v: .5 };
  updatePivotLabel();
  pivotPending = true;
  controls();
  const update = queueEffectChange(applyEffectSettings, {
    allowPicking: true,
    afterChange: () => {
      picking = false;
      pivotPending = false;
      controls();
      el['pivot-pick'].focus();
      status('Centre de l’image sélectionné.');
    }
  });
  const revision = effectUpdateRevision;
  if (!await update && revision === effectUpdateRevision) {
    pivot = previous;
    pivotPending = false;
    updatePivotLabel();
    controls();
  }
}

function runTask(work) {
  if (busy) return task;
  busy = true; controller = new AbortController();
  const signal = controller.signal;
  el.source.pause(); el.result.pause(); el.error.textContent = '';
  controls();
  task = (async () => {
    try { await stopPreview(); signal.throwIfAborted(); await work(signal); }
    catch (error) {
      if (signal.aborted || error.name === 'AbortError') status('Traitement annulé. Tu peux réessayer.');
      else {
        el.error.textContent = error.message || 'Le traitement a échoué.';
        status('Le traitement n’a pas abouti.');
      }
    } finally {
      busy = false; controller = null; task = null;
      el.progress.hidden = true; controls(); showTimings();
    }
  })();
  return task;
}

async function loadFile(file) {
  if (!file) return;
  const version = ++loadVersion;
  effectUpdateRevision++;
  await cancelWork();
  if (version !== loadVersion) return;
  clearResult(); resetV2State(); effect?.dispose(); effect = null; effectTime = null;
  resetPickerState();
  disposeDepth(); media?.dispose(); media = null;
  if (sourceURL) URL.revokeObjectURL(sourceURL);
  sourceURL = null; el.source.removeAttribute('src'); el.source.load();
  el['preview-wrap'].hidden = el['source-wrap'].hidden = true;
  el['drop-zone'].hidden = false;
  for (const key of Object.keys(timings)) delete timings[key];
  await runTask(async signal => {
    status('Vérification de la vidéo et des capacités de ton navigateur…', null);
    checkRendering();
    media = await openVideo(file, signal);
    sourceURL = URL.createObjectURL(file); el.source.src = sourceURL;
    el['file-name'].textContent = file.name;
    el.metadata.textContent = `${media.width} × ${media.height} · ${seconds(media.duration)}`;
    el.timestamp.max = Math.max(0, Math.ceil(media.duration * 30) / 30 - 1 / 30);
    el.timestamp.value = Math.floor(media.duration / 2 * 30) / 30;
    el.source.addEventListener('loadedmetadata', seekSelection, { once: true });
    updateTimeLabel();
    el['drop-zone'].hidden = true; el['source-wrap'].hidden = false;
    status('Choisis ton instant, ou laisse la démo en suggérer un.');
  });
}

function updateTimeLabel() { el['time-label'].value = seconds(Number(el.timestamp.value)); }
function seekSelection() {
  // Avoid seeking just before a frame boundary due to the range step rounding.
  el.source.currentTime = media.start + Math.round(Number(el.timestamp.value) * 30) / 30 + 1e-6;
}

async function changeSelection(time) {
  effectUpdateRevision++;
  motionController?.abort(new DOMException('Frame modifiée', 'AbortError'));
  motionController = null; motionPending = false;
  await stopPreview();
  resetPickerState();
  el.timestamp.value = Math.max(0, Math.min(Number(el.timestamp.max), Math.round(time * 30) / 30));
  updateTimeLabel();
  resetV2State(); effect?.dispose(); effect = null; effectTime = null; clearResult();
  for (const key of ['Masque du sujet', 'Complétion du fond', 'Construction des couches', 'Validation du trajet']) delete timings[key];
  el['preview-wrap'].hidden = true; el['source-wrap'].hidden = !media;
  status('Instant modifié. Prépare l’aperçu pour choisir le sujet.');
  controls();
}

el.file.addEventListener('change', () => { const file = el.file.files[0]; el.file.value = ''; loadFile(file); });
for (const name of ['dragenter', 'dragover']) el['drop-zone'].addEventListener(name, event => {
  event.preventDefault(); el['drop-zone'].classList.add('dragging');
});
for (const name of ['dragleave', 'drop']) el['drop-zone'].addEventListener(name, () => el['drop-zone'].classList.remove('dragging'));
document.addEventListener('dragover', event => event.preventDefault());
document.addEventListener('drop', event => { event.preventDefault(); loadFile(event.dataTransfer.files[0]); });
el.timestamp.addEventListener('input', async () => {
  const time = Number(el.timestamp.value);
  await changeSelection(time);
  if (media) seekSelection();
});
el.source.addEventListener('seeked', () => {
  if (!busy && media && Math.abs(el.source.currentTime - media.start - Number(el.timestamp.value)) > .04) changeSelection(el.source.currentTime - media.start);
});
el.intensity.addEventListener('input', async () => {
  el['intensity-label'].value = `${el.intensity.value} %`;
  clearResult();
  if (effect) await queueEffectChange(applyEffectSettings);
});
el.trajectory.addEventListener('change', async () => {
  trajectory = el.trajectory.value;
  clearResult();
  if (effect) await queueEffectChange(applyEffectSettings);
});
el.amplitude.addEventListener('input', async () => {
  updateAmplitudeLabel();
  clearResult();
  if (effect) await queueEffectChange(applyEffectSettings);
});
el['v2-enable'].addEventListener('click', enableAmplifiedMode);
el['v2-build'].addEventListener('click', buildAmplifiedLayers);
el['v2-abandon'].addEventListener('click', abandonAmplifiedMode);
el['v2-edit'].addEventListener('click', editAmplifiedMask);
el['v2-use'].addEventListener('click', switchToV2);
el['v2-v1'].addEventListener('click', switchToV1);
el['mask-size'].addEventListener('input', () => {
  el['mask-size-label'].value = `${el['mask-size'].value} %`;
});
el['mask-undo'].addEventListener('click', () => {
  if (!maskEditorOpen || busy || !maskStrokes.length) return;
  if (activeStroke) finishMaskStroke();
  maskStrokes.pop(); rebuildMask(); status('Dernier trait annulé.');
});
el['mask-reset'].addEventListener('click', () => {
  if (!maskEditorOpen || busy || !maskBaseline) return;
  activeStroke = null;
  maskStrokes = []; maskData = maskBaseline.slice(); redrawMaskOverlay(); maskStatus();
  status('Masque automatique réinitialisé.');
});
el['mask-overlay'].addEventListener('pointerdown', event => {
  if (!maskEditorOpen || busy || motionPending || !maskData || event.button > 0) return;
  const rect = canvasContentRect();
  if (!rect || event.clientX < rect.left || event.clientX > rect.left + rect.width ||
      event.clientY < rect.top || event.clientY > rect.top + rect.height) return;
  const point = maskPoint(event.clientX, event.clientY);
  if (!point) return;
  event.preventDefault();
  activeStroke = { add: el['mask-brush'].value === 'add', radius: brushRadius(), points: [point] };
  redrawMaskOverlay(applyStroke(maskData, activeStroke));
  el['mask-overlay'].setPointerCapture(event.pointerId);
});
el['mask-overlay'].addEventListener('pointermove', event => {
  if (!activeStroke || !el['mask-overlay'].hasPointerCapture(event.pointerId)) return;
  const point = maskPoint(event.clientX, event.clientY);
  if (!point) return;
  const previous = activeStroke.points[activeStroke.points.length - 1];
  if (Math.hypot(point.x - previous.x, point.y - previous.y) < .5) return;
  activeStroke.points.push(point);
  const dirty = applyStroke(maskData, { ...activeStroke, points: [previous, point] });
  redrawMaskOverlay(dirty);
});
function finishMaskStroke() {
  if (!activeStroke) return;
  maskStrokes.push(activeStroke); activeStroke = null;
  maskStatus();
  status('Trait ajouté au masque.');
}
el['mask-overlay'].addEventListener('pointerup', finishMaskStroke);
el['mask-overlay'].addEventListener('pointercancel', finishMaskStroke);
el['mask-overlay'].addEventListener('lostpointercapture', finishMaskStroke);
el['pivot-pick'].addEventListener('click', () => picking ? cancelPicking() : enterPicking());
el['pivot-reset'].addEventListener('click', fitPivotToCenter);
el['pivot-overlay'].addEventListener('click', event => {
  if (!picking || busy || pivotPending) return;
  const point = event.detail === 0 ? { u: .5, v: .5 } : contentPoint(event.clientX, event.clientY);
  if (point) void choosePivot(point);
});
document.addEventListener('keydown', event => {
  if (picking && event.key === 'Escape') {
    event.preventDefault();
    void cancelPicking();
  } else if (maskEditorOpen && event.key === 'Escape') {
    event.preventDefault();
    void abandonAmplifiedMode();
  } else if (maskEditorOpen && !busy && !motionPending && (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
    event.preventDefault();
    if (maskStrokes.length) { maskStrokes.pop(); rebuildMask(); status('Dernier trait annulé.'); }
  }
});
el.cancel.addEventListener('click', cancelWork);
el.replay.addEventListener('click', () => { if (!picking) playPreview(); });
el.suggest.addEventListener('click', () => runTask(async signal => {
  status('Recherche d’un instant actif et net…', 0);
  const time = await suggestMoment(media, signal, ({ progress }) => status('Recherche d’un instant actif et net…', progress * 100));
  await changeSelection(time); signal.throwIfAborted();
  seekSelection();
  status(`Instant suggéré : ${seconds(Number(el.timestamp.value))}. Tu peux l’ajuster.`);
}));

el.prepare.addEventListener('click', async () => {
  await runTask(async signal => {
    resetV2State();
    effect?.dispose(); effect = null; clearResult();
    for (const key of ['Masque du sujet', 'Complétion du fond', 'Construction des couches', 'Validation du trajet']) delete timings[key];
    status('Extraction de l’image choisie…', null);
    const time = Number(el.timestamp.value);
    const frame = await frameAt(media, time, signal);
    const prediction = await estimateDepth(frame, signal, data => {
      if (data.type === 'fallback') status('Passage au moteur CPU pour cet appareil…', null);
      else if (data.event?.status === 'progress') {
        const progress = data.event.progress;
        status(`Téléchargement du modèle… ${Number.isFinite(progress) ? Math.round(progress) + ' %' : ''}`, Number.isFinite(progress) ? progress : null);
      } else status('Chargement du modèle et estimation de la profondeur…', null);
    });
    // One image is inferred per effect: release the model before allocating the
    // splat renderer and encoder, particularly on devices with shared GPU memory.
    disposeDepth();
    timings['Chargement modèle'] = prediction.loadSeconds;
    timings['Inférence profondeur'] = prediction.inferenceSeconds;
    timings['Moteur'] = prediction.backend === 'webgpu' ? 'WebGPU' : 'WASM (CPU)';
    status('Construction des surfaces et préparation du mouvement…', null);
    effect = await createEffect(frame, prediction, signal);
    applyEffectSettings(effect);
    effect.mode = 'v1'; mode = 'v1';
    effectTime = time;
    timings['Construction splats'] = effect.seconds;
    if (Number.isFinite(effect.refinementSeconds)) timings['Affinage des contours'] = effect.refinementSeconds;
    else delete timings['Affinage des contours'];
    el.preview.width = media.width; el.preview.height = media.height;
    status('Aperçu prêt. Ajuste le mouvement ou génère la vidéo.');
  });
  if (effect && !busy) await playPreview();
});

el.generate.addEventListener('click', () => {
  if (picking || maskEditorOpen || motionPending) return;
  return runTask(async signal => {
    clearResult();
    if (effect?.mode === 'v2') {
      status('Vérification des 60 positions avant export…', 0);
      const validation = await validateCurrentMotion(effect, signal);
      if (!validation?.ok) {
        setV2Fallback(validation?.message);
        status('Export arrêté : le mouvement amplifié ne passe pas les vérifications. Tu peux exporter la version initiale ou ajuster les réglages.');
        return;
      }
      mode = 'v2'; effect.mode = 'v2';
    }
    status('Encodage de la vidéo muette…', 0);
    const result = await exportVideo(media, { time: effectTime, effect, signal,
      onProgress: ({ progress }) => status('Encodage de la vidéo muette…', progress * 100) });
    signal.throwIfAborted();
    timings['Export vidéo'] = result.seconds;
    resultURL = URL.createObjectURL(result.blob);
    el.result.src = resultURL;
    el.download.href = resultURL; el.download.download = `bullet-time${result.extension}`;
    el['result-wrap'].hidden = false;
    el['preview-wrap'].hidden = true;
    el['result-info'].textContent = `${seconds(result.duration)} · ${(result.blob.size / 1e6).toFixed(1)} Mo · sans son`;
    status('Terminé. Tu peux regarder et télécharger ta vidéo.');
  });
});

window.addEventListener('pagehide', () => {
  controller?.abort(); previewToken++; disposeDepth(); effect?.dispose(); media?.dispose();
  if (sourceURL) URL.revokeObjectURL(sourceURL);
  if (resultURL) URL.revokeObjectURL(resultURL);
});
controls();
updatePivotLabel();
window.addEventListener('resize', positionPivotMarker);
window.addEventListener('resize', maskOverlayGeometry);
updateAmplitudeLabel();
document.documentElement.dataset.ready = 'true';

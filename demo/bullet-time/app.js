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
  for (const id of ['suggest', 'prepare', 'intensity', 'trajectory']) el[id].disabled = busy || !media || picking || pivotPending;
  el.generate.disabled = busy || !effect || picking || pivotPending;
  el.replay.disabled = busy || !effect || picking || pivotPending;
  el['pivot-pick'].disabled = busy || !effect || pivotPending;
  el['pivot-pick'].textContent = picking ? 'Annuler le choix' : 'Choisir le sujet';
  el['pivot-reset'].disabled = busy || !effect || pivotPending;
  el['pivot-overlay'].hidden = !picking;
  el['pivot-overlay'].disabled = busy || !picking || pivotPending;
  el['pivot-instructions'].hidden = !picking;
  el.cancel.hidden = !busy;
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
  controller?.abort(new DOMException('Traitement annulé', 'AbortError'));
  if (task) await task;
  await stopPreview();
  disposeDepth();
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

function queueEffectChange(change, { allowPicking = false, afterChange } = {}) {
  clearResult();
  const targetEffect = effect;
  const revision = ++effectUpdateRevision;
  if (!targetEffect) return Promise.resolve(false);
  const operation = effectUpdates.then(async () => {
    if (revision !== effectUpdateRevision || busy || (!allowPicking && picking) || targetEffect !== effect) return false;
    await stopPreview();
    if (revision !== effectUpdateRevision || busy || (!allowPicking && picking) || targetEffect !== effect) return false;
    change(targetEffect);
    afterChange?.();
    controls();
    void playPreview();
    return true;
  });
  const handled = operation.catch(error => {
    if (revision === effectUpdateRevision) el.error.textContent = error.message || 'L’aperçu n’a pas pu être mis à jour.';
    return false;
  });
  effectUpdates = handled.then(() => undefined);
  return handled;
}

function applyEffectSettings(currentEffect) {
  currentEffect.intensity = Number(el.intensity.value) / 100;
  currentEffect.trajectory = trajectory;
  currentEffect.pivot = { ...pivot };
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
  clearResult(); effect?.dispose(); effect = null; effectTime = null;
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
  await stopPreview();
  resetPickerState();
  el.timestamp.value = Math.max(0, Math.min(Number(el.timestamp.max), Math.round(time * 30) / 30));
  updateTimeLabel();
  effect?.dispose(); effect = null; effectTime = null; clearResult();
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
    effect?.dispose(); effect = null; clearResult();
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
    effect.intensity = Number(el.intensity.value) / 100;
    effect.trajectory = trajectory;
    effect.pivot = { ...pivot };
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
  if (picking) return;
  return runTask(async signal => {
    clearResult();
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
document.documentElement.dataset.ready = 'true';

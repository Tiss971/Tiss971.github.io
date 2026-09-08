import * as GaussianSplats3D from 'gaussian-splat-renderer-for-lam';
import { FilesetResolver, FaceLandmarker } from '@mediapipe/tasks-vision';

const container = document.getElementById('avatar-container');
const loadingEl = document.getElementById('demo-loading');
const errorEl = document.getElementById('demo-error');
const panelEl = document.getElementById('demo-panel');
const hintEl = document.getElementById('demo-hint');
const webcamBtn = document.getElementById('webcam-btn');
const webcamStatus = document.getElementById('webcam-status');
const videoEl = document.getElementById('webcam-video');
const landmarkToggleRow = document.getElementById('landmark-toggle-row');
const landmarkToggle = document.getElementById('landmark-toggle');
const landmarkCanvas = document.getElementById('landmark-overlay');
const landmarkCtx = landmarkCanvas.getContext('2d');
const gsDebugCanvas = document.getElementById('gs-debug-overlay');
const gsDebugCtx = gsDebugCanvas.getContext('2d');
const emotionLabel = document.getElementById('emotion-label');
const perfLabel = document.getElementById('perf-label');
const emotionDebugLabel = document.getElementById('emotion-debug');
const modelSelect = document.getElementById('model-select');

const DEFAULT_ASSET = 'p2-1-example';

// ?asset=mathis or ?asset=p2-1-example picks a short name under assets/lam/<name>.zip;
// ?asset=../../assets/lam/custom.zip (or an http(s) URL) is used as-is for one-off testing.
const requestedAssetId = new URLSearchParams(location.search).get('asset');

function resolveAssetPath() {
  if (!requestedAssetId) return `../../assets/lam/${DEFAULT_ASSET}.zip`;
  if (requestedAssetId.includes('/') || requestedAssetId.startsWith('http')) return requestedAssetId;
  return `../../assets/lam/${requestedAssetId}.zip`;
}

const ASSET_PATH = resolveAssetPath();

// Reflect the active model in the picker, and switch via a full reload on
// change — simplest and safest way to swap the renderer/WebGL context.
modelSelect.value = requestedAssetId && !requestedAssetId.includes('/') ? requestedAssetId : DEFAULT_ASSET;
modelSelect.addEventListener('change', () => {
  const params = new URLSearchParams(location.search);
  params.set('asset', modelSelect.value);
  location.search = params.toString();
});
const EXPRESSION_PATH = '../../assets/lam/test_expression_1s.json';
const FRAME_INTERVAL = 1 / 30;
const FACE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';
const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';

// LAM's morph-target dict expects "mouthCheekPuff"; MediaPipe's ARKit blendshape
// set names the same category "cheekPuff" — without this alias the puff
// expression is silently dropped (LAM's renderer overwrites its whole default
// dict with whatever getExpressionData() returns, no per-key merge/fallback).
const NAME_ALIASES = { cheekPuff: 'mouthCheekPuff' };

// MediaPipe's blendshape categories are anatomical (the subject's own left/
// right, matching the true camera image). Applied as-is, the avatar blinking
// "its left eye" when the visitor blinks their own left eye reads as wrong —
// people expect a mirror (like a webcam self-view or an actual mirror), where
// the reflection's left is the viewer's right. Swap every laterality-paired
// blendshape so the avatar mirrors the visitor instead of copying them.
const MIRROR_PAIRS = [
  'browDown', 'browOuterUp', 'cheekSquint', 'eyeBlink', 'eyeLookDown',
  'eyeLookIn', 'eyeLookOut', 'eyeLookUp', 'eyeSquint', 'eyeWide', 'jaw',
  'mouthDimple', 'mouthFrown', 'mouth', 'mouthLowerDown', 'mouthPress',
  'mouthSmile', 'mouthStretch', 'mouthUpperUp', 'noseSneer',
];
const MIRROR_MAP = {};
MIRROR_PAIRS.forEach((base) => {
  MIRROR_MAP[`${base}Left`] = `${base}Right`;
  MIRROR_MAP[`${base}Right`] = `${base}Left`;
});

// LAM_WebRender applies expression weights raw, every frame, with zero
// interpolation (updateBS() just replaces its influence dict wholesale) — so
// any per-frame jitter from MediaPipe's landmark detection shows up directly
// on the avatar. Smooth it here instead.
const SMOOTHING = 0.45; // higher = snappier/more jittery, lower = smoother/more laggy
const MISS_FRAMES_BEFORE_WARN = 45; // ~1.5s at 30fps before flagging "no face"

// --- EXPERIMENTAL: live head rotation ---------------------------------
// getExpressionData()/getChatState() are the only two inputs the public
// GaussianSplatRenderer API consumes (verified by reading the bundle) — head
// rotation isn't exposed at all; the avatar's head motion normally comes from
// a baked animation.glb clip inside the asset, unrelated to any live input.
// This reaches into renderer-internal fields (viewer.skinModel.skeleton,
// found by bone name) to drive a "head" bone directly from MediaPipe's head
// pose. It is UNSUPPORTED, UNDOCUMENTED, untested with a real camera, and may
// silently no-op (bone not found), get fought every frame by the library's
// own AnimationMixer (which may also target the head bone), or not propagate
// to the gaussian splats at all if their bone texture isn't refreshed the way
// we assume. Wrapped defensively so a failure here can never break the
// (working) facial pilotage. Flip the *_SIGN constants if a turn goes the
// wrong way once tested live.
const HEAD_YAW_SIGN = -1; // mirrored, like the facial blendshapes
const HEAD_PITCH_SIGN = 1;
const HEAD_ROLL_SIGN = -1;
// Confirmed live (2026-09-08): all 3 rotation axes read correctly, but the
// whole bust rotates with the head, not just the head gaussians. Skeleton
// hierarchy is clean (logged: "head" sits under neckUpper/neckLower/
// chestUpper, shoulders are a sibling branch off chestUpper, not a
// descendant of "head") — so this isn't a parenting bug, it's skin-weight
// bleed from the one-shot rig (a generic Daz "Female003" skeleton fitted to
// LAM's single-photo geometry without fine weight painting). Can't repaint
// weights from here, so damp the applied rotation instead: trades head
// motion amplitude for less visible bust drag. 1 = full rotation (max
// bleed), 0 = no head motion at all.
const HEAD_ROTATION_SCALE = 0.5;
let headBone = null;
let headBoneSearched = false;
// Rotating a bone moves its *children's* world position, not its own origin
// (a bone's translation comes from its parent chain, untouched here) — so
// the head marker must track a bone that's a descendant of "head" (moves
// when head rotates), not "head" itself (would stay static, as first
// observed live 2026-09-08).
let headMarkerBone = null;
let headMarkerScratchVec = null; // lazily cloned from a bone's position (same bundled THREE realm)

function findHeadBone(renderer) {
  const bones = renderer?.viewer?.skinModel?.skeleton?.bones;
  if (!bones) return null;
  headMarkerBone = bones.find((b) => b.name === 'Nose')
    ?? bones.find((b) => b.name === 'lEye')
    ?? null;
  return bones.find((b) => b.name === 'head')
    ?? bones.find((b) => b.name?.toLowerCase().includes('head'))
    ?? null;
}

// MediaPipe's facialTransformationMatrixes[0].data is a column-major 4x4;
// convert the rotation submatrix to a quaternion (Shepperd's method).
function quaternionFromColumnMajor(d) {
  const m11 = d[0], m21 = d[1], m31 = d[2];
  const m12 = d[4], m22 = d[5], m32 = d[6];
  const m13 = d[8], m23 = d[9], m33 = d[10];
  const trace = m11 + m22 + m33;
  if (trace > 0) {
    const s = 0.5 / Math.sqrt(trace + 1);
    return { w: 0.25 / s, x: (m32 - m23) * s, y: (m13 - m31) * s, z: (m21 - m12) * s };
  }
  if (m11 > m22 && m11 > m33) {
    const s = 2 * Math.sqrt(1 + m11 - m22 - m33);
    return { w: (m32 - m23) / s, x: 0.25 * s, y: (m12 + m21) / s, z: (m13 + m31) / s };
  }
  if (m22 > m33) {
    const s = 2 * Math.sqrt(1 + m22 - m11 - m33);
    return { w: (m13 - m31) / s, x: (m12 + m21) / s, y: 0.25 * s, z: (m23 + m32) / s };
  }
  const s = 2 * Math.sqrt(1 + m33 - m11 - m22);
  return { w: (m21 - m12) / s, x: (m13 + m31) / s, y: (m23 + m32) / s, z: 0.25 * s };
}

// Normalized lerp from the identity quaternion — a cheap, good-enough way to
// scale a rotation's amplitude down without a full slerp implementation.
function dampQuaternion(x, y, z, w, t) {
  const nx = x * t, ny = y * t, nz = z * t, nw = 1 * (1 - t) + w * t;
  const len = Math.hypot(nx, ny, nz, nw) || 1;
  return { x: nx / len, y: ny / len, z: nz / len, w: nw / len };
}

// Shared by the head-rotation hack and the camera-follow experiment below,
// so both read the same sign-corrected orientation instead of duplicating
// the matrix/quaternion math.
function getSignedHeadQuaternion(result) {
  const mats = result.facialTransformationMatrixes;
  if (!mats || mats.length === 0) return null;
  const q = quaternionFromColumnMajor(mats[0].data);
  return {
    x: HEAD_PITCH_SIGN * q.x,
    y: HEAD_YAW_SIGN * q.y,
    z: HEAD_ROLL_SIGN * q.z,
    w: q.w,
  };
}

function applyHeadRotation(signedQ, renderer) {
  if (!headBoneSearched) {
    headBoneSearched = true;
    headBone = findHeadBone(renderer);
    if (!headBone) console.warn('[avatar-demo] no "head" bone found — live head rotation disabled');
  }
  if (!headBone || !signedQ) return;
  try {
    const damped = dampQuaternion(signedQ.x, signedQ.y, signedQ.z, signedQ.w, HEAD_ROTATION_SCALE);
    headBone.quaternion.set(damped.x, damped.y, damped.z, damped.w);
    headBone.updateMatrixWorld(true);
    renderer.viewer.skinModel.skeleton.update?.();
  } catch (err) {
    console.warn('[avatar-demo] live head rotation failed, disabling', err);
    headBone = null;
  }
}
// --- end experimental head rotation ------------------------------------

// --- Live emotion label (fun, not scientific) ---------------------------
// Calibrated thresholds over already-smoothed/mirrored ARKit weights — just
// enough to label the pilotage output, not a real affect-recognition model.
// Calibration zeroes out resting face/eyebrow posture (a per-person baseline
// captured over the first ~30 frames of each webcam session) so thresholds
// can be tighter without false-positiving on neutral faces.
class ARKitEmotionEstimator {
  constructor() {
    this.baseline = null;
    this.isCalibrating = false;
    this.calibrationSamples = [];
    this.sampleLimit = 30; // ~1s at this demo's ~30fps detect loop
  }

  startCalibration() {
    this.calibrationSamples = [];
    this.isCalibrating = true;
  }

  updateCalibration(weights) {
    if (!this.isCalibrating) return false;

    this.calibrationSamples.push(weights);

    if (this.calibrationSamples.length >= this.sampleLimit) {
      this.baseline = {};
      const keys = Object.keys(weights);
      for (const key of keys) {
        const sum = this.calibrationSamples.reduce((acc, frame) => acc + (frame[key] ?? 0), 0);
        this.baseline[key] = sum / this.calibrationSamples.length;
      }
      this.isCalibrating = false;
      return true; // calibration complete
    }

    return false; // still gathering frames
  }

  // Delta scores only, factored out of estimate() so the debug overlay can
  // show live values next to their thresholds (see #emotion-debug in the
  // detect loop) without duplicating this math.
  computeScores(weights) {
    const g = (name) => {
      const current = weights[name] ?? 0;
      const base = this.baseline?.[name] ?? 0;
      return Math.max(0, current - base);
    };

    const smileLeft = g('mouthSmileLeft');
    const smileRight = g('mouthSmileRight');
    // max() alongside the average so a half-smile/wink isn't averaged away.
    const smile = Math.max((smileLeft + smileRight) / 2, Math.max(smileLeft, smileRight) * 0.85);

    return {
      smile,
      frown: (g('mouthFrownLeft') + g('mouthFrownRight')) / 2,
      cheekSquint: (g('cheekSquintLeft') + g('cheekSquintRight')) / 2,
      browUp: (g('browOuterUpLeft') + g('browOuterUpRight') + g('browInnerUp')) / 3,
      browDown: (g('browDownLeft') + g('browDownRight')) / 2,
      browInnerUp: g('browInnerUp'),
      eyeWide: (g('eyeWideLeft') + g('eyeWideRight')) / 2,
      eyeSquint: (g('eyeSquintLeft') + g('eyeSquintRight')) / 2,
      jawOpen: g('jawOpen'),
      noseSneer: (g('noseSneerLeft') + g('noseSneerRight')) / 2,
      mouthUpperUp: (g('mouthUpperUpLeft') + g('mouthUpperUpRight')) / 2,
      mouthStretch: (g('mouthStretchLeft') + g('mouthStretchRight')) / 2,
      mouthPress: (g('mouthPressLeft') + g('mouthPressRight')) / 2,
    };
  }

  estimate(weights) {
    const { smile, browUp, browDown, browInnerUp, eyeSquint, jawOpen, mouthPress } =
      this.computeScores(weights);

    // --- Priority decision tree ---
    // Simplifié 2026-09-08 : mesuré en conditions réelles que cheekSquint,
    // eyeWide, noseSneer, mouthStretch et mouthFrown ne varient quasiment
    // jamais avec ce MediaPipe/cette webcam (signal mort, pas un problème de
    // seuil) — dégoût et peur en dépendaient entièrement, retirés. joie
    // (duchenne) dépendait de cheekSquint seul, fusionnée dans sourire.
    // 1. Rire / joie extrême.
    if (jawOpen > 0.35 && smile > 0.25) return 'rire';
    // 2. Sourire (fusionne l'ancienne distinction "duchenne", invérifiable
    //    sans cheekSquint).
    if (smile > 0.28) return 'sourire';
    // 3. Surprise (sourcils entiers levés — AU1+AU2 moyennés).
    if (browUp > 0.25) return 'surprise';
    // 4. Tristesse (sourcils *internes* seuls levés, sans que l'ensemble du
    //    sourcil suive — sinon ça aurait déjà matché "surprise" au-dessus ;
    //    approximation d'AU1 seul, chevauchement résiduel assumé avec
    //    "surprise" puisque browUp inclut browInnerUp dans sa moyenne).
    if (browInnerUp > 0.15) return 'tristesse';
    // 5. Colère / frustration (AU4 sourcils froncés + AU7 yeux plissés ou lèvres pressées).
    if (browDown > 0.25 && (eyeSquint > 0.2 || mouthPress > 0.2)) return 'colère';
    // 6. Froncement léger / concentration.
    if (browDown > 0.25) return 'froncement';
    // 7. Bouche ouverte / parole.
    if (jawOpen > 0.3) return 'bouche ouverte';
    return 'neutre';
  }
}

const emotionEstimator = new ARKitEmotionEstimator();
// --- end emotion label ---------------------------------------------------

// --- Pilotage latency / FPS counter --------------------------------------
let perfFrameCount = 0;
let perfWindowStart = 0;
let perfLastFPS = 0;
let perfLastLatencyMs = 0;

function recordPerfFrame(latencyMs) {
  perfLastLatencyMs = latencyMs;
  perfFrameCount++;
  const now = performance.now();
  if (!perfWindowStart) perfWindowStart = now;
  const elapsed = now - perfWindowStart;
  if (elapsed >= 1000) {
    perfLastFPS = Math.round((perfFrameCount * 1000) / elapsed);
    perfFrameCount = 0;
    perfWindowStart = now;
  }
}
// --- end perf counter -----------------------------------------------------

let expressionData = null;
let startTime = 0;
let faceLandmarker = null;
let smoothedBlendshapes = null;
let webcamActive = false;
let missedFrames = 0;
let rendererInstance = null;

function getChatState() {
  return 'Idle';
}

function getExpressionData() {
  if (webcamActive && smoothedBlendshapes) return smoothedBlendshapes;
  if (!expressionData) return {};
  const { names, frames } = expressionData;
  const loopDuration = frames.length * FRAME_INTERVAL;
  const elapsed = (performance.now() / 1000 - startTime) % loopDuration;
  const frameIndex = Math.floor(elapsed / FRAME_INTERVAL);
  const weights = frames[frameIndex].weights;
  const out = {};
  names.forEach((name, i) => { out[name] = weights[i]; });
  return out;
}

function smoothInto(target, value) {
  const prev = smoothedBlendshapes[target] ?? 0;
  smoothedBlendshapes[target] = prev + (value - prev) * SMOOTHING;
}

function applySmoothedFrame(categories) {
  if (!smoothedBlendshapes) smoothedBlendshapes = {};
  categories.forEach(({ categoryName, score }) => {
    const mirrored = MIRROR_MAP[categoryName] ?? categoryName;
    smoothInto(mirrored, score);
    const alias = NAME_ALIASES[categoryName];
    if (alias) smoothInto(alias, score);
  });
}

// Debug aid to make the pilotage layer itself visible, instead of only its
// end effect on the avatar: draws the raw video frame plus MediaPipe's 468
// face landmarks as detected, before any smoothing/mirroring is applied.
function drawLandmarkOverlay(result) {
  if (landmarkCanvas.width !== videoEl.videoWidth) {
    landmarkCanvas.width = videoEl.videoWidth;
    landmarkCanvas.height = videoEl.videoHeight;
  }
  landmarkCtx.drawImage(videoEl, 0, 0);
  const landmarks = result.faceLandmarks?.[0];
  if (!landmarks) return;
  landmarkCtx.fillStyle = '#7dd3fc';
  for (const { x, y } of landmarks) {
    landmarkCtx.fillRect(x * landmarkCanvas.width - 1, y * landmarkCanvas.height - 1, 2, 2);
  }
}

// Marks where the pilotage layer thinks the head bone is, directly on the
// gaussian splat render — not just on the video feed. `viewer` exposes its
// live Three.js `camera` and `canvas` (confirmed by logging its surface),
// so the bone's world position can be projected to screen space with the
// same Vector3.project() used internally by the bundle, without importing a
// separate Three.js instance.
function drawHeadMarker(renderer) {
  const camera = renderer?.viewer?.camera;
  const canvas3d = renderer?.viewer?.canvas;
  const bone = headMarkerBone ?? headBone; // fall back to the static origin if no child bone was found
  if (!bone || !camera || !canvas3d) return;
  if (!headMarkerScratchVec) headMarkerScratchVec = bone.position.clone();
  bone.getWorldPosition(headMarkerScratchVec);
  headMarkerScratchVec.project(camera);

  const w = canvas3d.clientWidth, h = canvas3d.clientHeight;
  if (gsDebugCanvas.width !== w) gsDebugCanvas.width = w;
  if (gsDebugCanvas.height !== h) gsDebugCanvas.height = h;
  gsDebugCtx.clearRect(0, 0, w, h);
  if (headMarkerScratchVec.z > 1) return; // behind camera

  const x = (headMarkerScratchVec.x * 0.5 + 0.5) * w;
  const y = (-headMarkerScratchVec.y * 0.5 + 0.5) * h;
  gsDebugCtx.strokeStyle = '#7dd3fc';
  gsDebugCtx.lineWidth = 2;
  gsDebugCtx.beginPath();
  gsDebugCtx.arc(x, y, 8, 0, Math.PI * 2);
  gsDebugCtx.moveTo(x - 12, y);
  gsDebugCtx.lineTo(x + 12, y);
  gsDebugCtx.moveTo(x, y - 12);
  gsDebugCtx.lineTo(x, y + 12);
  gsDebugCtx.stroke();
}

function webcamDetectLoop() {
  if (!webcamActive) return;
  if (videoEl.videoWidth) {
    const detectStart = performance.now();
    const result = faceLandmarker.detectForVideo(videoEl, performance.now());
    if (landmarkToggle.checked) recordPerfFrame(performance.now() - detectStart);

    const categories = result.faceBlendshapes?.[0]?.categories;
    const signedQ = getSignedHeadQuaternion(result);
    if (categories) {
      missedFrames = 0;
      applySmoothedFrame(categories);
      if (rendererInstance) applyHeadRotation(signedQ, rendererInstance);
      webcamStatus.textContent = 'Pilotage webcam actif — visage détecté.';
      if (emotionEstimator.isCalibrating) {
        emotionEstimator.updateCalibration(smoothedBlendshapes);
        emotionLabel.textContent = 'Émotion : calibration… reste neutre';
      } else {
        emotionLabel.textContent = `Émotion : ${emotionEstimator.estimate(smoothedBlendshapes)}`;
      }
    } else {
      missedFrames++;
      if (missedFrames === MISS_FRAMES_BEFORE_WARN) {
        webcamStatus.textContent = 'Visage non détecté — replace-toi face à la caméra.';
      }
    }
    if (landmarkToggle.checked) {
      drawLandmarkOverlay(result);
      if (rendererInstance) drawHeadMarker(rendererInstance);
      perfLabel.textContent = `FPS pilotage : ${perfLastFPS} · latence détection : ${perfLastLatencyMs.toFixed(1)}ms`;
      if (smoothedBlendshapes) {
        const s = emotionEstimator.computeScores(smoothedBlendshapes);
        const v = (x) => x.toFixed(2);
        emotionDebugLabel.textContent =
          `sourire:${v(s.smile)} \njoue(0.15):${v(s.cheekSquint)} \nbouche(0.30):${v(s.jawOpen)}\n` +
          `sourcils↑(0.25):${v(s.browUp)} \nsourcilsIn(0.20/0.15):${v(s.browInnerUp)} \nyeux+(0.25/0.20):${v(s.eyeWide)}\n` +
          `nez(0.25):${v(s.noseSneer)} \nlèvreH(0.20):${v(s.mouthUpperUp)} \nétir.(0.20):${v(s.mouthStretch)}\n` +
          `frown(0.20):${v(s.frown)} \nsourcils↓(0.25):${v(s.browDown)} \nyeux~(0.20):${v(s.eyeSquint)} \npressée(0.20):${v(s.mouthPress)}`;
      }
    }
  }
  requestAnimationFrame(webcamDetectLoop);
}

async function startWebcam() {
  webcamBtn.disabled = true;
  webcamStatus.textContent = 'Activation de la webcam…';
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 480, height: 360 } });
    videoEl.srcObject = stream;
    await videoEl.play();

    if (!faceLandmarker) {
      webcamStatus.textContent = 'Chargement du modèle de suivi facial…';
      const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
      faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: FACE_MODEL_URL },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
    }

    missedFrames = 0;
    webcamActive = true;
    emotionEstimator.startCalibration();
    webcamStatus.textContent = 'Pilotage webcam actif — tout reste local.';
    webcamBtn.textContent = 'Désactiver le pilotage webcam';
    webcamBtn.disabled = false;
    landmarkToggleRow.classList.remove('hidden');
    emotionLabel.classList.remove('hidden');
    webcamDetectLoop();
  } catch (err) {
    console.error(err);
    webcamStatus.textContent = 'Webcam indisponible ou permission refusée.';
    webcamBtn.disabled = false;
  }
}

function stopWebcam() {
  webcamActive = false;
  smoothedBlendshapes = null;
  const stream = videoEl.srcObject;
  stream?.getTracks().forEach((track) => track.stop());
  videoEl.srcObject = null;
  webcamStatus.textContent = '';
  webcamBtn.textContent = 'Activer le pilotage webcam';
  landmarkToggleRow.classList.add('hidden');
  landmarkToggle.checked = false;
  landmarkCanvas.classList.add('hidden');
  gsDebugCanvas.classList.add('hidden');
  gsDebugCtx.clearRect(0, 0, gsDebugCanvas.width, gsDebugCanvas.height);

  emotionLabel.classList.add('hidden');
  perfLabel.classList.add('hidden');
  emotionDebugLabel.classList.add('hidden');
  perfFrameCount = 0;
  perfWindowStart = 0;
}

webcamBtn.addEventListener('click', () => {
  if (webcamActive) stopWebcam();
  else startWebcam();
});

landmarkToggle.addEventListener('change', () => {
  landmarkCanvas.classList.toggle('hidden', !landmarkToggle.checked);
  gsDebugCanvas.classList.toggle('hidden', !landmarkToggle.checked);
  perfLabel.classList.toggle('hidden', !landmarkToggle.checked);
  emotionDebugLabel.classList.toggle('hidden', !landmarkToggle.checked);
  perfFrameCount = 0;
  perfWindowStart = 0;
});

async function start() {
  const res = await fetch(EXPRESSION_PATH);
  expressionData = await res.json();
  startTime = performance.now() / 1000;

  rendererInstance = await GaussianSplats3D.GaussianSplatRenderer.getInstance(container, ASSET_PATH, {
    getChatState,
    getExpressionData,
    // '#'-prefixed strings pass the lib's own isHexColorStrict() validation but
    // then choke on its parseInt(value, 16) call ("#0c0c0e" -> NaN, since '#'
    // isn't a hex digit) — a real bug in the library, confirmed by reading its
    // source. Using the "0x" prefix instead satisfies both the validator and
    // parseInt. Silently fell back to a default that happened to look right,
    // so this was invisible until the console was actually read (2026-09-08).
    backgroundColor: '0x0c0c0e',
    alpha: 1,
  });

  loadingEl.classList.add('hidden');
  panelEl.classList.remove('hidden');
  hintEl.classList.remove('hidden');
  window.__renderer = rendererInstance; // DEBUG
}

start().catch((err) => {
  console.error(err);
  errorEl.textContent = "Le chargement de l'avatar a échoué. Vérifie ta connexion et réessaie.";
});

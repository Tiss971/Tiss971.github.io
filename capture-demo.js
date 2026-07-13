import * as THREE from "three";
import {
  FilesetResolver,
  ImageSegmenter,
  PoseLandmarker,
} from "@mediapipe/tasks-vision";

// =====================
// CONFIG
// =====================
const CAPTURE_DURATION_MS = 12000;
const SAMPLE_INTERVAL_MS = 350;
const FOREGROUND_THRESHOLD = 0.5;
const PIXEL_STRIDE = 3; // subsample mask grid to control point count
const CARD_WIDTH = 0.9;
const CARD_HEIGHT = 1.3;

const SELFIE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_BASE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

// =====================
// DOM
// =====================
const stageEl = document.getElementById("demo-stage");
const instructionsEl = document.getElementById("demo-instructions");
const videoEl = document.getElementById("demo-video");
const maskCanvas = document.getElementById("demo-mask-canvas");
const maskCtx = maskCanvas.getContext("2d");
const progressArc = document.getElementById("demo-progress-arc");
const startBtn = document.getElementById("demo-start-btn");
const errorEl = document.getElementById("demo-error");

const container = document.getElementById("demo-container");
const panelEl = document.getElementById("demo-panel");
const hintEl = document.getElementById("demo-hint");
const restartBtn = document.getElementById("demo-restart");
const statFps = document.getElementById("stat-fps");
const statCount = document.getElementById("stat-count");
const statFrames = document.getElementById("stat-frames");

const ARC_CIRCUMFERENCE = 2 * Math.PI * 46;

// =====================
// STATE
// =====================
let stream = null;
let segmenter = null;
let poseLandmarker = null;
let phase = "idle"; // idle | loading | ready | capturing | reconstructing | result
let samples = [];
let offscreenCanvas = null;
let offscreenCtx = null;

function setError(msg) {
  errorEl.textContent = msg;
}

function setInstructions(text) {
  instructionsEl.textContent = text;
}

// =====================
// WEBCAM + MODELS INIT
// =====================
async function activateWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setError("Ce navigateur ne supporte pas l'accès à la webcam (getUserMedia indisponible).");
    return;
  }

  startBtn.disabled = true;
  startBtn.textContent = "Chargement…";
  setError("");

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();
  } catch (err) {
    setError("Permission webcam refusée ou indisponible : " + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "Activer la webcam";
    return;
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    segmenter = await ImageSegmenter.createFromOptions(vision, {
      baseOptions: { modelAssetPath: SELFIE_MODEL_URL },
      outputCategoryMask: false,
      outputConfidenceMasks: true,
      runningMode: "VIDEO",
    });
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL_URL },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  } catch (err) {
    setError("Échec du chargement des modèles MediaPipe : " + err.message);
    startBtn.disabled = false;
    startBtn.textContent = "Activer la webcam";
    return;
  }

  maskCanvas.width = 256;
  maskCanvas.height = 256;
  offscreenCanvas = document.createElement("canvas");
  offscreenCtx = offscreenCanvas.getContext("2d", { willReadFrequently: true });

  phase = "ready";
  startBtn.disabled = false;
  startBtn.textContent = "Démarrer la capture (12s)";
  setInstructions("Place-toi face à la webcam, puis tourne lentement sur toi-même pendant que la barre se remplit.");
  livePreviewLoop();
}

// =====================
// LIVE PREVIEW (mask overlay before/while capturing)
// =====================
function livePreviewLoop() {
  if (phase !== "ready" && phase !== "capturing") return;
  runSegmentationFrame(false);
  requestAnimationFrame(livePreviewLoop);
}

function runSegmentationFrame(record) {
  if (!videoEl.videoWidth) return null;
  const ts = performance.now();
  const segResult = segmenter.segmentForVideo(videoEl, ts);
  const confMask = segResult.confidenceMasks[0];
  const maskW = confMask.width;
  const maskH = confMask.height;
  const maskData = confMask.getAsFloat32Array();

  // live overlay: paint foreground green-ish
  maskCanvas.width = maskW;
  maskCanvas.height = maskH;
  const imgData = maskCtx.createImageData(maskW, maskH);
  for (let i = 0; i < maskData.length; i++) {
    const fg = maskData[i] > FOREGROUND_THRESHOLD;
    imgData.data[i * 4 + 0] = fg ? 200 : 0;
    imgData.data[i * 4 + 1] = fg ? 240 : 0;
    imgData.data[i * 4 + 2] = fg ? 77 : 0;
    imgData.data[i * 4 + 3] = fg ? 90 : 0;
  }
  maskCtx.putImageData(imgData, 0, 0);

  segResult.close?.();

  if (!record) return null;

  const poseResult = poseLandmarker.detectForVideo(videoEl, ts);
  const personDetected = poseResult.landmarks && poseResult.landmarks.length > 0;

  offscreenCanvas.width = maskW;
  offscreenCanvas.height = maskH;
  offscreenCtx.drawImage(videoEl, 0, 0, maskW, maskH);
  const colorData = offscreenCtx.getImageData(0, 0, maskW, maskH).data;

  return { maskData, colorData, maskW, maskH, personDetected };
}

// =====================
// TIMED CAPTURE
// =====================
function startCapture() {
  phase = "capturing";
  samples = [];
  startBtn.disabled = true;
  setInstructions("Capture en cours — continue de tourner lentement sur toi-même.");

  const startTime = performance.now();
  const sampleTimer = setInterval(() => {
    const elapsed = performance.now() - startTime;
    const t = Math.min(elapsed / CAPTURE_DURATION_MS, 1);
    progressArc.style.strokeDashoffset = String(ARC_CIRCUMFERENCE * (1 - t));

    // Angle from elapsed time (linear turntable assumption). A monocular fixed
    // camera can't reliably recover true rotation angle from body landmarks
    // alone (depth/orientation ambiguity), so PoseLandmarker is used only to
    // gate frames where no person is detected, not to derive the angle.
    const angle = t * Math.PI * 2;
    const frame = runSegmentationFrame(true);
    if (frame && frame.personDetected) {
      samples.push({ ...frame, angle });
      statFrames.textContent = String(samples.length);
    }

    if (elapsed >= CAPTURE_DURATION_MS) {
      clearInterval(sampleTimer);
      finishCapture();
    }
  }, SAMPLE_INTERVAL_MS);
}

function finishCapture() {
  phase = "reconstructing";
  setInstructions("Reconstruction du nuage de points…");
  stopWebcam();

  requestAnimationFrame(() => {
    buildPointCloud();
    showResult();
  });
}

function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

// =====================
// HEURISTIC RECONSTRUCTION
// =====================
function buildPointCloud() {
  const positions = [];
  const colors = [];

  for (const sample of samples) {
    const { maskData, colorData, maskW, maskH, angle } = sample;
    const sinA = Math.sin(angle);
    const cosA = Math.cos(angle);

    for (let py = 0; py < maskH; py += PIXEL_STRIDE) {
      for (let px = 0; px < maskW; px += PIXEL_STRIDE) {
        const idx = py * maskW + px;
        if (maskData[idx] <= FOREGROUND_THRESHOLD) continue;

        const u = px / maskW - 0.5; // -0.5..0.5 across the frame
        const v = py / maskH; // 0..1 top..bottom

        // Each captured frame is a flat "billboard" of the segmented subject,
        // centered on the origin and rotated around Y by the capture angle.
        // All frames overlap at the center → a turntable-style fan that reads
        // as one centered figure, instead of a ring of copies laid out on a
        // cylinder surface.
        const worldX = u * cosA * CARD_WIDTH;
        const worldZ = -u * sinA * CARD_WIDTH;
        const worldY = (0.5 - v) * CARD_HEIGHT;

        positions.push(worldX, worldY, worldZ);

        const ci = idx * 4;
        colors.push(colorData[ci] / 255, colorData[ci + 1] / 255, colorData[ci + 2] / 255);
      }
    }
  }

  statCount.textContent = (positions.length / 3).toLocaleString("fr-FR");
  initSceneAndRender(positions, colors);
}

// =====================
// THREE.JS RENDER
// =====================
let renderer, scene, camera, points;
let yaw = 0, pitch = 0.1, radius = 3.2;
let isDragging = false, lastX = 0, lastY = 0;

function makeSpriteTexture() {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = size;
  c.height = size;
  const ctx = c.getContext("2d");
  const grad = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.6, "rgba(255,255,255,0.6)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}

function initSceneAndRender(positions, colors) {
  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.045,
    map: makeSpriteTexture(),
    vertexColors: true,
    transparent: true,
    alphaTest: 0.05,
    depthWrite: false,
    sizeAttenuation: true,
  });

  points = new THREE.Points(geometry, material);
  scene.add(points);

  renderer.setAnimationLoop(renderLoop);
}

let frameCount = 0;
let lastFpsSample = performance.now();

function renderLoop() {
  camera.position.x = Math.cos(pitch) * Math.sin(yaw) * radius;
  camera.position.y = Math.sin(pitch) * radius;
  camera.position.z = Math.cos(pitch) * Math.cos(yaw) * radius;
  camera.lookAt(0, 0, 0);
  renderer.render(scene, camera);

  frameCount++;
  const now = performance.now();
  if (now - lastFpsSample >= 500) {
    statFps.textContent = Math.round((frameCount * 1000) / (now - lastFpsSample));
    frameCount = 0;
    lastFpsSample = now;
  }
}

function attachOrbitControls() {
  container.addEventListener("mousedown", (e) => { isDragging = true; lastX = e.clientX; lastY = e.clientY; });
  container.addEventListener("touchstart", (e) => {
    isDragging = true; lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  }, { passive: true });

  window.addEventListener("mouseup", () => { isDragging = false; });
  window.addEventListener("touchend", () => { isDragging = false; });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    yaw -= (e.clientX - lastX) * 0.006;
    pitch -= (e.clientY - lastY) * 0.006;
    pitch = Math.max(-Math.PI / 2.1, Math.min(Math.PI / 2.1, pitch));
    lastX = e.clientX; lastY = e.clientY;
  });
  window.addEventListener("touchmove", (e) => {
    if (!isDragging) return;
    yaw -= (e.touches[0].clientX - lastX) * 0.006;
    pitch -= (e.touches[0].clientY - lastY) * 0.006;
    pitch = Math.max(-Math.PI / 2.1, Math.min(Math.PI / 2.1, pitch));
    lastX = e.touches[0].clientX; lastY = e.touches[0].clientY;
  }, { passive: true });

  container.addEventListener("wheel", (e) => {
    e.preventDefault();
    radius += e.deltaY * 0.002;
    radius = Math.max(0.8, Math.min(10, radius));
  }, { passive: false });
}

function onResize() {
  if (!renderer) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function showResult() {
  phase = "result";
  stageEl.classList.add("hidden");
  container.classList.remove("hidden");
  panelEl.classList.remove("hidden");
  hintEl.classList.remove("hidden");
}

// =====================
// RESTART
// =====================
function reset() {
  stopWebcam();
  if (renderer) {
    renderer.setAnimationLoop(null);
    renderer.dispose();
    container.removeChild(renderer.domElement);
    renderer = null;
  }
  points = null;
  samples = [];
  yaw = 0; pitch = 0.1; radius = 3.2;

  container.classList.add("hidden");
  panelEl.classList.add("hidden");
  hintEl.classList.add("hidden");
  stageEl.classList.remove("hidden");
  progressArc.style.strokeDashoffset = String(ARC_CIRCUMFERENCE);
  statFrames.textContent = "—";
  statCount.textContent = "—";
  statFps.textContent = "—";
  setInstructions(
    "Ta webcam capture ~12 secondes pendant que tu tournes sur toi-même. Un modèle de segmentation isole " +
    "ton silhouette du fond en temps réel, et une reconstruction 3D approximative est générée — tout se " +
    "passe dans ton navigateur, rien n'est envoyé sur un serveur."
  );
  startBtn.textContent = "Activer la webcam";
  startBtn.disabled = false;
  phase = "idle";
}

// =====================
// EVENTS
// =====================
startBtn.addEventListener("click", () => {
  if (phase === "idle") activateWebcam();
  else if (phase === "ready") startCapture();
});

restartBtn.addEventListener("click", reset);

// Orbit controls + resize are attached once — the container/camera persist
// across capture sessions, so re-attaching on every restart would stack
// duplicate listeners and make dragging increasingly oversensitive.
attachOrbitControls();
window.addEventListener("resize", onResize);

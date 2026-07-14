import * as THREE from "three";
import {
  FilesetResolver,
  ImageSegmenter,
  PoseLandmarker,
  FaceLandmarker,
} from "@mediapipe/tasks-vision";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

// =====================
// CONFIG
// =====================
const CAPTURE_DURATION_MS = 8000;
const SAMPLE_INTERVAL_MS = 500;
const CAPTURE_SECONDS = Math.round(CAPTURE_DURATION_MS / 1000); // for UI copy
const FOREGROUND_THRESHOLD = 0.5;
// Silhouette → world scale (also the visual-hull bounding box: half-extent = ×0.5).
const CARD_WIDTH = 0.9;
const CARD_HEIGHT = 1.3;

// Visual hull (shape-from-silhouette): a voxel grid is carved by intersecting
// the segmented silhouettes projected from each frame's estimated angle.
const VOXEL_NX = 64;
const VOXEL_NY = 96;
const VOXEL_NZ = 64;
const CARVE_KEEP_RATIO = 0.7; // keep a voxel inside the silhouette in ≥70% of the views it projects into
                              // (tolerant of imperfect monocular poses; raise toward 0.9 for a tighter hull)
const MIN_CARVE_VIEWS = 3;    // require at least this many in-bounds views to trust a voxel
const MIN_VOXEL_NEIGHBORS = 3; // drop carved voxels with fewer occupied 26-neighbors (kills floating specks)

// Rendered as real gaussians (SparkJS): one isotropic splat per surviving voxel.
const SPLAT_SCALE = 0.010;   // gaussian radius in world units (~ half the voxel pitch 0.9/64 ≈ 0.014)
const SPLAT_OPACITY = 1.0;

// Per-frame rotation angle is measured from real head pose (MediaPipe
// FaceLandmarker yaw). When no face is visible we fall back to integrating the
// signed horizontal optical flow between frames. Nothing assumes a full 360°
// turn — the subject can move freely (e.g. show one cheek, then the other).
const FLOW_MAX_LAG_RATIO = 0.25; // horizontal-shift search window (× frame width)
const FLOW_RAD_PER_PX = 0.012;   // optical-flow px → radians (fallback only)
const FACE_YAW_SIGN = 1;         // flip to -1 if left/right ends up mirrored

const SELFIE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
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
const framesBtn = document.getElementById("demo-frames-btn");
const exportBtn = document.getElementById("demo-export-btn");
const filmstripEl = document.getElementById("demo-filmstrip");
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
let faceLandmarker = null;
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

// Single source of truth for the idle/reset instruction copy — the capture
// duration comes from CAPTURE_DURATION_MS, not a hardcoded number.
function defaultInstructions() {
  return `Ta webcam capture ~${CAPTURE_SECONDS} secondes pendant que tu tournes sur toi-même. ` +
    "Un modèle de segmentation isole ton silhouette du fond en temps réel, et une reconstruction 3D " +
    "approximative est générée — tout se passe dans ton navigateur, rien n'est envoyé sur un serveur.";
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
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: FACE_MODEL_URL },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true, // head pose (yaw)
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
  startBtn.textContent = `Démarrer la capture (${CAPTURE_SECONDS}s)`;
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
    imgData.data[i * 4 + 0] = fg ? 76 : 0;
    imgData.data[i * 4 + 1] = fg ? 194 : 0;
    imgData.data[i * 4 + 2] = fg ? 255 : 0;
    imgData.data[i * 4 + 3] = fg ? 90 : 0;
  }
  maskCtx.putImageData(imgData, 0, 0);

  segResult.close?.();

  if (!record) return null;

  const poseResult = poseLandmarker.detectForVideo(videoEl, ts);
  const personDetected = poseResult.landmarks && poseResult.landmarks.length > 0;

  // Real head pose (yaw) for this frame, or null if no face is visible.
  const faceResult = faceLandmarker.detectForVideo(videoEl, ts);
  const faceYaw = yawFromFaceResult(faceResult);

  offscreenCanvas.width = maskW;
  offscreenCanvas.height = maskH;
  offscreenCtx.drawImage(videoEl, 0, 0, maskW, maskH);
  const colorData = offscreenCtx.getImageData(0, 0, maskW, maskH).data;

  // Keep a full-resolution frame (raw, non-mirrored) for the COLMAP export.
  const fw = videoEl.videoWidth, fh = videoEl.videoHeight;
  const frameCanvas = document.createElement("canvas");
  frameCanvas.width = fw;
  frameCanvas.height = fh;
  frameCanvas.getContext("2d").drawImage(videoEl, 0, 0, fw, fh);

  return { maskData, colorData, maskW, maskH, personDetected, faceYaw, frameCanvas };
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

    // PoseLandmarker only gates frames where a subject is present. The rotation
    // angle is NOT derived here — it's reconstructed afterward from the measured
    // head pose (with optical-flow fallback), so no constant-speed / full-turn
    // assumption is baked into the capture.
    const frame = runSegmentationFrame(true);
    if (frame && frame.personDetected) {
      samples.push(frame);
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
  stopWebcam();
  setInstructions("Reconstruction du nuage de points…");

  requestAnimationFrame(() => {
    samples.forEach(keepLargestForeground); // drop stray mask islands (non-human)
    estimateFrameAngles();
    buildPointCloud();
    showResult();
  });
}

// Segmentation sometimes latches onto background patches. Keep only the largest
// connected foreground component (the subject) per frame; zero out the rest, so
// stray islands feed neither the flow estimate, the hull, nor the filmstrip.
function keepLargestForeground(s) {
  const { maskData, maskW, maskH } = s;
  const N = maskW * maskH;
  const label = new Int32Array(N); // 0 = unvisited/background
  const stack = [];
  let cur = 0, bestLabel = 0, bestSize = 0;

  for (let start = 0; start < N; start++) {
    if (label[start] !== 0 || maskData[start] <= FOREGROUND_THRESHOLD) continue;
    cur++;
    let size = 0;
    stack.push(start);
    label[start] = cur;
    while (stack.length) {
      const idx = stack.pop();
      size++;
      const x = idx % maskW, y = (idx / maskW) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= maskH) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= maskW || (dx === 0 && dy === 0)) continue;
          const nidx = ny * maskW + nx;
          if (label[nidx] === 0 && maskData[nidx] > FOREGROUND_THRESHOLD) {
            label[nidx] = cur;
            stack.push(nidx);
          }
        }
      }
    }
    if (size > bestSize) { bestSize = size; bestLabel = cur; }
  }

  for (let i = 0; i < N; i++) {
    if (label[i] !== bestLabel) maskData[i] = 0;
  }
}

// =====================
// POSE ESTIMATION (head-pose yaw + optical-flow fallback)
// =====================
// Assign each frame a rotation angle. Frames with a detected face use the
// measured head yaw directly (absolute); frames without a face are filled by
// integrating the signed horizontal optical flow from the previous frame. This
// makes no assumption of a full turn — free movement and back-and-forth (e.g.
// showing one cheek then the other) are handled naturally.
function estimateFrameAngles() {
  const n = samples.length;
  if (n === 0) return;

  const profiles = samples.map(columnProfile);

  // Signed horizontal optical flow (px) between consecutive frames.
  const shift = new Float32Array(n);
  for (let k = 1; k < n; k++) shift[k] = bestHorizontalShift(profiles[k - 1], profiles[k]);

  // Calibrate the px→radians scale against the head-pose anchors: over each
  // segment between two face-detected frames the true Δyaw is known, so we scale
  // the summed flow to match it. Fixes the flow's unknown/under-estimated scale
  // (a fixed constant badly under-rotates). Falls back to FLOW_RAD_PER_PX when
  // there aren't ≥2 usable anchors.
  const anchors = [];
  for (let k = 0; k < n; k++) if (samples[k].faceYaw != null) anchors.push(k);

  let radPerPx = FLOW_RAD_PER_PX;
  let calibrated = false;
  if (anchors.length >= 2) {
    let trueSum = 0, pxSum = 0;
    for (let a = 0; a < anchors.length - 1; a++) {
      const i = anchors[a], j = anchors[a + 1];
      trueSum += samples[j].faceYaw - samples[i].faceYaw;
      for (let k = i + 1; k <= j; k++) pxSum += shift[k];
    }
    // Need enough total flow to trust the fit, and keep the scale plausible
    // (guards against noisy near-static anchor segments blowing up the ratio).
    if (Math.abs(pxSum) > 5) {
      const r = trueSum / pxSum; // sign auto-corrects flow direction vs head pose
      if (Math.abs(r) >= 0.002 && Math.abs(r) <= 0.05) {
        radPerPx = r;
        calibrated = true;
      }
    }
  }

  // Rebuild angles: anchor at face frames (absolute yaw), integrate calibrated
  // flow through the gaps and beyond the outer anchors.
  let running = samples[0].faceYaw ?? 0;
  samples[0].angle = running;
  for (let k = 1; k < n; k++) {
    const s = samples[k];
    if (s.faceYaw != null) running = s.faceYaw;
    else running += shift[k] * radPerPx;
    s.angle = running;
  }

  // debug probe: pose sources + the calibrated flow scale
  window.__angleStats = {
    total: n,
    face: anchors.length,
    flow: n - anchors.length,
    calibrated,
    radPerPx: Number(radPerPx.toFixed(5)),
  };
}

// Head yaw (rotation about the vertical axis) from MediaPipe's 4x4 column-major
// facial transformation matrix, or null if no face was found.
function yawFromFaceResult(faceResult) {
  const mats = faceResult.facialTransformationMatrixes;
  if (!mats || mats.length === 0) return null;
  const d = mats[0].data; // column-major; rotation is the top-left 3x3
  // R02 = d[8], R22 = d[10]  →  yaw = atan2(R02, R22)
  return FACE_YAW_SIGN * Math.atan2(d[8], d[10]);
}

// Foreground luminance summed per column → a 1-D horizontal signature of the
// subject that shifts sideways as it turns.
function columnProfile(s) {
  const { colorData, maskData, maskW, maskH } = s;
  const prof = new Float32Array(maskW);
  for (let px = 0; px < maskW; px++) {
    let sum = 0;
    for (let py = 0; py < maskH; py++) {
      const idx = py * maskW + px;
      if (maskData[idx] <= FOREGROUND_THRESHOLD) continue;
      const ci = idx * 4;
      sum += 0.299 * colorData[ci] + 0.587 * colorData[ci + 1] + 0.114 * colorData[ci + 2];
    }
    prof[px] = sum;
  }
  return prof;
}

// Signed integer horizontal shift maximizing normalized cross-correlation
// between two column profiles (magnitude ≈ how far the subject moved sideways).
function bestHorizontalShift(a, b) {
  const w = a.length;
  const maxLag = Math.max(1, Math.floor(w * FLOW_MAX_LAG_RATIO));
  const am = mean(a), bm = mean(b);
  let best = 0, bestScore = -Infinity;
  for (let dLag = -maxLag; dLag <= maxLag; dLag++) {
    let dot = 0, na = 0, nb = 0;
    for (let x = 0; x < w; x++) {
      const xb = x + dLag;
      if (xb < 0 || xb >= w) continue;
      const va = a[x] - am, vb = b[xb] - bm;
      dot += va * vb; na += va * va; nb += vb * vb;
    }
    const score = na > 0 && nb > 0 ? dot / Math.sqrt(na * nb) : -Infinity;
    if (score > bestScore) { bestScore = score; best = dLag; }
  }
  return best;
}

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return arr.length ? s / arr.length : 0;
}

function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
}

// =====================
// VISUAL HULL RECONSTRUCTION (shape-from-silhouette)
// =====================
// Carve a voxel grid: each voxel is rotated into every frame's camera space (by
// that frame's estimated angle) and projected orthographically onto the mask. A
// voxel is kept only if it lands inside the silhouette in ≥ CARVE_KEEP_RATIO of
// the views it projects into — the intersection of the silhouette cones. Profile
// views (from head yaw) constrain depth; frontal views constrain width/height.
// Surviving voxels are colored from the most camera-facing view. Orthographic
// projection is an approximation (real webcam is perspective) but matches the
// silhouette→world scale used everywhere else.
function buildPointCloud() {
  const positions = [];
  const colors = [];

  if (samples.length === 0) {
    statCount.textContent = "0";
    initSceneAndRender(positions, colors);
    return;
  }

  const halfW = CARD_WIDTH / 2;
  const halfH = CARD_HEIGHT / 2;
  const trig = samples.map((s) => ({ sin: Math.sin(s.angle), cos: Math.cos(s.angle) }));

  const NX = VOXEL_NX, NY = VOXEL_NY, NZ = VOXEL_NZ;
  const occ = new Uint8Array(NX * NY * NZ);
  const colR = new Float32Array(NX * NY * NZ);
  const colG = new Float32Array(NX * NY * NZ);
  const colB = new Float32Array(NX * NY * NZ);

  // --- pass 1: carve the visual hull into an occupancy grid ---
  for (let iz = 0; iz < NZ; iz++) {
    const pz = -halfW + ((iz + 0.5) / NZ) * (2 * halfW);
    for (let iy = 0; iy < NY; iy++) {
      const py = -halfH + ((iy + 0.5) / NY) * (2 * halfH);
      const v = 0.5 - py / CARD_HEIGHT; // yCam = py → row is camera-independent
      if (v < 0 || v >= 1) continue;
      for (let ix = 0; ix < NX; ix++) {
        const px = -halfW + ((ix + 0.5) / NX) * (2 * halfW);

        let views = 0, inside = 0;
        let bestZ = -Infinity, br = 0.7, bg = 0.7, bb = 0.7;

        for (let k = 0; k < samples.length; k++) {
          const s = samples[k];
          const { sin, cos } = trig[k];
          // rotate the object-space point into this frame's camera space (about Y)
          const xCam = px * cos + pz * sin;
          const u = xCam / CARD_WIDTH + 0.5; // orthographic → mask column
          if (u < 0 || u >= 1) continue;

          const idx = ((v * s.maskH) | 0) * s.maskW + ((u * s.maskW) | 0);
          views++;
          if (s.maskData[idx] > FOREGROUND_THRESHOLD) {
            inside++;
            const zCam = -px * sin + pz * cos; // larger = closer to camera
            if (zCam > bestZ) {
              bestZ = zCam;
              const ci = idx * 4;
              br = s.colorData[ci] / 255;
              bg = s.colorData[ci + 1] / 255;
              bb = s.colorData[ci + 2] / 255;
            }
          }
        }

        if (views < MIN_CARVE_VIEWS) continue;
        if (inside < views * CARVE_KEEP_RATIO) continue; // carved away by a silhouette

        const vi = (iz * NY + iy) * NX + ix;
        occ[vi] = 1;
        colR[vi] = br; colG[vi] = bg; colB[vi] = bb;
      }
    }
  }

  // --- pass 2: drop floating specks (voxels with too few occupied neighbors) ---
  for (let iz = 0; iz < NZ; iz++) {
    for (let iy = 0; iy < NY; iy++) {
      for (let ix = 0; ix < NX; ix++) {
        const vi = (iz * NY + iy) * NX + ix;
        if (!occ[vi]) continue;

        let neigh = 0;
        for (let dz = -1; dz <= 1; dz++) {
          const z2 = iz + dz; if (z2 < 0 || z2 >= NZ) continue;
          for (let dy = -1; dy <= 1; dy++) {
            const y2 = iy + dy; if (y2 < 0 || y2 >= NY) continue;
            for (let dx = -1; dx <= 1; dx++) {
              const x2 = ix + dx; if (x2 < 0 || x2 >= NX) continue;
              if (dx === 0 && dy === 0 && dz === 0) continue;
              if (occ[(z2 * NY + y2) * NX + x2]) neigh++;
            }
          }
        }
        if (neigh < MIN_VOXEL_NEIGHBORS) continue; // isolated speck

        positions.push(px3(ix, halfW, NX), py3(iy, halfH, NY), pz3(iz, halfW, NZ));
        colors.push(colR[vi], colG[vi], colB[vi]);
      }
    }
  }

  statCount.textContent = (positions.length / 3).toLocaleString("fr-FR");
  initSceneAndRender(positions, colors);
}

// Voxel-index → world-coordinate helpers (grid is centered on the origin).
function px3(ix, halfW, NX) { return -halfW + ((ix + 0.5) / NX) * (2 * halfW); }
function py3(iy, halfH, NY) { return -halfH + ((iy + 0.5) / NY) * (2 * halfH); }
function pz3(iz, halfW, NZ) { return -halfW + ((iz + 0.5) / NZ) * (2 * halfW); }

// =====================
// THREE.JS RENDER
// =====================
let renderer, scene, camera, points, spark, splatMesh;
let yaw = 0, pitch = 0.1, radius = 3.2;
let isDragging = false, lastX = 0, lastY = 0;

function initSceneAndRender(positions, colors) {
  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000);

  // Real gaussian splatting via SparkJS: one isotropic gaussian per hull voxel
  // (same stack as the portfolio background). Falls back to THREE.Points if
  // SparkJS can't initialize (e.g. no WebGL2).
  try {
    spark = new SparkRenderer({ renderer });
    scene.add(spark);

    const center = new THREE.Vector3();
    const scales = new THREE.Vector3(SPLAT_SCALE, SPLAT_SCALE, SPLAT_SCALE);
    const quaternion = new THREE.Quaternion();
    const color = new THREE.Color();

    splatMesh = new SplatMesh({
      constructSplats: (splats) => {
        for (let i = 0; i < positions.length; i += 3) {
          center.set(positions[i], positions[i + 1], positions[i + 2]);
          color.setRGB(colors[i], colors[i + 1], colors[i + 2]);
          splats.pushSplat(center, scales, quaternion, SPLAT_OPACITY, color);
        }
      },
    });
    scene.add(splatMesh);
  } catch (err) {
    console.warn("SparkJS unavailable — falling back to THREE.Points:", err);
    renderAsPoints(positions, colors);
  }

  renderer.setAnimationLoop(renderLoop);
}

// Fallback renderer: colored point sprites (used only if SparkJS fails to init).
function renderAsPoints(positions, colors) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.PointsMaterial({
    size: 0.045,
    vertexColors: true,
    transparent: true,
    alphaTest: 0.05,
    depthWrite: false,
    sizeAttenuation: true,
  });
  points = new THREE.Points(geometry, material);
  scene.add(points);
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

// Debug: render each captured frame as a thumbnail (foreground kept, background
// dimmed) with its estimated angle and whether it came from head pose or flow.
// Reveals at a glance whether the angles/silhouettes feeding the hull are sane.
function buildFilmstrip() {
  filmstripEl.innerHTML = "";
  samples.forEach((s, i) => {
    const cv = document.createElement("canvas");
    cv.width = s.maskW;
    cv.height = s.maskH;
    const cx = cv.getContext("2d");
    const img = cx.createImageData(s.maskW, s.maskH);
    for (let p = 0; p < s.maskW * s.maskH; p++) {
      const ci = p * 4;
      const dim = s.maskData[p] > FOREGROUND_THRESHOLD ? 1 : 0.22; // dim background
      img.data[ci] = s.colorData[ci] * dim;
      img.data[ci + 1] = s.colorData[ci + 1] * dim;
      img.data[ci + 2] = s.colorData[ci + 2] * dim;
      img.data[ci + 3] = 255;
    }
    cx.putImageData(img, 0, 0);

    const card = document.createElement("div");
    card.className = "film-frame";
    const label = document.createElement("div");
    label.className = "film-label";
    const deg = Math.round((s.angle * 180) / Math.PI);
    const src = s.faceYaw != null ? "<b>face</b>" : "<span class='flow'>flux</span>";
    label.innerHTML = `#${i} · ${deg}°<br>${src}`;
    card.appendChild(cv);
    card.appendChild(label);
    filmstripEl.appendChild(card);
  });
}

function showResult() {
  phase = "result";
  stageEl.classList.add("hidden");
  container.classList.remove("hidden");
  panelEl.classList.remove("hidden");
  hintEl.classList.remove("hidden");
  // filmstrip is rebuilt lazily on toggle for this fresh reconstruction
  filmstripEl.classList.add("hidden");
  filmstripEl.innerHTML = "";
  framesBtn.textContent = "Voir les frames (debug)";
}

// =====================
// COLMAP EXPORT (frames + poses → offline 3DGS pipeline)
// =====================
// Converts the turntable capture (fixed camera / rotating subject) into the
// "camera orbiting a static object" convention that COLMAP / 3DGS trainers read.
// Format is guaranteed well-formed; the geometric convention (azimuth sign,
// axis flip) must be validated in an actual COLMAP/gsplat run — two sign knobs
// below make that a one-line fix.
const COLMAP_CAM_DIST = 3.0;    // virtual camera distance (arbitrary world scale)
const COLMAP_HFOV_DEG = 60;     // assumed webcam horizontal FOV → focal length
const COLMAP_AZIMUTH_SIGN = -1; // flip to +1 if the reconstruction comes out mirrored

const _origin = new THREE.Vector3(0, 0, 0);
const _worldUp = new THREE.Vector3(0, 1, 0);
const _flipGLtoCV = new THREE.Matrix4().makeScale(1, -1, -1); // OpenGL (−Z,Y up) → COLMAP (+Z,Y down)

// World→camera pose (COLMAP convention) for a subject rotated by `angle` about Y.
function colmapPose(angle) {
  const psi = COLMAP_AZIMUTH_SIGN * angle;
  const eye = new THREE.Vector3(
    Math.sin(psi) * COLMAP_CAM_DIST, 0, Math.cos(psi) * COLMAP_CAM_DIST
  );
  const c2w = new THREE.Matrix4().lookAt(eye, _origin, _worldUp); // rotation (OpenGL)
  c2w.setPosition(eye);
  const w2c = _flipGLtoCV.clone().multiply(c2w.clone().invert());
  const q = new THREE.Quaternion().setFromRotationMatrix(w2c);
  const t = new THREE.Vector3().setFromMatrixPosition(w2c);
  return { q, t };
}

async function exportColmap() {
  if (!samples.length || !samples[0].frameCanvas) return;
  exportBtn.disabled = true;
  const prevLabel = exportBtn.textContent;
  exportBtn.textContent = "Export…";
  await new Promise((r) => requestAnimationFrame(r)); // let the label paint
  try {
    const W = samples[0].frameCanvas.width;
    const H = samples[0].frameCanvas.height;
    const f = (W / 2) / Math.tan((COLMAP_HFOV_DEG * Math.PI) / 360);
    const cx = W / 2, cy = H / 2;

    let camerasTxt =
      "# Camera list with one line of data per camera:\n" +
      "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" +
      `1 PINHOLE ${W} ${H} ${f.toFixed(6)} ${f.toFixed(6)} ${cx} ${cy}\n`;

    let imagesTxt =
      "# Image list with two lines of data per image:\n" +
      "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n" +
      "#   POINTS2D[] as (X, Y, POINT3D_ID)  (left empty here)\n";

    const enc = new TextEncoder();
    const files = [];
    for (let i = 0; i < samples.length; i++) {
      const name = `frame_${String(i).padStart(3, "0")}.png`;
      const { q, t } = colmapPose(samples[i].angle);
      imagesTxt +=
        `${i + 1} ${q.w.toFixed(9)} ${q.x.toFixed(9)} ${q.y.toFixed(9)} ${q.z.toFixed(9)} ` +
        `${t.x.toFixed(6)} ${t.y.toFixed(6)} ${t.z.toFixed(6)} 1 ${name}\n\n`;
      // toDataURL is synchronous and reliable; toBlob can hang in some headless setups
      const pngBytes = dataUrlToBytes(samples[i].frameCanvas.toDataURL("image/png"));
      files.push({ name: `images/${name}`, data: pngBytes });
    }

    files.push({ name: "cameras.txt", data: enc.encode(camerasTxt) });
    files.push({ name: "images.txt", data: enc.encode(imagesTxt) });
    files.push({ name: "points3D.txt", data: enc.encode("# 3D point list (empty — init random or from hull)\n") });

    const url = URL.createObjectURL(new Blob([makeZip(files)], { type: "application/zip" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "scan_colmap.zip";
    document.body.appendChild(a); // some browsers require the anchor to be in the DOM
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000); // revoke late so the download isn't aborted
  } catch (err) {
    console.error("COLMAP export failed:", err);
  } finally {
    exportBtn.disabled = false;
    exportBtn.textContent = prevLabel;
  }
}

// --- minimal STORE-method ZIP writer (no deflate; PNGs are already compressed) ---
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function concatBytes(arrs) {
  let len = 0;
  for (const a of arrs) len += a.length;
  const out = new Uint8Array(len);
  let o = 0;
  for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}
function dataUrlToBytes(dataUrl) {
  const bin = atob(dataUrl.split(",", 2)[1]);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}
function makeZip(files) {
  const enc = new TextEncoder();
  const u16 = (v) => new Uint8Array([v & 0xff, (v >>> 8) & 0xff]);
  const u32 = (v) => new Uint8Array([v & 0xff, (v >>> 8) & 0xff, (v >>> 16) & 0xff, (v >>> 24) & 0xff]);

  const localChunks = [];
  const centralChunks = [];
  let offset = 0;

  for (const file of files) {
    const nameBytes = enc.encode(file.name);
    const crc = crc32(file.data);
    const size = file.data.length;
    const local = concatBytes([
      u32(0x04034b50), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(size), u32(size), u16(nameBytes.length), u16(0),
      nameBytes, file.data,
    ]);
    localChunks.push(local);
    centralChunks.push(concatBytes([
      u32(0x02014b50), u16(20), u16(20), u16(0), u16(0), u16(0), u16(0),
      u32(crc), u32(size), u32(size), u16(nameBytes.length),
      u16(0), u16(0), u16(0), u16(0), u32(0), u32(offset), nameBytes,
    ]));
    offset += local.length;
  }

  let centralSize = 0;
  for (const c of centralChunks) centralSize += c.length;
  const end = concatBytes([
    u32(0x06054b50), u16(0), u16(0), u16(files.length), u16(files.length),
    u32(centralSize), u32(offset), u16(0),
  ]);
  return concatBytes([...localChunks, ...centralChunks, end]);
}

// =====================
// RESTART
// =====================
function reset() {
  stopWebcam();
  if (renderer) {
    renderer.setAnimationLoop(null);
    if (splatMesh) { scene?.remove(splatMesh); splatMesh.dispose?.(); splatMesh = null; }
    if (spark) { scene?.remove(spark); spark.dispose?.(); spark = null; }
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
  filmstripEl.classList.add("hidden");
  filmstripEl.innerHTML = "";
  framesBtn.textContent = "Voir les frames (debug)";
  stageEl.classList.remove("hidden");
  progressArc.style.strokeDashoffset = String(ARC_CIRCUMFERENCE);
  statFrames.textContent = "—";
  statCount.textContent = "—";
  statFps.textContent = "—";
  setInstructions(defaultInstructions());
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

framesBtn.addEventListener("click", () => {
  const willShow = filmstripEl.classList.contains("hidden");
  if (willShow && filmstripEl.childElementCount === 0) buildFilmstrip();
  filmstripEl.classList.toggle("hidden");
  framesBtn.textContent = willShow ? "Masquer les frames" : "Voir les frames (debug)";
});

exportBtn.addEventListener("click", exportColmap);

// Orbit controls + resize are attached once — the container/camera persist
// across capture sessions, so re-attaching on every restart would stack
// duplicate listeners and make dragging increasingly oversensitive.
attachOrbitControls();
window.addEventListener("resize", onResize);

// Reflect the configured capture duration in the initial copy (the HTML carries
// a neutral first-paint fallback for the brief moment before this runs).
setInstructions(defaultInstructions());

import * as THREE from "three";
import {
  FilesetResolver,
  ImageSegmenter,
  PoseLandmarker,
  FaceLandmarker,
} from "@mediapipe/tasks-vision";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { pipeline, RawImage } from "@huggingface/transformers";

// =====================
// CONFIG
// =====================
const SAMPLE_INTERVAL_MS = 500;

// Capture is driven by angular coverage, not a fixed timer: the subject turns
// until enough distinct azimuth sectors have been seen (better hull), with a hard
// time cap so it always terminates (headless tests, a subject who won't turn).
const COVERAGE_BINS = 24;            // azimuth sectors over a full turn (15° each)
const COVERAGE_TARGET_RATIO = 0.5;   // stop once this fraction of bins is covered
const MIN_FRAMES = 8;                // never stop below this many kept frames
const CAPTURE_MAX_MS = 18000;        // hard cap on capture duration
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
                              // (tolerant of imperfect monocular poses; raise toward 0.9 for a tighter hull —
                              // tried 0.85/5 views to kill ghost fragments, but it just as often over-carved
                              // the body into a thin stub instead; reverted, not a net win, see TODO)
const MIN_CARVE_VIEWS = 3;    // require at least this many in-bounds views to trust a voxel
// The torso's depth is never well constrained (only frames with a detected face anchor the
// angle precisely, and nothing guarantees those also give a good shoulder-profile view) —
// it's what collapses into the recurring "cone/dress" artifact. Since the torso isn't the
// point of the demo, crop it out instead of chasing that coverage problem. The cutoff is
// derived PER CAPTURE from the measured shoulder line (see buildPointCloud), not a fixed
// world-Y constant: a fixed value only matches one specific camera framing — if the subject
// sits lower/higher in the actual webcam frame than whoever it was tuned against, the same
// constant lands well above or below their real shoulders and keeps everything or nothing.
const MIN_CARVE_Y = 0.0;          // fallback world-Y cutoff, used only if shoulders are never
                                   // confidently seen this capture (see MIN_SHOULDER_SAMPLES)
const HEAD_CROP_MARGIN = 0.05;    // world units kept below the measured shoulder line
const MIN_SHOULDER_SAMPLES = 3;   // min frames with a shoulder reading before trusting the
                                   // per-capture cutoff over the MIN_CARVE_Y fallback
const MIN_VOXEL_NEIGHBORS = 3; // drop carved voxels with fewer occupied 26-neighbors (kills floating specks)

// Rendered as real gaussians (SparkJS): one splat per surviving voxel.
const SPLAT_SCALE = 0.010;   // gaussian radius in world units (~ half the voxel pitch 0.9/64 ≈ 0.014)
const SPLAT_OPACITY = 1.0;

// Anisotropic hull splats: instead of an isotropic sphere (SPLAT_SCALE in every axis), each
// voxel with a usable surface normal (see estimateVoxelNormal) is flattened into a disc
// aligned with that normal — thin along the normal, wider in the tangent plane — to read as
// a surface patch rather than a ball. Addresses the reported "gros amas" look of hull mode.
// Voxels with a degenerate/no normal (estimateVoxelNormal returns null) fall back to the
// isotropic SPLAT_SCALE. Tune against test/turn.mp4 like the other splat-scale constants.
const ANISO_NORMAL_SCALE_MULT = 0.6; // × SPLAT_SCALE, along the normal (the "thin" axis)
const ANISO_TANGENT_SCALE_MULT = 1.0; // × SPLAT_SCALE, in the tangent plane

// Alternative reconstruction: MediaPipe's canonical 478-point face mesh instead of the
// voxel hull. Fixed per-vertex identity across frames (landmark #N is always the same
// anatomical point) means the geometry can never collapse into a "cone"/ghost shape the
// way the hull can — the mesh is just whichever single frame's own landmark detection was
// used as reference, warts and all, but always face-shaped. Trade-off: face only, no
// torso — fine since the head-crop already dropped the torso. See TODO in
// .docs/CAPTURE_DEMO.md ("changement de méthode").
const RECON_MODES = ["hull", "mesh", "dense", "fused"]; // cycle order for the mode-toggle button
const RECON_MODE_DEFAULT = "hull"; // default view after a capture
// MediaPipe face landmarks are normalized to the image (x,y in [0,1], z roughly on a
// similar scale, more-negative = closer to camera). Converted to world units to roughly
// match the hull's head size (empirically tuned, see TODO).
const FACE_MESH_SCALE_XY = 3.4;
const FACE_MESH_SCALE_Z = 3.4;
// MediaPipe's own "identity rotation" orientation faces the opposite way from this scene's
// established camera-forward convention (validated for the hull) — without this, mesh/dense/
// fused default to showing the BACK of the head, not the face (reported after real captures;
// invisible in most static screenshots since SparkJS splats have no backface culling, so the
// colored points still "look like a face" even facing the wrong way). A 180 deg rotation
// about the vertical axis = negate X and Z.
const FACE_MESH_FRONT_SIGN = -1;
const FACE_MESH_SPLAT_SCALE = 0.017; // fallback gaussian radius when per-point scale isn't
                                      // available (e.g. an empty capture) — not otherwise used
                                      // by mesh mode, see FACE_MESH_SPLAT_SCALE_MIN/MAX below.
                                      // Tuned against test/turn.mp4: 0.028 blurred into a
                                      // featureless oval, 0.014 gave a visible-pointillist but
                                      // clearly readable face, 0.017 settled between the two —
                                      // as a single global radius. Superseded by a per-landmark
                                      // adaptive radius (below) once it became clear a single
                                      // scalar can't fit both the eyes (dense landmarks, needs
                                      // a small radius to stay sharp) and the cheeks (sparse
                                      // landmarks, needs a larger radius to avoid gaps) at once.
const FACE_MESH_SPLAT_SCALE_MIN = 0.010; // radius floor, for the densest landmark clusters
const FACE_MESH_SPLAT_SCALE_MAX = 0.026; // radius ceiling, for the sparsest (e.g. cheeks)
const FACE_MESH_SPLAT_SCALE_NN_MULT = 0.55; // world-space nearest-neighbor distance * this
                                      // = radius (before clamping to MIN/MAX) — tuned so the
                                      // eye/brow region (tightly packed landmarks) lands near
                                      // MIN and the cheeks (sparsest) land near MAX.

// Anisotropic mesh/dense/fused splats: same idea as the hull's ANISO_* constants, but the
// per-point normal comes from a local plane fit (estimatePointCloudNormals) over each point's
// neighbors within *_NORMAL_RADIUS, since these modes have a point cloud instead of hull's
// voxel grid. *_NORMAL_MIN_NEIGHBORS points must fall within that radius or the point falls
// back to an isotropic splat (same rule as hull's degenerate-gradient case). Dense/fused share
// one radius since both come from the same per-pixel sampling grid at comparable density.
const MESH_NORMAL_RADIUS = 0.08;
const MESH_NORMAL_MIN_NEIGHBORS = 5;
const DENSE_NORMAL_RADIUS = 0.025;
const DENSE_NORMAL_MIN_NEIGHBORS = 6;

// Denser variant of the mesh mode: instead of one gaussian per landmark (478 points,
// visibly pointillist), sample depth across a fine pixel grid over the WHOLE face region of
// the single reference frame, not just the 478 measured points. Depth at each sampled pixel
// comes from a real monocular depth model when available (see DEPTH_MODEL_ID below), falling
// back to inverse-distance-weighted interpolation between the 478 landmarks (an ESTIMATE
// between measurements, not itself a measurement) when the depth model failed to load — e.g.
// no network, or a browser without WASM/WebGPU support. Single reference frame only — see the
// "fused" mode below for a multi-frame variant.
const DENSE_GRID_STEP_PX = 3;      // sample every Nth full-res pixel inside the face region
const DENSE_IDW_NEIGHBORS = 5;     // nearest landmarks used to interpolate depth per pixel
const DENSE_BBOX_PADDING = 0.06;   // extra margin (normalized) around the landmark bounding box
const DENSE_SPLAT_SCALE = 0.007;   // much smaller than FACE_MESH_SPLAT_SCALE — points are
                                    // far denser here, a large radius would blur immediately

// Real monocular depth model (Depth Anything v2, via transformers.js/onnxruntime-web), used
// by dense AND fused modes instead of IDW interpolation between the 478 landmarks — a genuine
// per-pixel depth estimate (real nose bulge etc.) rather than a smooth guess between sparse
// points. Loaded in parallel with the MediaPipe models but treated as OPTIONAL: if it fails (no
// network, no WASM/WebGPU) both modes silently fall back to the IDW path, same as before this
// existed.
const DEPTH_MODEL_ID = "onnx-community/depth-anything-v2-small";
const DEPTH_MODEL_DTYPE = "q8"; // quantized, ~38MB — see .docs/CAPTURE_DEMO.md for the size
                                 // comparison against fp32/fp16/q4/q4f16 that led to this pick
// Below this fit quality (fitAffine's r2), a view's depth-model calibration is rejected in
// favor of IDW for that view (see calibratedDepthAt) — a poor affine fit (narrow/noisy
// landmark-z range for that particular frame) doesn't just add noise, it can shift the WHOLE
// view's depth onto a wrong scale/offset. In fused mode that showed up as a distinct "ghost"
// layer of points floating in front of/behind the real surface (side-by-side vs. dense mode on
// the same capture — see .docs/CAPTURE_DEMO.md), since each view calibrates independently and
// nothing constrained the fits to agree with each other.
const MIN_DEPTH_CALIB_R2 = -Infinity; // disabled — see .docs/CAPTURE_DEMO.md, r2 doesn't
                                       // separate good/bad calibrations for this model (measured
                                       // 0.42-0.66 across every view including the reference)
// Per-pixel trust radius for the depth model (normalized image-space distance to the nearest
// of the 478 landmarks) — beyond this, the model's calibrated output is unconstrained
// extrapolation (no real anchor nearby, e.g. hair/forehead margin) and is discarded in favor
// of the IDW fallback for that pixel specifically, even when the view's overall calibration
// looks fine. In fused mode, two views can each extrapolate confidently but DIFFERENTLY in an
// unanchored region, producing a distinct "ghost" surface floating apart from the real one —
// diagnosed by comparing dense (single view, never ghosts) against fused (multiple independently
// calibrated views) on the same capture, and noticing the ghosting concentrated at the hairline/
// edges where landmark coverage is sparsest, not across the face where landmarks are dense.
const DEPTH_MODEL_MAX_LANDMARK_DIST = 0.05;
// Above this mean per-landmark error (see viewAlignError), a fused-mode view's Rdelta rotation
// disagrees enough with the reference that fusing it in produces a visibly separate "ghost"
// surface rather than added coverage — measured directly (not guessed): a view whose points
// visibly split off into their own floating layer measured ~0.006-0.008, a well-aligned
// near-reference view measured ~0.002. Set between the two. Tune against real captures if this
// still lets a bad view through or rejects too many usable ones.
// Calibrated using test/reconstruction_quality.py against test/turn.mp4: 0.004 (first guess)
// routinely rejected every extra view. With the gate off entirely, alignErr clustered
// 0.003-0.0095 with no clean bimodal split, and geometric coherence (0.95-0.98) wasn't far
// below dense/mesh's own baseline — so a hard per-view cutoff has limited room to help without
// also rejecting normal views. 0.008 (above the typical range, cutting only the worst
// outliers) measured across 3 more runs: real multi-view fusion still happens (2-3 extra views
// survive per run, ~2-2.5x dense's point count) AND photometric consistency improved
// (mean multi-view color variance 0.006-0.007, vs 0.009-0.028 with the gate off) — the first
// change in this investigation with an actual measured improvement, not just a guess.
const DENSE_FUSION_MAX_ALIGN_ERR = 0.008;

// Multi-frame fusion of the dense mode: the plain "dense" mode above uses only the single
// most-frontal frame, so it doesn't actually exploit the fact that the subject turns — it's
// equivalent to one photo + a monocular depth model. This mode adds a few EXTRA views
// spread across the captured yaw range, each aligned into the SAME fixed canonical
// orientation as the reference (see toCanonicalOrientation), to fill in what a single
// frontal view can't see (side of the nose, cheek edges). Alignment uses each view's own
// measured rotation matrix (facialTransformationMatrix, see faceRotationMatrix) — full 3D
// rotation (pitch/roll included), not just the single-axis yaw used elsewhere in this file.
// No Kabsch/SVD needed: rotation matrices are orthogonal, so inverting one is just a
// transpose (transpose3). Points from different views that land near each other are
// averaged (position + color) instead of left as separate overlapping splats — see the
// merge grid below.
const DENSE_FUSION_MAX_VIEWS = 4;      // reference + up to this many extra views
const DENSE_FUSION_STEP_MULT = 1;      // extra views sample at the same grid density as the
                                        // reference — was 2 (coarser), but that made the wing
                                        // contributed by each extra view visibly patchier/
                                        // sparser than the reference's own tight grid, reading
                                        // as noise right at the seam. Testing at 1 first (see
                                        // .docs/CAPTURE_DEMO.md).
const DENSE_FUSION_MERGE_CELL = 0.010; // world units: points within the same cell are
                                        // averaged into one instead of left overlapping
const DENSE_FUSION_MAX_YAW_DELTA = (30 * Math.PI) / 180; // extra views beyond this yaw delta
                                        // from the reference are excluded from the candidate
                                        // pool. Was 55 deg; tightened after a real capture
                                        // (weird2.png) showed a large-delta view (+40 deg)
                                        // rendering as a whole flat plane floating disconnected
                                        // beside the head in profile (invisible from the front —
                                        // see side_view_check.py / capture-demo-depth-verification
                                        // memory) rather than curving into the face. Each dense
                                        // view has little real depth relief (flatness
                                        // limitation), so a large-angle view's near-flat card
                                        // just gets pushed sideways/backward by the rotation
                                        // instead of blending in — 30 deg was the largest delta
                                        // that stayed visually attached in a side-view retest.
const DENSE_FUSION_WING_INNER_FRAC = 0.55; // extra (non-reference) views only contribute their
                                        // OUTER band (beyond this fraction of their own
                                        // half-width from their own centroid) — the part a
                                        // frontal reference can't see (cheek/jaw/ear side).
                                        // Each dense view is built from IDW-interpolated depth
                                        // over the WHOLE face bbox, which has much less relief
                                        // than real face geometry (documented flatness
                                        // limitation) — near enough to a flat rectangular card.
                                        // Fusing several FULL cards at different yaw angles
                                        // rotates each nearly-flat sheet into its own plane,
                                        // producing a fan of crossing sheets instead of a
                                        // volume (reported by a real user capture, weird2.png).
                                        // Dropping each extra view's central portion — which
                                        // just redraws the same flat face the reference already
                                        // covers — keeps only the edge strips, so the fan effect
                                        // shrinks to a thin sliver instead of a full crossing card.

// Per-frame rotation angle is measured from real head pose (MediaPipe
// FaceLandmarker yaw). When no face is visible we fall back to integrating the
// signed horizontal optical flow between frames. Nothing assumes a full 360°
// turn — the subject can move freely (e.g. show one cheek, then the other).
const FLOW_MAX_LAG_RATIO = 0.25; // horizontal-shift search window (× frame width)
const FLOW_RAD_PER_PX = 0.012;   // optical-flow px → radians (fallback only)
const FACE_YAW_SIGN = 1;         // flip to -1 if left/right ends up mirrored
const ANGLE_SMOOTH_RADIUS = 1;   // final angle trajectory smoothing: +-N neighbors (triangular window),
                                  // damps per-frame FaceLandmarker/flow jitter; 0 disables smoothing
const MIN_CALIBRATION_PAIRS = 5;   // min pooled face-pairs before trusting the calibrated flow scale
const MIN_CALIBRATION_FLOW_PX = 60; // min *total* pooled flow (px) across those pairs — empirically, runs
                                     // just above a looser threshold (3 pairs/40px) still swung ~2.5x;
                                     // 5 pairs/60px was the regime where repeated runs agreed within ~6%
                                     // (see TODO in .docs/CAPTURE_DEMO.md). Below this, FLOW_RAD_PER_PX
                                     // (a safe fixed default) is used instead of a noisy calibrated value.

// Body azimuth from the shoulder line (MediaPipe PoseLandmarker worldLandmarks):
// used to refine the angle where the face is absent, but only when it agrees with
// the flow/face estimate — the shoulder-depth sign flips front/back near profile.
// Disabled by default: even under a tight agreement gate, shoulder-depth yaw proved
// too noisy in practice (a real test clip showed 6/24 frames sourced from body yaw,
// collapsing the visual hull into a "dress" shape) — see TODO #1 in .docs/CAPTURE_DEMO.md.
const BODY_YAW_REINFORCE = false; // flip true only after re-validating the agreement gate below
const BODY_AGREE_TOL = (20 * Math.PI) / 180; // body yaw must be this close to the flow/face base to be trusted
const BODY_YAW_SIGN = 1;         // flip to -1 if the torso rotation ends up mirrored
const SHOULDER_VIS_MIN = 0.5;    // min landmark visibility to trust the shoulder line
const L_SHOULDER = 11, R_SHOULDER = 12; // MediaPipe pose landmark indices

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
const progressRing = document.getElementById("demo-progress-ring");
const startBtn = document.getElementById("demo-start-btn");
const errorEl = document.getElementById("demo-error");

const container = document.getElementById("demo-container");
const panelEl = document.getElementById("demo-panel");
const hintEl = document.getElementById("demo-hint");
const restartBtn = document.getElementById("demo-restart");
const framesBtn = document.getElementById("demo-frames-btn");
const modeBtn = document.getElementById("demo-mode-btn");
const exportBtn = document.getElementById("demo-export-btn");
const saveJsonBtn = document.getElementById("demo-save-json-btn");
const loadJsonBtn = document.getElementById("demo-load-json-btn");
const loadJsonInput = document.getElementById("demo-load-json-input");
const filmstripEl = document.getElementById("demo-filmstrip");
const statFps = document.getElementById("stat-fps");
const statCount = document.getElementById("stat-count");
const statFrames = document.getElementById("stat-frames");

const debugEl = document.getElementById("demo-debug");
const dbgBins = document.getElementById("dbg-bins");
const dbgSrc = document.getElementById("dbg-src");
const dbgYaw = document.getElementById("dbg-yaw");
const dbgDetected = document.getElementById("dbg-detected");
const dbgTime = document.getElementById("dbg-time");

const ARC_CIRCUMFERENCE = 2 * Math.PI * 46;

// =====================
// STATE
// =====================
let stream = null;
let segmenter = null;
let poseLandmarker = null;
let faceLandmarker = null;
let depthEstimator = null; // null if unavailable (load failed) — dense mode falls back to IDW
let phase = "idle"; // idle | loading | ready | capturing | reconstructing | result
let samples = [];
let reconMode = RECON_MODE_DEFAULT; // "hull" | "mesh" — which reconstruction is on screen
let offscreenCanvas = null;
let offscreenCtx = null;

function setError(msg) {
  errorEl.textContent = msg;
}

function setInstructions(text) {
  instructionsEl.textContent = text;
}

// Single source of truth for the idle/reset instruction copy — the capture is
// coverage-driven (turn until the arc fills), capped at CAPTURE_MAX_MS.
function defaultInstructions() {
  return "Ta webcam te capture pendant que tu tournes lentement sur toi-même, jusqu'à ce que l'arc se " +
    "remplisse. Un modèle de segmentation isole ta silhouette du fond en temps réel, et une " +
    "reconstruction 3D approximative est générée — tout se passe dans ton navigateur, rien n'est " +
    "envoyé sur un serveur.";
}

// Loads the monocular depth model (see DEPTH_MODEL_ID). Never throws — an unavailable depth
// model (no network, no WASM/WebGPU, CDN blocked) just leaves depthEstimator null, and dense
// mode falls back to its IDW path, same as before this model existed.
async function loadDepthEstimator() {
  try {
    return await pipeline("depth-estimation", DEPTH_MODEL_ID, { dtype: DEPTH_MODEL_DTYPE });
  } catch (err) {
    console.warn("Depth model unavailable — dense mode falls back to landmark IDW interpolation:", err);
    return null;
  }
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
    // Depth model loads in parallel with the MediaPipe fileset (independent network
    // fetches) but is non-fatal: loadDepthEstimator() catches its own errors and resolves
    // to null, since dense mode has a working fallback (IDW) without it.
    const [vision, depthEst] = await Promise.all([
      FilesetResolver.forVisionTasks(WASM_BASE),
      loadDepthEstimator(),
    ]);
    depthEstimator = depthEst;
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
  startBtn.textContent = "Démarrer la capture";
  setInstructions("Place-toi face à la webcam, puis tourne lentement sur toi-même jusqu'à ce que l'arc se remplisse.");
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

  // Torso azimuth from the shoulder line (absolute, per-frame, no drift), or null
  // when the shoulders aren't confidently visible.
  const { bodyYaw, shoulderVis, shoulderImgY } = yawFromPoseResult(poseResult);

  // Real head pose (yaw) for this frame, or null if no face is visible.
  const faceResult = faceLandmarker.detectForVideo(videoEl, ts);
  const faceYaw = yawFromFaceResult(faceResult);
  // Raw 478-point face mesh (fixed topology, per-vertex identity stable across frames) —
  // used by the canonical-mesh reconstruction mode, kept lightweight (no pixel data, just
  // the landmark coordinates; color is sampled on demand from frameCanvas at build time).
  const faceLandmarks = faceResult.faceLandmarks && faceResult.faceLandmarks[0]
    ? faceResult.faceLandmarks[0]
    : null;
  const faceRot = faceRotationMatrix(faceResult);

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

  return { maskData, colorData, maskW, maskH, personDetected, faceYaw, faceLandmarks, faceRot, bodyYaw, shoulderVis, shoulderImgY, frameCanvas };
}

// =====================
// COVERAGE-DRIVEN CAPTURE
// =====================
// Instead of a fixed timer, the capture ends once enough distinct azimuth sectors
// have been seen — a wider spread of angles carves a better hull. A live coach
// nudges the subject to keep turning, and a hard time cap always terminates.
function startCapture() {
  phase = "capturing";
  samples = [];
  startBtn.disabled = true;
  progressRing.classList.remove("hidden");
  debugEl.classList.remove("hidden");

  const coveredBins = new Set();
  const targetBins = Math.max(1, Math.round(COVERAGE_BINS * COVERAGE_TARGET_RATIO));
  const startTime = performance.now();
  // Guidance state: nudge the subject when nothing useful has happened for a
  // while, instead of repeating the same generic sentence the whole time.
  let lastProgressAt = startTime;
  let noPersonStreak = 0;

  const sampleTimer = setInterval(() => {
    const elapsed = performance.now() - startTime;

    // PoseLandmarker only gates frames where a subject is present. The absolute
    // angle used to *drive coverage* comes from the live torso/head pose; the
    // final per-frame angles are still recomputed afterward (with flow fallback),
    // so no constant-speed / full-turn assumption is baked into the capture.
    const frame = runSegmentationFrame(true);
    if (frame && frame.personDetected) {
      noPersonStreak = 0;
      samples.push(frame);
      statFrames.textContent = String(samples.length);

      const sizeBefore = coveredBins.size;
      const live = frame.bodyYaw != null ? frame.bodyYaw : frame.faceYaw;
      if (live != null) {
        const bin = ((Math.round((live / (2 * Math.PI)) * COVERAGE_BINS) % COVERAGE_BINS) + COVERAGE_BINS) % COVERAGE_BINS;
        coveredBins.add(bin);
      }
      if (coveredBins.size > sizeBefore) lastProgressAt = performance.now();

      dbgSrc.textContent = frame.bodyYaw != null ? "body" : (frame.faceYaw != null ? "face" : "none");
      dbgYaw.textContent = live != null ? `${Math.round((live * 180) / Math.PI)}°` : "—";
    } else {
      noPersonStreak += 1;
      dbgSrc.textContent = "none";
      dbgYaw.textContent = "—";
    }
    dbgDetected.textContent = frame && frame.personDetected ? "oui" : "non";
    dbgBins.textContent = `${coveredBins.size}/${targetBins}`;
    dbgTime.textContent = `${(elapsed / 1000).toFixed(1)}s`;

    // Ring fills with coverage (not time); coach text nudges toward the target,
    // and switches message when nothing useful happened for a while (no person
    // in frame, or no new angle covered despite being visible).
    const cov = Math.min(coveredBins.size / targetBins, 1);
    progressArc.style.strokeDashoffset = String(ARC_CIRCUMFERENCE * (1 - cov));
    const stalledMs = performance.now() - lastProgressAt;
    let coach;
    if (cov >= 1) {
      coach = "✓ Couverture suffisante — tu peux t'arrêter.";
    } else if (noPersonStreak >= 3) {
      coach = "On ne te voit plus bien : recule-toi et reste face à la caméra.";
    } else if (stalledMs > 3000) {
      coach = `Continue de tourner, il reste des angles à montrer (${Math.round(cov * 100)}%)…`;
    } else {
      coach = `Tourne lentement sur toi-même — ${Math.round(cov * 100)}% couvert…`;
    }
    setInstructions(coach);

    const done = coveredBins.size >= targetBins && samples.length >= MIN_FRAMES;
    if (done || elapsed >= CAPTURE_MAX_MS) {
      clearInterval(sampleTimer);
      // debug probe: how the capture ended (coverage vs time cap) + coverage state
      window.__captureStats = {
        endedBy: done ? "coverage" : "cap",
        coveredBins: coveredBins.size,
        targetBins,
        frames: samples.length,
        elapsedMs: Math.round(elapsed),
      };
      finishCapture();
    }
  }, SAMPLE_INTERVAL_MS);
}

function finishCapture() {
  phase = "reconstructing";
  stopWebcam();
  debugEl.classList.add("hidden");
  setInstructions("Reconstruction du nuage de points…");

  requestAnimationFrame(() => {
    samples.forEach(keepLargestForeground); // drop stray mask islands (non-human)
    estimateFrameAngles();
    reconMode = RECON_MODE_DEFAULT;
    renderReconstruction();
    updateModeButton();
    showResult();
  });
}

// Dispatches to whichever reconstruction is currently selected — used both for the
// initial render and when the mode-toggle button switches views on already-captured
// samples (no re-capture, no re-run of angle estimation).
function renderReconstruction() {
  if (reconMode === "mesh") buildFaceMesh();
  else if (reconMode === "dense") buildDenseFaceMesh();
  else if (reconMode === "fused") buildDenseFusedFaceMesh();
  else buildPointCloud();
}

// Labelled by what clicking switches TO (the next mode in the cycle), not the current one —
// matches the original hull<->mesh toggle's convention.
const RECON_MODE_LABEL = {
  hull: "Vue : buste (hull)",
  mesh: "Vue : visage (mesh)",
  dense: "Vue : visage (dense)",
  fused: "Vue : visage (dense fusionné)",
};

function nextReconMode() {
  return RECON_MODES[(RECON_MODES.indexOf(reconMode) + 1) % RECON_MODES.length];
}

function updateModeButton() {
  const hasFace = samples.some((s) => s.faceLandmarks);
  modeBtn.disabled = !hasFace; // no face ever detected this capture — mesh/dense views are empty
  modeBtn.textContent = RECON_MODE_LABEL[nextReconMode()];
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
// POSE ESTIMATION (optical-flow backbone anchored by torso/head pose)
// =====================
// Assign each frame a rotation angle. Optical flow (smooth, no ambiguity) is the
// backbone; the absolute pose anchors (torso azimuth, else head yaw) fix its scale
// and offset. Pose is snapped per-frame ONLY when it agrees with the flow-predicted
// angle — this rejects MediaPipe's front/back shoulder-depth flips (which jump
// ~150–230° near profile) while keeping accurate frontal pose. No assumption of a
// full turn: free movement and back-and-forth are handled by the signed flow.
function estimateFrameAngles() {
  const n = samples.length;
  if (n === 0) return;

  const profiles = samples.map(columnProfile);

  // Signed horizontal optical flow (px) between consecutive frames.
  const shift = new Float32Array(n);
  for (let k = 1; k < n; k++) shift[k] = bestHorizontalShift(profiles[k - 1], profiles[k]);

  // Calibrate the px→radians scale from FACE anchors only. Head yaw is a
  // well-posed measurement (no front/back ambiguity), whereas torso yaw from
  // shoulder depth has smooth-but-wrong mirror branches that would poison the
  // scale. Pooled (flow-weighted) ratio over consecutive face pairs, not a
  // median of individual per-pair ratios — a short capture typically has only
  // a handful of valid face pairs, and a plain median let a single noisy
  // short-flow pair swing the scale by up to 6x between otherwise-identical
  // runs. Pooling sum(dPose)/sum(pxSum) instead weights each pair by how much
  // flow it actually contributes, so one short/noisy pair can't dominate.
  const JUMP_TOL = (70 * Math.PI) / 180;  // face-pair disagreement gate for flow calibration only (not body agreement, see BODY_AGREE_TOL)
  const faceIdx = [];
  for (let k = 0; k < n; k++) if (samples[k].faceYaw != null) faceIdx.push(k);
  let poseSum = 0, pxSumTotal = 0, pairCount = 0;
  for (let a = 0; a < faceIdx.length - 1; a++) {
    const i = faceIdx[a], j = faceIdx[a + 1];
    const dPose = wrapPi(samples[j].faceYaw - samples[i].faceYaw);
    let pxSum = 0;
    for (let k = i + 1; k <= j; k++) pxSum += shift[k];
    if (Math.abs(pxSum) > 5 && Math.abs(dPose) < JUMP_TOL) {
      poseSum += dPose;
      pxSumTotal += pxSum;
      pairCount++;
    }
  }
  let radPerPx = FLOW_RAD_PER_PX;
  let calibrated = false;
  if (pairCount >= MIN_CALIBRATION_PAIRS && Math.abs(pxSumTotal) > MIN_CALIBRATION_FLOW_PX) {
    const r = poseSum / pxSumTotal;
    if (Math.abs(r) >= 0.002 && Math.abs(r) <= 0.05) { radPerPx = r; calibrated = true; }
  }

  // Cumulative optical flow — a smooth, unambiguous (but unscaled) rotation signal.
  const fc = new Float64Array(n);
  for (let k = 1; k < n; k++) fc[k] = fc[k - 1] + shift[k];

  // Rebuild angles anchored on FACE frames (the gold standard: no front/back
  // ambiguity). Within a gap bounded by two face anchors we interpolate the angle
  // *proportionally to the accumulated flow* (so the sweep follows the real motion,
  // not a straight line in time); flow magnitude also tells us how many full turns
  // the face wrap hides. Outside the outer face anchors we integrate flow one-sided.
  // Torso yaw would only *refine* a frame when it agrees with that base (rejecting the
  // shoulder-depth flips) — disabled by default (BODY_YAW_REINFORCE); the raw flow base
  // is used whenever no face is present.
  const TWO_PI = 2 * Math.PI;
  const baseAngle = (k) => {
    let lo = -1, hi = -1;
    for (const f of faceIdx) { if (f <= k) lo = f; if (f >= k && hi === -1) hi = f; }
    if (lo !== -1 && hi !== -1 && lo !== hi) {           // between two face anchors
      const dFlow = (fc[hi] - fc[lo]) * radPerPx;
      const dWrap = wrapPi(samples[hi].faceYaw - samples[lo].faceYaw);
      const turns = Math.round((dFlow - dWrap) / TWO_PI); // full turns the face can't see
      const dTotal = dWrap + TWO_PI * turns;
      const denom = fc[hi] - fc[lo];
      const frac = Math.abs(denom) > 1e-6 ? (fc[k] - fc[lo]) / denom : 0;
      return samples[lo].faceYaw + dTotal * frac;
    }
    if (lo !== -1) return samples[lo].faceYaw + (fc[k] - fc[lo]) * radPerPx; // after last face
    if (hi !== -1) return samples[hi].faceYaw + (fc[k] - fc[hi]) * radPerPx; // before first face
    return fc[k] * radPerPx;                              // no face at all → relative flow
  };

  let bodyN = 0, faceN = 0, flowN = 0, rejected = 0, bodyDetected = 0;
  for (let k = 0; k < n; k++) {
    const s = samples[k];
    const base = baseAngle(k);
    if (s.bodyYaw != null) bodyDetected++; // measured regardless of whether it's used as a source
    if (s.faceYaw != null) {
      s.angle = s.faceYaw; s.angleSrc = "face"; faceN++;
    } else if (BODY_YAW_REINFORCE && s.bodyYaw != null && Math.abs(wrapPi(s.bodyYaw - base)) <= BODY_AGREE_TOL) {
      s.angle = s.bodyYaw; s.angleSrc = "body"; bodyN++; // torso agrees → use its precise value
    } else {
      if (BODY_YAW_REINFORCE && s.bodyYaw != null) rejected++; // torso pose existed but disagreed → flip/mirror branch
      s.angle = base; s.angleSrc = "flow";  flowN++;
    }
  }

  // Light temporal smoothing: circular mean over a small triangular window (so
  // wraparound at +-180 deg doesn't corrupt the average) to damp frame-to-frame
  // jitter, mostly from noisy per-frame FaceLandmarker yaw estimates.
  if (ANGLE_SMOOTH_RADIUS > 0) {
    const rawAngles = samples.map((s) => s.angle);
    for (let k = 0; k < n; k++) {
      let sx = 0, sy = 0;
      for (let d = -ANGLE_SMOOTH_RADIUS; d <= ANGLE_SMOOTH_RADIUS; d++) {
        const j = k + d;
        if (j < 0 || j >= n) continue;
        const w = ANGLE_SMOOTH_RADIUS + 1 - Math.abs(d); // triangular, center heaviest
        sx += w * Math.cos(rawAngles[j]);
        sy += w * Math.sin(rawAngles[j]);
      }
      samples[k].angle = Math.atan2(sy, sx);
    }
  }

  // debug probe: pose sources + the calibrated flow scale
  window.__angleStats = {
    total: n,
    body: bodyN,
    bodyDetected, // frames with a measurable torso yaw, whether or not BODY_YAW_REINFORCE used it
    face: faceN,
    flow: flowN,
    rejected, // pose anchors dropped as front/back flips
    calibrated,
    calibrationPairs: pairCount, // face-pairs pooled into the calibration (see MIN_CALIBRATION_PAIRS)
    calibrationFlowPx: Math.round(pxSumTotal),
    radPerPx: Number(radPerPx.toFixed(5)),
  };
  // debug probe: per-frame angle trajectory (deg), raw signals, source, visibility
  const R2D = 180 / Math.PI;
  let fcum = 0;
  window.__angleTrace = samples.map((s, i) => {
    fcum += shift[i]; // shift[0]=0
    return {
      i,
      deg: Math.round(s.angle * R2D),
      rawBody: s.bodyYaw != null ? Math.round(s.bodyYaw * R2D) : null,
      rawFace: s.faceYaw != null ? Math.round(s.faceYaw * R2D) : null,
      flowPx: Math.round(shift[i]),
      flowCumDeg: Math.round(fcum * radPerPx * R2D),
      src: s.angleSrc,
      vis: s.shoulderVis != null ? Number(s.shoulderVis.toFixed(2)) : null,
    };
  });
}

// Torso yaw from the shoulder line. worldLandmarks are metric (origin at the
// hips) with a depth (z toward/away from camera), so the shoulder→shoulder vector
// rotates about the vertical as the body turns: yaw = atan2(dz, dx). Absolute and
// per-frame, and stays valid once the face turns away — BUT the z sign flips
// front/back near profile, so estimateFrameAngles only trusts it when it agrees
// with the face/flow estimate. Returns { bodyYaw, shoulderVis, shoulderImgY }
// (bodyYaw null if the shoulders aren't confidently visible; shoulderImgY — the
// normalized 0..1 image-space vertical position of the shoulder midpoint — is
// reported whenever landmarks exist at all, independent of yaw confidence, since
// it's only used to locate the shoulder line for the head-crop, not to measure
// rotation).
function yawFromPoseResult(poseResult) {
  const world = poseResult.worldLandmarks && poseResult.worldLandmarks[0];
  const img = poseResult.landmarks && poseResult.landmarks[0];
  if (!world || !img) return { bodyYaw: null, shoulderVis: 0, shoulderImgY: null };
  const L = world[L_SHOULDER], R = world[R_SHOULDER];
  const Limg = img[L_SHOULDER], Rimg = img[R_SHOULDER];
  const vis = Math.min(Limg?.visibility ?? 0, Rimg?.visibility ?? 0);
  const shoulderImgY = (Limg && Rimg) ? (Limg.y + Rimg.y) / 2 : null;
  if (!L || !R || vis < SHOULDER_VIS_MIN) return { bodyYaw: null, shoulderVis: vis, shoulderImgY };
  const dx = L.x - R.x, dz = L.z - R.z;
  return { bodyYaw: BODY_YAW_SIGN * Math.atan2(dz, dx), shoulderVis: vis, shoulderImgY };
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

// Full 3x3 head-rotation matrix (row-major [r00,r01,r02, r10,r11,r12, r20,r21,r22]) from the
// same facialTransformationMatrix, or null if no face was found. Used by the "fused" dense
// mode to align frames by their actual measured rotation (pitch/roll included), instead of
// the single-axis yaw approximation used for the hull's angle model.
function faceRotationMatrix(faceResult) {
  const mats = faceResult.facialTransformationMatrixes;
  if (!mats || mats.length === 0) return null;
  const d = mats[0].data; // column-major 4x4
  return [d[0], d[4], d[8], d[1], d[5], d[9], d[2], d[6], d[10]];
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

// Median (robust to outliers, e.g. an occasional bad shoulder-landmark reading).
function median(arr) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  const m = a.length >> 1;
  return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

// Minimal 3x3 matrix helpers for undoing a frame's full measured head rotation
// (facialTransformationMatrix) instead of a single-axis yaw approximation — see
// toCanonicalOrientation. Rotation matrices are orthogonal, so the inverse is just the
// transpose — no decomposition needed.
function transpose3(m) {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}
function applyMat3(m, x, y, z) {
  return [
    m[0] * x + m[1] * y + m[2] * z,
    m[3] * x + m[4] * y + m[5] * z,
    m[6] * x + m[7] * y + m[8] * z,
  ];
}
// a * b (apply b's rotation first, then a's) — used by buildDenseFusedFaceMesh to compose a
// view-to-reference delta rotation instead of each view's absolute orientation (see the
// comment on that Rdelta computation for why relative alignment, not toCanonicalOrientation,
// is used there).
function multiply3(a, b) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  return r;
}

// How well a candidate fused-mode view's Rdelta rotation actually agrees with the reference,
// measured directly rather than assumed: transform the view's OWN 478 landmarks (relative to
// its centroid) through Rdelta, and compare each to the reference's own landmark at the same
// index (relative to the reference's centroid) — landmark topology is fixed (index i is always
// the same anatomical point in every frame), so this is a real, per-index correspondence, not
// an approximation. Mean distance over all 478 — large values mean this view's whole rotation
// is untrustworthy (e.g. MediaPipe head-pose grows less reliable near-profile), which fusing in
// anyway produces a visibly separate "ghost" surface (diagnosed by color-coding points by
// source view — see .docs/CAPTURE_DEMO.md). Diagnostic/gating use only — never called per-pixel.
function viewAlignError(view, reference, refRotT) {
  const lm = view.faceLandmarks, refLm = reference.faceLandmarks;
  let lcx = 0, lcy = 0, lcz = 0, rcx = 0, rcy = 0, rcz = 0;
  for (const p of lm) { lcx += p.x; lcy += p.y; lcz += p.z; }
  lcx /= lm.length; lcy /= lm.length; lcz /= lm.length;
  for (const p of refLm) { rcx += p.x; rcy += p.y; rcz += p.z; }
  rcx /= refLm.length; rcy /= refLm.length; rcz /= refLm.length;
  const Rdelta = multiply3(view.faceRot, refRotT);
  let sumErr = 0;
  for (let i = 0; i < lm.length; i++) {
    const [rx, ry, rz] = applyMat3(Rdelta, lm[i].x - lcx, lm[i].y - lcy, lm[i].z - lcz);
    const tx = refLm[i].x - rcx, ty = refLm[i].y - rcy, tz = refLm[i].z - rcz;
    sumErr += Math.hypot(rx - tx, ry - ty, rz - tz);
  }
  return sumErr / lm.length;
}

// For each point, the distance to its nearest neighbor among the same list — used by mesh
// mode to scale each gaussian to its local landmark density (see FACE_MESH_SPLAT_SCALE_MIN/
// MAX/NN_MULT). O(n^2) but n=478, computed once per render — negligible. A point with no
// neighbors (list of length 1) gets Infinity; callers must clamp before using it as a radius.
function nearestNeighborDistances(points) {
  const n = points.length;
  const dists = new Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      const dz = points[i][2] - points[j][2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d < dists[i]) dists[i] = d;
      if (d < dists[j]) dists[j] = d;
    }
  }
  return dists;
}

// Least-squares affine fit y ≈ a*x + b — used to calibrate the depth model's arbitrary
// relative-depth units against MediaPipe's landmark z convention (more-negative = closer).
// Self-corrects for both scale AND sign: no assumption is made about which direction the
// depth model's raw values run, the fit finds whatever slope (positive or negative) best
// matches the 478 real landmark z values used as calibration anchors. Degenerate input (all
// xs identical) returns a=0 rather than dividing by zero.
function fitAffine(xs, ys) {
  const n = xs.length;
  let meanX = 0, meanY = 0;
  for (let i = 0; i < n; i++) { meanX += xs[i]; meanY += ys[i]; }
  meanX /= n; meanY /= n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    num += dx * (ys[i] - meanY);
    den += dx * dx;
  }
  const a = den === 0 ? 0 : num / den;
  const b = meanY - a * meanX;
  // r2 (coefficient of determination): how much of the real landmark-z variance the fitted
  // line actually explains — see MIN_DEPTH_CALIB_R2 for why this matters for fused mode.
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < n; i++) {
    const pred = a * xs[i] + b;
    ssRes += (ys[i] - pred) ** 2;
    ssTot += (ys[i] - meanY) ** 2;
  }
  const r2 = ssTot === 0 ? (ssRes === 0 ? 1 : 0) : 1 - ssRes / ssTot;
  return { a, b, r2 };
}

// Runs the monocular depth model (see DEPTH_MODEL_ID) on `view`'s frame and calibrates its
// arbitrary relative-depth units against this view's own measured landmark z values
// (fitAffine) — the 478 landmarks give real (pixel, z) anchor pairs to fit against, so the
// model's raw output doesn't need a known scale or sign convention. Cached on the view object
// (view.__depthCalib) so repeated mode-switches don't re-run inference. Returns null —
// meaning "use the IDW fallback" — if depthEstimator never loaded or inference fails for this
// frame (e.g. a decode error), never throws.
async function calibratedDepthAt(view) {
  if (view.__depthCalib !== undefined) return view.__depthCalib;
  if (!depthEstimator) { view.__depthCalib = null; return null; }
  try {
    const lm = view.faceLandmarks;
    const fw = view.frameCanvas.width, fh = view.frameCanvas.height;

    // Crop to the (padded) face bbox before inference: feeding the model the WHOLE frame
    // (person + background) dilutes its limited internal resolution across the whole frame,
    // leaving very little of it on the small face region once the output is upsampled back —
    // verified empirically (whole-frame inference produced a visibly flat profile, no better
    // than the IDW fallback it's meant to improve on, side-view-checked). Cropping first
    // spends the model's resolution budget entirely on the face.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of lm) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const cropX = Math.max(0, Math.floor((minX - DENSE_BBOX_PADDING) * fw));
    const cropY = Math.max(0, Math.floor((minY - DENSE_BBOX_PADDING) * fh));
    const cropW = Math.min(fw, Math.ceil((maxX + DENSE_BBOX_PADDING) * fw)) - cropX;
    const cropH = Math.min(fh, Math.ceil((maxY + DENSE_BBOX_PADDING) * fh)) - cropY;

    const cropCanvas = document.createElement("canvas");
    cropCanvas.width = cropW; cropCanvas.height = cropH;
    cropCanvas.getContext("2d").drawImage(view.frameCanvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);

    const { predicted_depth } = await depthEstimator(RawImage.fromCanvas(cropCanvas));
    const raw = predicted_depth.data; // resized to cropW x cropH by the pipeline, row-major
    if (raw.length !== cropW * cropH) throw new Error(`depth map ${raw.length} px != crop ${cropW * cropH} px`);

    const xs = [], ys = [];
    for (const p of lm) {
      const px = Math.min(cropW - 1, Math.max(0, Math.round(p.x * fw) - cropX));
      const py = Math.min(cropH - 1, Math.max(0, Math.round(p.y * fh) - cropY));
      xs.push(raw[py * cropW + px]);
      ys.push(p.z);
    }
    const { a, b, r2 } = fitAffine(xs, ys);
    // A poor fit (see MIN_DEPTH_CALIB_R2) means this view's whole depth map is on an unreliable
    // scale/offset — worse than just noisy, it can be a genuine outlier relative to every other
    // view. Reject it here rather than let a bad calibration poison a fused merge.
    view.__depthCalib = r2 < MIN_DEPTH_CALIB_R2 ? null : { raw, cropX, cropY, cropW, cropH, a, b, r2 };
  } catch (err) {
    console.warn("Depth inference failed for this frame — falling back to IDW:", err);
    view.__depthCalib = null;
  }
  return view.__depthCalib;
}

// Wrap an angle to (-π, π].
function wrapPi(x) {
  return x - 2 * Math.PI * Math.round(x / (2 * Math.PI));
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
  const normals = []; // parallel to positions/3 — null entries fall back to isotropic

  if (samples.length === 0) {
    statCount.textContent = "0";
    initSceneAndRender(positions, colors);
    return;
  }

  const halfW = CARD_WIDTH / 2;
  const halfH = CARD_HEIGHT / 2;
  const trig = samples.map((s) => ({ sin: Math.sin(s.angle), cos: Math.cos(s.angle) }));

  // Locate the shoulder line in *this capture's* actual framing (how high/low the subject
  // sits in the webcam view varies with distance, seat height, camera angle...) instead of
  // assuming a fixed world-Y always lands there — see MIN_CARVE_Y. Median of the observed 2D
  // shoulder-midpoint Y across frames, converted to world-Y via the same v<->py relationship
  // used by the carve loop below (v = 0.5 - py/CARD_HEIGHT  =>  py = (0.5 - v) * CARD_HEIGHT).
  const shoulderYs = samples.map((s) => s.shoulderImgY).filter((y) => y != null);
  const carveMinY = shoulderYs.length >= MIN_SHOULDER_SAMPLES
    ? (0.5 - median(shoulderYs)) * CARD_HEIGHT + HEAD_CROP_MARGIN
    : MIN_CARVE_Y; // shoulders never confidently seen this capture — fall back to the fixed default
  // debug probe: what the dynamic head-crop actually resolved to this capture
  window.__carveStats = {
    shoulderSamples: shoulderYs.length,
    medianShoulderImgY: shoulderYs.length ? Number(median(shoulderYs).toFixed(3)) : null,
    carveMinY: Number(carveMinY.toFixed(3)),
    usedFallback: shoulderYs.length < MIN_SHOULDER_SAMPLES,
  };

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
      if (py < carveMinY) continue; // crop out the torso, see carveMinY / MIN_CARVE_Y above
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
        normals.push(estimateVoxelNormal(occ, ix, iy, iz, NX, NY, NZ));
      }
    }
  }

  statCount.textContent = (positions.length / 3).toLocaleString("fr-FR");
  initSceneAndRender(positions, colors, SPLAT_SCALE, null, normals);
}

// =====================
// CANONICAL-MESH RECONSTRUCTION (alternative to the voxel hull)
// =====================
// Geometry comes from a SINGLE reference frame's 478 face landmarks (the most-frontal
// detection) — fixed topology, per-vertex identity stable across frames, so the shape can
// never collapse into a "cone"/ghost the way silhouette-intersection can; it's exactly as
// good (or limited) as that one frame's own face detection, but always face-shaped. Color
// is fused ACROSS frames per vertex: for each of the 478 points, whichever captured frame
// saw it most face-on (smallest landmark z = closest to camera) supplies its color, sampled
// straight from that frame's own full-res image at the landmark's own pixel — exact, no
// misaligned "nearest voxel" guessing. No torso (matches the head-crop decision already
// made for the hull). See TODO in .docs/CAPTURE_DEMO.md ("changement de méthode").
const FACE_MESH_LANDMARK_COUNT = 478;
const FACE_MESH_Z_SIGN = 1; // flip to -1 if the face renders depth-inverted (concave)

// Frames with a usable face mesh, plus the single most-frontal one (smallest |yaw|) used
// as reference geometry for both mesh modes, and its landmark centroid (used to center the
// mesh at the origin before scaling to world units).
function pickReferenceFace(withFace) {
  let reference = withFace[0];
  for (const s of withFace) {
    if (s.faceYaw != null && (reference.faceYaw == null || Math.abs(s.faceYaw) < Math.abs(reference.faceYaw))) {
      reference = s;
    }
  }
  const refLm = reference.faceLandmarks;
  let cx = 0, cy = 0, cz = 0;
  for (const p of refLm) { cx += p.x; cy += p.y; cz += p.z; }
  cx /= refLm.length; cy /= refLm.length; cz /= refLm.length;
  return { reference, refLm, cx, cy, cz };
}

// Rotates a point's local offset (relative to ITS OWN source frame's centroid) into a fixed
// "canonical" orientation — head facing the default camera straight-on — by undoing that
// frame's own measured head rotation (transpose = inverse for an orthogonal rotation matrix).
// Without this, mesh/dense render in whatever raw orientation the reference frame happened to
// be captured at: if the "most frontal" frame wasn't quite dead-on, the default camera view
// isn't quite facing the reconstruction either.
// Used by buildFaceMesh/buildDenseFaceMesh (single reference view only). NOT used by
// buildDenseFusedFaceMesh: applying this independently to every fused view (undoing each
// view's own absolute rotation) turned out less robust than aligning views RELATIVE TO THE
// REFERENCE — MediaPipe's per-frame rotation noise cancels better between two nearby frames
// than against an absolute "identity" target (see the Rdelta comment there, and the
// weird3.png bug in .docs/CAPTURE_DEMO.md).
function toCanonicalOrientation(faceRot, dx, dy, dz) {
  return applyMat3(transpose3(faceRot), dx, dy, dz);
}

function buildFaceMesh() {
  const positions = [];
  const colors = [];

  const withFace = samples.filter((s) => s.faceLandmarks && s.faceLandmarks.length === FACE_MESH_LANDMARK_COUNT && s.faceRot);
  if (withFace.length === 0) {
    statCount.textContent = "0";
    initSceneAndRender(positions, colors, FACE_MESH_SPLAT_SCALE);
    return;
  }

  const { reference, refLm, cx, cy, cz } = pickReferenceFace(withFace);

  // Cache one full-frame ImageData per source frame (color is sampled from whichever frame
  // wins each vertex, so a given frame's data may be reused across many vertices) — a single
  // getImageData per frame instead of one 1x1 readback per landmark (478 of them).
  const imgDataCache = new Map();
  const imgDataFor = (s) => {
    if (!imgDataCache.has(s)) {
      const { width, height } = s.frameCanvas;
      imgDataCache.set(s, s.frameCanvas.getContext("2d").getImageData(0, 0, width, height).data);
    }
    return imgDataCache.get(s);
  };

  for (let i = 0; i < FACE_MESH_LANDMARK_COUNT; i++) {
    // Best (most camera-facing) observation of this exact vertex across all frames.
    let best = withFace[0];
    for (const s of withFace) {
      if (s.faceLandmarks[i].z < best.faceLandmarks[i].z) best = s;
    }
    const bp = best.faceLandmarks[i];
    const fw = best.frameCanvas.width, fh = best.frameCanvas.height;
    const px = Math.min(fw - 1, Math.max(0, Math.round(bp.x * fw)));
    const py = Math.min(fh - 1, Math.max(0, Math.round(bp.y * fh)));
    const ci = (py * fw + px) * 4;
    const pixel = imgDataFor(best);
    colors.push(pixel[ci] / 255, pixel[ci + 1] / 255, pixel[ci + 2] / 255);

    const p = refLm[i];
    const [rx, ry, rz] = toCanonicalOrientation(reference.faceRot, p.x - cx, p.y - cy, p.z - cz);
    positions.push(
      FACE_MESH_FRONT_SIGN * rx * FACE_MESH_SCALE_XY,
      -ry * FACE_MESH_SCALE_XY, // image Y grows downward, world Y grows upward
      FACE_MESH_FRONT_SIGN * FACE_MESH_Z_SIGN * rz * FACE_MESH_SCALE_Z,
    );
  }

  // Per-landmark gaussian radius: a single scalar can't fit both the eyes/brows (densely
  // packed landmarks, need a small radius to stay sharp) and the cheeks (sparse landmarks,
  // need a larger radius to avoid gaps) — see FACE_MESH_SPLAT_SCALE_MIN/MAX/NN_MULT.
  const points3 = [];
  for (let i = 0; i < positions.length; i += 3) points3.push([positions[i], positions[i + 1], positions[i + 2]]);
  const perPointScale = nearestNeighborDistances(points3).map((d) =>
    Math.min(FACE_MESH_SPLAT_SCALE_MAX, Math.max(FACE_MESH_SPLAT_SCALE_MIN, d * FACE_MESH_SPLAT_SCALE_NN_MULT)));
  const perPointNormal = estimatePointCloudNormals(points3, MESH_NORMAL_RADIUS, MESH_NORMAL_MIN_NEIGHBORS);

  statCount.textContent = (positions.length / 3).toLocaleString("fr-FR");
  initSceneAndRender(positions, colors, FACE_MESH_SPLAT_SCALE, perPointScale, perPointNormal);
}

// Depth at (nx, ny) = inverse-distance-weighted average of the DENSE_IDW_NEIGHBORS nearest
// landmarks (a smooth estimate between measured points, not itself measured — see TODO).
// Shared by buildDenseFaceMesh and buildDenseFusedFaceMesh (both sample this per pixel across
// a face bbox). Keeps a running top-K via insertion instead of sorting all ~478 landmarks per
// pixel — only the K nearest are ever needed. topD2/topZ are caller-owned scratch arrays of
// length DENSE_IDW_NEIGHBORS, reused across calls to avoid a per-pixel allocation.
function idwDepthAt(landmarks, nx, ny, topD2, topZ) {
  let count = 0;
  for (let i = 0; i < landmarks.length; i++) {
    const dx = landmarks[i].x - nx, dy = landmarks[i].y - ny;
    const d2 = dx * dx + dy * dy;
    if (count < DENSE_IDW_NEIGHBORS || d2 < topD2[count - 1]) {
      let j = Math.min(count, DENSE_IDW_NEIGHBORS - 1);
      while (j > 0 && topD2[j - 1] > d2) { topD2[j] = topD2[j - 1]; topZ[j] = topZ[j - 1]; j--; }
      topD2[j] = d2; topZ[j] = landmarks[i].z;
      if (count < DENSE_IDW_NEIGHBORS) count++;
    }
  }
  let wSum = 0, zSum = 0;
  for (let k = 0; k < count; k++) {
    const w = 1 / (topD2[k] + 1e-6);
    wSum += w; zSum += w * topZ[k];
  }
  return zSum / wSum;
}

// Denser variant: instead of one point per landmark, sample a fine pixel grid across the
// WHOLE face region of the single reference frame, interpolating depth at each pixel from
// its nearest landmarks (inverse-distance weighting) when the real depth model isn't
// available. See the DENSE_* config comment above for why this is single-frame only, and
// DEPTH_MODEL_ID for the real-depth path (calibratedDepthAt) that replaces the IDW estimate
// whenever the model loaded successfully.
async function buildDenseFaceMesh() {
  const positions = [];
  const colors = [];

  const withFace = samples.filter((s) => s.faceLandmarks && s.faceLandmarks.length === FACE_MESH_LANDMARK_COUNT && s.faceRot);
  if (withFace.length === 0) {
    statCount.textContent = "0";
    initSceneAndRender(positions, colors, DENSE_SPLAT_SCALE);
    return;
  }

  const { reference, refLm, cx, cy, cz } = pickReferenceFace(withFace);
  const fw = reference.frameCanvas.width, fh = reference.frameCanvas.height;
  const imgData = reference.frameCanvas.getContext("2d").getImageData(0, 0, fw, fh).data;
  const { maskData, maskW, maskH } = reference;

  const depthCalib = await calibratedDepthAt(reference);
  if (reconMode !== "dense") return; // stale — user switched view while inference was running

  // debug probe: whether the real depth model was used for this render, and the calibration
  // fit against the 478 landmarks (a/b near 0 or a degenerate fit signals something's off).
  window.__depthStats = depthCalib
    ? { source: "depth-model", a: depthCalib.a, b: depthCalib.b, r2: depthCalib.r2 }
    : { source: depthEstimator ? "idw (inference failed)" : "idw (model unavailable)" };

  // Landmark bounding box (normalized image space), padded, clamped to the frame.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of refLm) {
    if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
  }
  const pxMin = Math.max(0, Math.floor((minX - DENSE_BBOX_PADDING) * fw));
  const pxMax = Math.min(fw - 1, Math.ceil((maxX + DENSE_BBOX_PADDING) * fw));
  const pyMin = Math.max(0, Math.floor((minY - DENSE_BBOX_PADDING) * fh));
  const pyMax = Math.min(fh - 1, Math.ceil((maxY + DENSE_BBOX_PADDING) * fh));

  const topD2 = new Array(DENSE_IDW_NEIGHBORS), topZ = new Array(DENSE_IDW_NEIGHBORS);
  for (let py = pyMin; py <= pyMax; py += DENSE_GRID_STEP_PX) {
    const ny = py / fh;
    const my = Math.min(maskH - 1, (ny * maskH) | 0);
    for (let px = pxMin; px <= pxMax; px += DENSE_GRID_STEP_PX) {
      const nx = px / fw;
      const mx = Math.min(maskW - 1, (nx * maskW) | 0);
      if (maskData[my * maskW + mx] <= FOREGROUND_THRESHOLD) continue; // outside the subject

      const cpx = px - (depthCalib ? depthCalib.cropX : 0), cpy = py - (depthCalib ? depthCalib.cropY : 0);
      const inCrop = depthCalib && cpx >= 0 && cpx < depthCalib.cropW && cpy >= 0 && cpy < depthCalib.cropH;
      // idwDepthAt also fills topD2 (ascending) as a side effect — topD2[0] afterward is the
      // squared distance to the nearest landmark, reused below to gate the depth model (see
      // DEPTH_MODEL_MAX_LANDMARK_DIST) instead of computing that distance a second time.
      const idwZ = idwDepthAt(refLm, nx, ny, topD2, topZ);
      const trusted = inCrop && Math.sqrt(topD2[0]) <= DEPTH_MODEL_MAX_LANDMARK_DIST;
      const z = trusted ? depthCalib.a * depthCalib.raw[cpy * depthCalib.cropW + cpx] + depthCalib.b : idwZ;
      const [rx, ry, rz] = toCanonicalOrientation(reference.faceRot, nx - cx, ny - cy, z - cz);
      const ci = (py * fw + px) * 4;
      colors.push(imgData[ci] / 255, imgData[ci + 1] / 255, imgData[ci + 2] / 255);
      positions.push(
        FACE_MESH_FRONT_SIGN * rx * FACE_MESH_SCALE_XY,
        -ry * FACE_MESH_SCALE_XY,
        FACE_MESH_FRONT_SIGN * FACE_MESH_Z_SIGN * rz * FACE_MESH_SCALE_Z,
      );
    }
  }

  const points3 = [];
  for (let i = 0; i < positions.length; i += 3) points3.push([positions[i], positions[i + 1], positions[i + 2]]);
  const perPointNormal = estimatePointCloudNormals(points3, DENSE_NORMAL_RADIUS, DENSE_NORMAL_MIN_NEIGHBORS);

  statCount.textContent = (positions.length / 3).toLocaleString("fr-FR");
  initSceneAndRender(positions, colors, DENSE_SPLAT_SCALE, null, perPointNormal);
}

// Multi-frame variant of the dense mode: fuses a few extra views (spread across the
// captured yaw range) into the reference frame's local space, instead of using only the
// single most-frontal frame. See the DENSE_FUSION_* config comment above for the alignment
// approach and its trade-offs. Depth per view comes from the real depth model when available
// (calibratedDepthAt, one calibrated inference per view — see DEPTH_MODEL_ID), falling back
// to landmark IDW per view otherwise, same fallback rule as the single-view dense mode.
async function buildDenseFusedFaceMesh() {
  const withFace = samples.filter((s) =>
    s.faceLandmarks && s.faceLandmarks.length === FACE_MESH_LANDMARK_COUNT && s.angle != null && s.faceRot);
  if (withFace.length === 0) {
    statCount.textContent = "0";
    initSceneAndRender([], [], DENSE_SPLAT_SCALE);
    return;
  }

  const { reference } = pickReferenceFace(withFace);

  // Per-view alignment for fusion uses each view's rotation RELATIVE TO THE REFERENCE
  // (Rdelta below), not the absolute toCanonicalOrientation used by the single-view mesh/
  // dense modes. The two are equivalent in theory (undoing a view's own rotation should land
  // it in the same fixed frame regardless of path), but not in practice: MediaPipe's
  // facialTransformationMatrix carries per-frame noise/bias, and that bias cancels out well
  // when comparing two nearby frames (view vs. reference) but not when each view is unrotated
  // independently to an absolute "identity" orientation — a real user capture (weird3.png)
  // still showed disconnected planes after switching to the absolute approach for the 180°
  // front-facing fix, even after the wing-crop and yaw-delta fixes below. Reverting fusion's
  // alignment to relative-to-reference (as it was before that fix, when a side-view retest
  // had already confirmed clean, non-streaky geometry) and applying the FACE_MESH_FRONT_SIGN
  // 180° flip as a final step restores that known-good alignment while still facing forward.
  const refRotT = transpose3(reference.faceRot);

  // Extra views spread evenly across the captured yaw range (beyond the reference), to fill
  // in what a single frontal frame can't see — excluding candidates too far from the
  // reference (see DENSE_FUSION_MAX_YAW_DELTA) so an unreliable near-profile detection can't
  // flare out into disconnected "wings" once fused in.
  const candidates = withFace.filter((s) => Math.abs(s.angle - reference.angle) <= DENSE_FUSION_MAX_YAW_DELTA);
  const sorted = candidates.sort((a, b) => a.angle - b.angle);
  const n = Math.min(DENSE_FUSION_MAX_VIEWS, sorted.length);
  const views = [reference];
  for (let i = 0; i < n; i++) {
    const idx = n > 1 ? Math.round((i * (sorted.length - 1)) / (n - 1)) : 0;
    const s = sorted[idx];
    if (!views.includes(s)) views.push(s);
  }

  // Reject any view whose measured Rdelta rotation disagrees too much with the reference (see
  // DENSE_FUSION_MAX_ALIGN_ERR/viewAlignError) — a view that passes DENSE_FUSION_MAX_YAW_DELTA
  // can still have an unreliable individual head-pose estimate; this catches that case
  // directly instead of hoping the yaw-delta cutoff alone is a good enough proxy for it.
  for (let i = views.length - 1; i >= 0; i--) {
    if (views[i] === reference) continue;
    if (viewAlignError(views[i], reference, refRotT) > DENSE_FUSION_MAX_ALIGN_ERR) views.splice(i, 1);
  }

  // Run (and cache) depth-model inference for every fused view in parallel — each view's
  // calibration is independent (its own crop, its own landmark anchors), so there's no
  // ordering dependency between them. Falls back to per-view IDW below wherever a view's
  // calibration failed or the model never loaded (calibratedDepthAt never throws).
  await Promise.all(views.map((v) => calibratedDepthAt(v)));
  if (reconMode !== "fused") return; // stale — user switched view while inference was running

  // Merge grid: points from different views that land in the same cell are averaged
  // (position + color) instead of left as separate overlapping splats. Also accumulates
  // sum-of-squares per color channel — free byproduct used to compute each cell's color
  // VARIANCE (see window.__fusionColorVariance below): if two views actually see the same
  // physical surface point, they should sample close to the same real color; high variance in
  // a multi-view cell means they don't agree, a photometric sign of the same kind of
  // misalignment the geometric ghosting diagnostic looks for, independent of it.
  const cell = DENSE_FUSION_MERGE_CELL;
  const grid = new Map();
  const addPoint = (x, y, z, r, g, b) => {
    const key = `${Math.round(x / cell)},${Math.round(y / cell)},${Math.round(z / cell)}`;
    let e = grid.get(key);
    if (!e) { e = { x: 0, y: 0, z: 0, r: 0, g: 0, b: 0, r2: 0, g2: 0, b2: 0, n: 0 }; grid.set(key, e); }
    e.x += x; e.y += y; e.z += z; e.r += r; e.g += g; e.b += b;
    e.r2 += r * r; e.g2 += g * g; e.b2 += b * b; e.n++;
  };

  const topD2 = new Array(DENSE_IDW_NEIGHBORS), topZ = new Array(DENSE_IDW_NEIGHBORS);
  // DEBUG: window.__fusedDebugColorByView=true colors every point by its source view index
  // instead of sampled color, to see which view produces a given region in the fused result.
  const DEBUG_VIEW_COLORS = [[1, 0, 0], [0, 1, 0], [0, 0.4, 1], [1, 0.85, 0], [1, 0, 1]];
  let __viewIdx = -1;
  for (const view of views) {
    __viewIdx++;
    const isReference = view === reference;
    const lm = view.faceLandmarks;
    const fw = view.frameCanvas.width, fh = view.frameCanvas.height;
    const imgData = view.frameCanvas.getContext("2d").getImageData(0, 0, fw, fh).data;
    const { maskData, maskW, maskH } = view;

    // This view's own landmark centroid — points are computed relative to it, then (for
    // non-reference views) rotated into the reference's own frame via Rdelta below.
    let lcx = 0, lcy = 0, lcz = 0;
    for (const p of lm) { lcx += p.x; lcy += p.y; lcz += p.z; }
    lcx /= lm.length; lcy /= lm.length; lcz /= lm.length;
    // Rdelta maps this view's local point into the reference's own (raw, un-rotated) frame —
    // view.faceRot @ transpose(reference.faceRot), the composition order previously
    // validated via a numeric diagnostic (see .docs/CAPTURE_DEMO.md) — so all non-reference
    // views land in one consistent frame. The reference itself uses its points as-is (no
    // rotation applied), matching how this alignment worked before the 180° fix.
    const Rdelta = isReference ? null : multiply3(view.faceRot, refRotT);
    const depthCalib = view.__depthCalib; // resolved by the Promise.all above; may be null

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of lm) {
      if (p.x < minX) minX = p.x; if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y; if (p.y > maxY) maxY = p.y;
    }
    const pxMin = Math.max(0, Math.floor((minX - DENSE_BBOX_PADDING) * fw));
    const pxMax = Math.min(fw - 1, Math.ceil((maxX + DENSE_BBOX_PADDING) * fw));
    const pyMin = Math.max(0, Math.floor((minY - DENSE_BBOX_PADDING) * fh));
    const pyMax = Math.min(fh - 1, Math.ceil((maxY + DENSE_BBOX_PADDING) * fh));
    // Outer-band cutoff for non-reference views (see DENSE_FUSION_WING_INNER_FRAC) — the
    // half-width on each side of this view's own centroid, past which points are kept.
    const halfWidthL = lcx - minX, halfWidthR = maxX - lcx;
    const wingXMin = lcx - halfWidthL * DENSE_FUSION_WING_INNER_FRAC;
    const wingXMax = lcx + halfWidthR * DENSE_FUSION_WING_INNER_FRAC;

    const step = isReference ? DENSE_GRID_STEP_PX : DENSE_GRID_STEP_PX * DENSE_FUSION_STEP_MULT;

    for (let py = pyMin; py <= pyMax; py += step) {
      const ny = py / fh;
      const my = Math.min(maskH - 1, (ny * maskH) | 0);
      for (let px = pxMin; px <= pxMax; px += step) {
        const nx = px / fw;
        if (!isReference && nx > wingXMin && nx < wingXMax) continue; // central card already covered by the reference
        const mx = Math.min(maskW - 1, (nx * maskW) | 0);
        if (maskData[my * maskW + mx] <= FOREGROUND_THRESHOLD) continue; // outside the subject

        const cpx = px - (depthCalib ? depthCalib.cropX : 0), cpy = py - (depthCalib ? depthCalib.cropY : 0);
        const inCrop = depthCalib && cpx >= 0 && cpx < depthCalib.cropW && cpy >= 0 && cpy < depthCalib.cropH;
        // See DEPTH_MODEL_MAX_LANDMARK_DIST — tested as a fix for the fused-mode "ghost
        // surface" artifact (a distinct floating point layer, side-view-visible), but measured
        // to have NO effect on it (see .docs/CAPTURE_DEMO.md) — kept anyway since discarding
        // unanchored depth-model extrapolation is correct on its own merits, just not the fix.
        const idwZ = idwDepthAt(lm, nx, ny, topD2, topZ);
        const trusted = inCrop && Math.sqrt(topD2[0]) <= DEPTH_MODEL_MAX_LANDMARK_DIST;
        const z = trusted ? depthCalib.a * depthCalib.raw[cpy * depthCalib.cropW + cpx] + depthCalib.b : idwZ;

        // Point relative to this view's own centroid: rotated into the reference's own frame
        // via Rdelta for non-reference views, used as-is for the reference itself.
        const dx0 = nx - lcx, dy0 = ny - lcy, dz0 = z - lcz;
        const [rx, ry, rz] = isReference ? [dx0, dy0, dz0] : applyMat3(Rdelta, dx0, dy0, dz0);

        const ci = (py * fw + px) * 4;
        const [dr, dg, db] = window.__fusedDebugColorByView
          ? DEBUG_VIEW_COLORS[__viewIdx % DEBUG_VIEW_COLORS.length]
          : [imgData[ci] / 255, imgData[ci + 1] / 255, imgData[ci + 2] / 255];
        addPoint(
          FACE_MESH_FRONT_SIGN * rx * FACE_MESH_SCALE_XY,
          -ry * FACE_MESH_SCALE_XY,
          FACE_MESH_FRONT_SIGN * FACE_MESH_Z_SIGN * rz * FACE_MESH_SCALE_Z,
          dr, dg, db,
        );
      }
    }
  }

  const positions = [];
  const colors = [];
  // Photometric consistency: for cells that merged more than one view's contribution, how much
  // do their sampled colors actually disagree (per-channel variance, averaged over channels
  // then over cells) — see the addPoint comment above. Single-view cells are excluded: with n=1
  // there's nothing to disagree with, and including them would just dilute the signal toward 0
  // regardless of how bad the multi-view cells are.
  let colorVarSum = 0, multiViewCells = 0;
  for (const e of grid.values()) {
    positions.push(e.x / e.n, e.y / e.n, e.z / e.n);
    colors.push(e.r / e.n, e.g / e.n, e.b / e.n);
    if (e.n > 1) {
      const varR = e.r2 / e.n - (e.r / e.n) ** 2, varG = e.g2 / e.n - (e.g / e.n) ** 2, varB = e.b2 / e.n - (e.b / e.n) ** 2;
      colorVarSum += (varR + varG + varB) / 3;
      multiViewCells++;
    }
  }
  window.__fusionColorVariance = {
    meanMultiViewVariance: multiViewCells > 0 ? colorVarSum / multiViewCells : null,
    multiViewCells,
    totalCells: grid.size,
  };

  // debug probe: which views actually got fused, and how far each was from the reference —
  // check this if the fused mode ever looks warped ("wings"/streaks), it names the suspect view.
  const R2D = 180 / Math.PI;
  window.__fusionStats = {
    referenceDeg: Math.round(reference.angle * R2D),
    candidatePool: candidates.length,
    views: views.map((v) => ({
      deg: Math.round(v.angle * R2D),
      deltaDeg: Math.round((v.angle - reference.angle) * R2D),
      isReference: v === reference,
      depthSource: v.__depthCalib ? "depth-model" : "idw",
      calibA: v.__depthCalib ? v.__depthCalib.a : null,
      calibB: v.__depthCalib ? v.__depthCalib.b : null,
      calibR2: v.__depthCalib ? v.__depthCalib.r2 : null,
      alignErr: v === reference ? 0 : viewAlignError(v, reference, refRotT),
    })),
  };

  const points3 = [];
  for (let i = 0; i < positions.length; i += 3) points3.push([positions[i], positions[i + 1], positions[i + 2]]);
  const perPointNormal = estimatePointCloudNormals(points3, DENSE_NORMAL_RADIUS, DENSE_NORMAL_MIN_NEIGHBORS);

  statCount.textContent = (positions.length / 3).toLocaleString("fr-FR");
  initSceneAndRender(positions, colors, DENSE_SPLAT_SCALE, null, perPointNormal);
}

// hull cloud (object-space x,y,z + r,g,b 0..1) — same world frame as the COLMAP poses
let lastCloud = null;

// Voxel-index → world-coordinate helpers (grid is centered on the origin).
function px3(ix, halfW, NX) { return -halfW + ((ix + 0.5) / NX) * (2 * halfW); }
function py3(iy, halfH, NY) { return -halfH + ((iy + 0.5) / NY) * (2 * halfH); }
function pz3(iz, halfW, NZ) { return -halfW + ((iz + 0.5) / NZ) * (2 * halfW); }

function occAt(occ, ix, iy, iz, NX, NY, NZ) {
  if (ix < 0 || ix >= NX || iy < 0 || iy >= NY || iz < 0 || iz >= NZ) return 0;
  return occ[(iz * NY + iy) * NX + ix];
}
// Outward surface normal at a kept voxel, from the occupancy gradient of its 6 face
// neighbors (out-of-bounds treated as empty) — used to orient anisotropic gaussians (see
// ANISO_* constants) instead of rendering every hull voxel as an isotropic sphere. Points
// from occupied toward empty — e.g. if the -X neighbor is occupied and +X is empty, the
// surface faces +X. Returns null when the gradient is ~zero (voxel is symmetric on all 3
// axes — fully interior, or an isolated speck with no directional info); the caller then
// falls back to an isotropic splat for that point.
function estimateVoxelNormal(occ, ix, iy, iz, NX, NY, NZ) {
  const nx = occAt(occ, ix - 1, iy, iz, NX, NY, NZ) - occAt(occ, ix + 1, iy, iz, NX, NY, NZ);
  const ny = occAt(occ, ix, iy - 1, iz, NX, NY, NZ) - occAt(occ, ix, iy + 1, iz, NX, NY, NZ);
  const nz = occAt(occ, ix, iy, iz - 1, NX, NY, NZ) - occAt(occ, ix, iy, iz + 1, NX, NY, NZ);
  const mag = Math.sqrt(nx * nx + ny * ny + nz * nz);
  if (mag < 1e-6) return null;
  return [nx / mag, ny / mag, nz / mag];
}

// Point-cloud counterpart to estimateVoxelNormal, used by mesh/dense/fused (see MESH_NORMAL_*/
// DENSE_NORMAL_* above) since those modes have no voxel grid to take an occupancy gradient
// from. For each point, fits a local plane via PCA over its neighbors within `radius` (a
// uniform spatial hash keeps this near-linear instead of the O(n^2) brute force used by
// nearestNeighborDistances — dense/fused can have thousands of points, unlike mesh's 478) and
// returns that plane's normal (the covariance matrix's smallest-eigenvalue eigenvector).
// Points with fewer than `minNeighbors` within radius get null (too sparse for a reliable
// fit) and the caller falls back to an isotropic splat. Returns one entry per point (null or
// unit [x,y,z]), same convention as estimateVoxelNormal.
function estimatePointCloudNormals(points, radius, minNeighbors) {
  const cellSize = radius / 1.5; // small enough that a 3x3x3 cell search can't miss a
                                  // same-radius neighbor, without over-fragmenting into tiny buckets
  const grid = new Map();
  const cellOf = (p) => `${Math.floor(p[0] / cellSize)},${Math.floor(p[1] / cellSize)},${Math.floor(p[2] / cellSize)}`;
  for (let i = 0; i < points.length; i++) {
    const key = cellOf(points[i]);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  }

  const r2 = radius * radius;
  const normals = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    const [px, py, pz] = points[i];
    const cx = Math.floor(px / cellSize), cy = Math.floor(py / cellSize), cz = Math.floor(pz / cellSize);
    const nbrs = [];
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (let dz = -1; dz <= 1; dz++) {
      const bucket = grid.get(`${cx + dx},${cy + dy},${cz + dz}`);
      if (!bucket) continue;
      for (const j of bucket) {
        if (j === i) continue;
        const ddx = points[j][0] - px, ddy = points[j][1] - py, ddz = points[j][2] - pz;
        if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) nbrs.push(j);
      }
    }
    if (nbrs.length < minNeighbors) { normals[i] = null; continue; }

    let mx = px, my = py, mz = pz;
    for (const j of nbrs) { mx += points[j][0]; my += points[j][1]; mz += points[j][2]; }
    const n = nbrs.length + 1;
    mx /= n; my /= n; mz /= n;

    const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    const accum = (x, y, z) => {
      const dx = x - mx, dy = y - my, dz = z - mz;
      C[0][0] += dx * dx; C[0][1] += dx * dy; C[0][2] += dx * dz;
      C[1][1] += dy * dy; C[1][2] += dy * dz;
      C[2][2] += dz * dz;
    };
    accum(px, py, pz);
    for (const j of nbrs) accum(points[j][0], points[j][1], points[j][2]);
    C[1][0] = C[0][1]; C[2][0] = C[0][2]; C[2][1] = C[1][2];

    normals[i] = smallestEigenvector3(C);
  }
  return normals;
}

// Smallest-eigenvalue eigenvector of a symmetric 3x3 matrix, closed-form (Smith 1961) rather
// than power iteration: a starting vector can land exactly orthogonal to the target eigenvector
// under symmetric point distributions (verified with a 45deg-tilted-plane test case) and then
// never converge to it — real face geometry has enough bilateral symmetry that this isn't a
// contrived edge case, so a deterministic closed form is used instead of hoping a seed avoids it.
function smallestEigenvector3(C) {
  const [a00, a01, a02] = C[0], [a10, a11, a12] = C[1], [, , a22] = C[2];
  const p1 = a01 * a01 + a02 * a02 + a12 * a12;
  let eig;
  if (p1 < 1e-14) {
    eig = Math.min(a00, a11, a22);
  } else {
    const q = (a00 + a11 + a22) / 3;
    const p2 = (a00 - q) ** 2 + (a11 - q) ** 2 + (a22 - q) ** 2 + 2 * p1;
    const p = Math.sqrt(p2 / 6);
    const b00 = (a00 - q) / p, b01 = a01 / p, b02 = a02 / p;
    const b11 = (a11 - q) / p, b12 = a12 / p, b22 = (a22 - q) / p;
    const detB = b00 * (b11 * b22 - b12 * b12) - b01 * (b01 * b22 - b12 * b02) + b02 * (b01 * b12 - b11 * b02);
    const r = Math.min(1, Math.max(-1, detB / 2));
    const phi = Math.acos(r) / 3;
    const eig1 = q + 2 * p * Math.cos(phi);
    const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
    const eig2 = 3 * q - eig1 - eig3;
    eig = Math.min(eig1, eig2, eig3);
  }
  const m00 = a00 - eig, m11 = a11 - eig, m22 = a22 - eig;
  const rows = [[m00, a01, a02], [a10, m11, a12], [a02, a12, m22]];
  const cross = (u, v) => [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  let best = null, bestLen = -1;
  for (let i = 0; i < 3; i++) {
    const v = cross(rows[i], rows[(i + 1) % 3]);
    const len = Math.hypot(v[0], v[1], v[2]);
    if (len > bestLen) { bestLen = len; best = v; }
  }
  if (bestLen < 1e-12) return null; // degenerate (near-zero covariance) — no reliable normal
  return [best[0] / bestLen, best[1] / bestLen, best[2] / bestLen];
}

// =====================
// THREE.JS RENDER
// =====================
let renderer, scene, camera, points, spark, splatMesh;
let yaw = 0, pitch = 0.1, radius = 3.2;
let isDragging = false, lastX = 0, lastY = 0;

function initSceneAndRender(positions, colors, splatScale = SPLAT_SCALE, perPointScale = null, perPointNormal = null) {
  // Safe to call repeatedly (e.g. switching between hull/mesh view on already-captured
  // samples): tear down any previous scene first instead of stacking canvases/render loops.
  teardownScene();

  // debug probe: the raw point cloud actually rendered, for external tooling (test/) that
  // needs to score reconstruction quality (e.g. a geometric-coherence metric via connected
  // components) instead of judging a screenshot by eye. Exposed for every mode, not just fused.
  // Also doubles as the payload for the "Sauver (JSON)" export (see saveCloudJson): it's
  // exactly the set of inputs initSceneAndRender needs, so reloading a saved file is just
  // calling this same function again — no separate serialization of the derived per-splat
  // scale/quaternion (see the ANISO_*/estimatePointCloudNormals comments) is needed.
  window.__lastPoints = { mode: reconMode, positions, colors, splatScale, perPointScale, perPointNormal };
  lastCloud = { positions, colors }; // kept for the COLMAP points3D.txt init — now follows
                                      // whichever mode is on screen, not hull only (previous
                                      // behavior: buildPointCloud was the only place setting it)

  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 1000);

  // Real gaussian splatting via SparkJS: one gaussian per point (same stack as the portfolio
  // background). Falls back to THREE.Points if SparkJS can't initialize (e.g. no WebGL2).
  // perPointScale (one entry per point, e.g. mesh mode's per-landmark density-adaptive
  // radius — see FACE_MESH_SPLAT_SCALE_MIN/MAX) overrides the shared isotropic radius.
  // perPointNormal (one unit-vector-or-null entry per point, hull mode only — see
  // estimateVoxelNormal/ANISO_*) flattens a splat into a disc aligned with that normal
  // instead of a sphere; a null entry (no usable normal for that point) stays isotropic.
  try {
    spark = new SparkRenderer({ renderer });
    scene.add(spark);

    const center = new THREE.Vector3();
    const scales = new THREE.Vector3(splatScale, splatScale, splatScale);
    const quaternion = new THREE.Quaternion();
    const color = new THREE.Color();
    const upAxis = new THREE.Vector3(0, 0, 1);
    const normalVec = new THREE.Vector3();

    splatMesh = new SplatMesh({
      constructSplats: (splats) => {
        for (let i = 0; i < positions.length; i += 3) {
          center.set(positions[i], positions[i + 1], positions[i + 2]);
          color.setRGB(colors[i], colors[i + 1], colors[i + 2]);
          const base = perPointScale ? perPointScale[i / 3] : splatScale;
          const n = perPointNormal ? perPointNormal[i / 3] : null;
          if (n) {
            normalVec.set(n[0], n[1], n[2]);
            quaternion.setFromUnitVectors(upAxis, normalVec);
            scales.set(base * ANISO_TANGENT_SCALE_MULT, base * ANISO_TANGENT_SCALE_MULT, base * ANISO_NORMAL_SCALE_MULT);
          } else {
            quaternion.identity();
            scales.set(base, base, base);
          }
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
    const src = s.angleSrc === "body" ? "<b>torse</b>"
      : s.angleSrc === "face" ? "<b>face</b>"
      : "<span class='flow'>flux</span>";
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

// Seed the sparse cloud a 3DGS trainer initialises from, straight out of the hull.
// Hull voxels live in the same object-centred world frame as the exported poses, so
// their coordinates drop in as-is; colours are the per-voxel RGB (0..255 for COLMAP).
function buildPoints3DTxt() {
  let txt =
    "# 3D point list with one line of data per point:\n" +
    "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
  if (!lastCloud || lastCloud.positions.length === 0) return txt;
  const { positions, colors } = lastCloud;
  const n = positions.length / 3;
  txt += `# Number of points: ${n}, mean track length: 0\n`;
  const lines = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    const x = positions[p].toFixed(6), y = positions[p + 1].toFixed(6), z = positions[p + 2].toFixed(6);
    const r = Math.round(colors[p] * 255);
    const g = Math.round(colors[p + 1] * 255);
    const b = Math.round(colors[p + 2] * 255);
    lines[i] = `${i + 1} ${x} ${y} ${z} ${r} ${g} ${b} 0`; // ERROR=0, empty track
  }
  return txt + lines.join("\n") + "\n";
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
    files.push({ name: "points3D.txt", data: enc.encode(buildPoints3DTxt()) });

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

// Save/reload the CURRENTLY RENDERED cloud (whichever mode is on screen) as a standalone JSON
// file — a lighter round-trip than COLMAP (no frames/poses, just what's needed to re-render):
// exactly window.__lastPoints' fields, which are exactly initSceneAndRender's own inputs. This
// means reloading never re-derives per-splat scale/quaternion itself — it just calls
// initSceneAndRender again with the saved arrays, reusing the same anisotropic-gaussian math
// (see ANISO_*/estimatePointCloudNormals) instead of duplicating it in a separate import path.
function saveCloudJson() {
  const pts = window.__lastPoints;
  if (!pts || !pts.positions.length) return;
  const json = JSON.stringify(pts);
  const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = `scan_${pts.mode}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

async function loadCloudJson(file) {
  let pts;
  try {
    pts = JSON.parse(await file.text());
  } catch (err) {
    setError("Fichier JSON invalide : " + err.message);
    return;
  }
  if (!Array.isArray(pts.positions) || !Array.isArray(pts.colors)) {
    setError("Fichier JSON: champs positions/colors manquants ou invalides.");
    return;
  }
  // Loading a saved cloud replaces the live capture entirely — the mode-cycle button and any
  // further captures no longer make sense against it, so it's disabled rather than left to
  // silently re-render over the loaded points on the next click.
  modeBtn.disabled = true;
  // reconMode must reflect the LOADED file's mode, not whatever it happened to be before —
  // initSceneAndRender reads this module-level variable to tag window.__lastPoints/lastCloud,
  // so leaving it stale would mislabel the reloaded cloud (verified: without this line, a
  // reloaded "fused" export was tagged "hull" — the default — since nothing had set it).
  reconMode = pts.mode ?? reconMode;
  statCount.textContent = (pts.positions.length / 3).toLocaleString("fr-FR");
  panelEl.classList.remove("hidden");
  container.classList.remove("hidden");
  stageEl.classList.add("hidden");
  initSceneAndRender(pts.positions, pts.colors, pts.splatScale ?? SPLAT_SCALE, pts.perPointScale ?? null, pts.perPointNormal ?? null);
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

// Tear down the current Three.js/SparkJS scene (if any) so a new one can be built without
// stacking canvases or duplicate render loops. Used both on restart and when switching
// between reconstruction modes (hull/mesh) on already-captured samples.
function teardownScene() {
  if (!renderer) return;
  renderer.setAnimationLoop(null);
  if (splatMesh) { scene?.remove(splatMesh); splatMesh.dispose?.(); splatMesh = null; }
  if (spark) { scene?.remove(spark); spark.dispose?.(); spark = null; }
  renderer.dispose();
  container.removeChild(renderer.domElement);
  renderer = null;
  points = null;
}

// =====================
// RESTART
// =====================
function reset() {
  stopWebcam();
  teardownScene();
  samples = [];
  reconMode = RECON_MODE_DEFAULT;
  yaw = 0; pitch = 0.1; radius = 3.2;

  container.classList.add("hidden");
  panelEl.classList.add("hidden");
  hintEl.classList.add("hidden");
  filmstripEl.classList.add("hidden");
  filmstripEl.innerHTML = "";
  framesBtn.textContent = "Voir les frames (debug)";
  stageEl.classList.remove("hidden");
  progressRing.classList.add("hidden");
  debugEl.classList.add("hidden");
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

modeBtn.addEventListener("click", () => {
  reconMode = nextReconMode();
  renderReconstruction();
  updateModeButton();
});

exportBtn.addEventListener("click", exportColmap);
saveJsonBtn.addEventListener("click", saveCloudJson);
loadJsonBtn.addEventListener("click", () => loadJsonInput.click());
loadJsonInput.addEventListener("change", () => {
  const file = loadJsonInput.files[0];
  loadJsonInput.value = ""; // allow re-selecting the same file name later
  if (file) loadCloudJson(file);
});

// Orbit controls + resize are attached once — the container/camera persist
// across capture sessions, so re-attaching on every restart would stack
// duplicate listeners and make dragging increasingly oversensitive.
attachOrbitControls();
window.addEventListener("resize", onResize);

// Reflect the configured capture duration in the initial copy (the HTML carries
// a neutral first-paint fallback for the brief moment before this runs).
setInstructions(defaultInstructions());

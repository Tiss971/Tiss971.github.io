import * as THREE from 'three';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { refineDepth } from './depth-refinement.js';
import { moveCamera, anchorProjection } from './camera-motion.js';

const smooth = t => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };

export function checkRendering() {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('webgl2');
  if (!context) throw new Error('Le rendu 3D nécessite WebGL 2. Essaie un navigateur récent avec l’accélération graphique activée.');
  context.getExtension('WEBGL_lose_context')?.loseContext();
}

// Depth Anything predicts relative inverse depth. Choose a bounded scene scale;
// this is a surface reconstruction, not metric depth or multi-view training.
export async function createEffect(frame, prediction, signal) {
  signal.throwIfAborted();
  const start = performance.now();
  const { depth, width: dw, height: dh } = prediction;
  if (!depth.length || !dw || !dh) throw new Error('Le modèle n’a pas fourni de profondeur exploitable.');
  const sorted = Array.from(depth).filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) throw new Error('La carte de profondeur contient des valeurs invalides.');
  const low = sorted[Math.floor(sorted.length * .02)];
  const high = sorted[Math.floor(sorted.length * .98)];
  // Inference resolution and appearance resolution are independent. Downscaling
  // RGB to the depth grid discarded detail even for infinitesimal camera moves.
  const rgbScale = Math.min(1, Math.sqrt(1_000_000 / (frame.width * frame.height)));
  const w = Math.max(2, Math.floor(frame.width * rgbScale));
  const h = Math.max(2, Math.floor(frame.height * rgbScale));
  const rgb = document.createElement('canvas'); rgb.width = w; rgb.height = h;
  const rgbContext = rgb.getContext('2d', { willReadFrequently: true });
  rgbContext.drawImage(frame, 0, 0, w, h);
  const rgba = rgbContext.getImageData(0, 0, w, h).data;
  const refinement = await refineDepth(depth, dw, dh, rgba, w, h, signal);
  signal.throwIfAborted();
  const distances = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const raw = refinement.values[y * w + x];
    const n = Number.isFinite(raw) && high > low ? Math.max(0, Math.min(1, (raw - low) / (high - low))) : .5;
    distances[y * w + x] = 1 / (.35 + .65 * n);
  }
  const aspect = frame.width / frame.height;
  const fov = 50, tangent = Math.tan(THREE.MathUtils.degToRad(fov / 2));
  let renderer, spark, mesh, viewTarget, resolveMaterial, resolveGeometry;
  let disposed = false;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(fov, aspect, .1, 10);
  const output = document.createElement('canvas');
  output.width = frame.width; output.height = frame.height;
  const ctx = output.getContext('2d');
  const dispose = () => {
    if (disposed) return;
    disposed = true;
    mesh?.dispose(); spark?.dispose(); viewTarget?.dispose();
    resolveMaterial?.dispose(); resolveGeometry?.dispose();
    renderer?.dispose(); renderer?.forceContextLoss();
    scene.clear(); output.width = output.height = 1;
  };
  try {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: false, preserveDrawingBuffer: true });
    renderer.setPixelRatio(1);
    renderer.setSize(frame.width, frame.height, false);
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    viewTarget = new THREE.WebGLRenderTarget(frame.width, frame.height, { depthBuffer: false });
    // Gaussian coverage is fractional even on a fully reconstructed surface.
    // Resolve premultiplied colour to an opaque moved view instead of exposing
    // the unmoved source image through every semi-transparent splat.
    resolveMaterial = new THREE.ShaderMaterial({
      uniforms: {
        view: { value: viewTarget.texture },
        texel: { value: new THREE.Vector2(1 / frame.width, 1 / frame.height) },
      },
      vertexShader: 'varying vec2 uvView; void main() { uvView = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
      fragmentShader: `
        uniform sampler2D view;
        uniform vec2 texel;
        varying vec2 uvView;
        void main() {
          vec4 sampleView = texture2D(view, uvView);
          if (sampleView.a < 0.02) {
            // Close only small gaps using neighbours from this same moved view.
            // No unmoved RGB layer or invented hidden geometry is composited.
            for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
              vec4 neighbour = texture2D(view, uvView + vec2(float(x), float(y)) * texel);
              if (neighbour.a > sampleView.a) sampleView = neighbour;
            }
          }
          gl_FragColor = vec4(sampleView.rgb / max(sampleView.a, 0.001), 1.0);
        }`,
      depthTest: false, depthWrite: false,
    });
    const resolveScene = new THREE.Scene();
    resolveGeometry = new THREE.PlaneGeometry(2, 2);
    resolveScene.add(new THREE.Mesh(resolveGeometry, resolveMaterial));
    const resolveCamera = new THREE.Camera();
    spark = new SparkRenderer({ renderer, autoUpdate: false, enableLod: false, blurAmount: 0, preBlurAmount: 0 });
    scene.add(spark);
    const center = new THREE.Vector3(), scale = new THREE.Vector3();
    const quaternion = new THREE.Quaternion(), color = new THREE.Color();
    const normal = new THREE.Vector3(), axis = new THREE.Vector3(0, 0, 1);
    const point = (x, y) => {
      x = Math.max(0, Math.min(w - 1, x)); y = Math.max(0, Math.min(h - 1, y));
      const d = distances[y * w + x];
      return new THREE.Vector3(((x + .5) / w * 2 - 1) * tangent * aspect * d,
        (1 - (y + .5) / h * 2) * tangent * d, -d);
    };
    mesh = new SplatMesh({ maxSplats: w * h, lod: false, constructSplats: splats => {
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x, d = distances[i];
        const edge = refinement.edgeMask[i] !== 0;
        center.copy(point(x, y));
        const radius = edge ? .35 : .45;
        scale.set(2 * tangent * aspect * d / w * radius, 2 * tangent * d / h * radius, .001);
        // Orient patches on smooth surfaces, but don't tilt across occlusions.
        if (!edge && x > 0 && y > 0 && x < w - 1 && y < h - 1) {
          normal.crossVectors(point(x + 1, y).sub(point(x - 1, y)), point(x, y - 1).sub(point(x, y + 1))).normalize();
          quaternion.setFromUnitVectors(axis, normal);
        } else quaternion.identity();
        // Spark stores display RGB in its splat buffer (same convention as capture).
        color.setRGB(rgba[i * 4] / 255, rgba[i * 4 + 1] / 255, rgba[i * 4 + 2] / 255);
        splats.pushSplat(center, scale, quaternion, 1, color);
      }
    } });
    await mesh.initialized;
    signal.throwIfAborted();
    scene.add(mesh);
    const effect = {
      intensity: .35,
      trajectory: 'lateral',
      pivot: { u: .5, v: .5 },
      seconds: (performance.now() - start) / 1000,
      refinementSeconds: refinement.seconds,
      count: w * h,
      async render(progress) {
        if (disposed) throw new Error('L’aperçu a été libéré.');
        if (renderer.getContext().isContextLost()) throw new Error('Le rendu 3D a été interrompu par l’appareil. Réessaie avec une vidéo plus petite.');
        const p = Math.max(0, Math.min(1, progress));
        if (effect.intensity === 0 || p === 0 || p === 1) {
          ctx.clearRect(0, 0, output.width, output.height);
          ctx.drawImage(frame, 0, 0);
          return output;
        }
        const travel = Math.sin(Math.PI * p) ** 2;
        const u = Number.isFinite(effect.pivot?.u) ? Math.max(.01, Math.min(.99, effect.pivot.u)) : .5;
        const v = Number.isFinite(effect.pivot?.v) ? Math.max(.01, Math.min(.99, effect.pivot.v)) : .5;
        const pivotDepth = distances[Math.min(h - 1, Math.floor(v * h)) * w + Math.min(w - 1, Math.floor(u * w))];
        const pivotWorld = new THREE.Vector3((2 * u - 1) * tangent * aspect * pivotDepth,
          (1 - 2 * v) * tangent * pivotDepth, -pivotDepth);
        moveCamera(camera, pivotWorld, effect.trajectory, effect.intensity * travel);
        camera.zoom = 1;
        camera.updateProjectionMatrix(); camera.updateMatrixWorld();
        // Crop to the projected image perimeter, rather than showing the old
        // frame underneath exposed borders. Depth varies along the perimeter.
        let cropZoom = 1;
        const px = 2 * u - 1, py = 1 - 2 * v;
        const cover = (edge, anchor, extent) => {
          const ratio = (extent - anchor) / (edge - anchor);
          if (ratio > 0 && Number.isFinite(ratio)) cropZoom = Math.max(cropZoom, ratio);
        };
        for (let y = 0; y < h; y++) {
          cover(point(0, y).project(camera).x, px, -(1 - 1 / w));
          cover(point(w - 1, y).project(camera).x, px, 1 - 1 / w);
        }
        for (let x = 0; x < w; x++) {
          cover(point(x, 0).project(camera).y, py, 1 - 1 / h);
          cover(point(x, h - 1).project(camera).y, py, -(1 - 1 / h));
        }
        const zoom = Math.max(1 + .025 * effect.intensity * travel,
          (1 + .005 * effect.intensity * travel) * cropZoom);
        anchorProjection(camera, zoom, u, v);
        const projectedPivot = pivotWorld.clone().project(camera);
        effect.lastView = { position: camera.position.toArray(), target: pivotWorld.toArray(), zoom,
          pivot: { u: (projectedPivot.x + 1) / 2, v: (1 - projectedPivot.y) / 2 } };
        await spark.update({ scene, camera });
        if (disposed) throw new DOMException('Annulé', 'AbortError');
        renderer.setRenderTarget(viewTarget);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        renderer.render(resolveScene, resolveCamera);
        ctx.clearRect(0, 0, output.width, output.height);
        const transition = smooth(p / .04) * smooth((1 - p) / .04);
        // Brief raccords only. The middle is exclusively the moved 3D view.
        if (transition < 1) ctx.drawImage(frame, 0, 0);
        ctx.globalAlpha = transition;
        ctx.drawImage(renderer.domElement, 0, 0);
        ctx.globalAlpha = 1;
        return output;
      },
      dispose,
    };
    await effect.render(0);
    signal.throwIfAborted();
    return effect;
  } catch (error) { dispose(); throw error; }
}

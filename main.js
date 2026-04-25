import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

// =====================
// BACKGROUND 3D SCENE
// =====================
const container = document.getElementById("three-container");
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const BackgroundScene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60,
  window.innerWidth / window.innerHeight, 0.01, 1000
);
camera.position.z = 3;

const spark = new SparkRenderer({ renderer });
BackgroundScene.add(spark);

const splat = new SplatMesh({ url: "./daisy.sog" });
BackgroundScene.add(splat);

renderer.setAnimationLoop((t) => {
  const time = t * 0.001;
  splat.position.x = Math.sin(time * 0.3) * 0.4;
  splat.position.y = Math.cos(time * 0.2) * 0.2;
  renderer.render(BackgroundScene, camera);
});

// =====================
// AVATAR SCENE
// =====================
const avatarContainer = document.getElementById("avatar");
const avatarRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
avatarRenderer.setSize(110, 110);
avatarRenderer.setPixelRatio(window.devicePixelRatio);
const avatarCanvas = avatarRenderer.domElement;
avatarContainer.appendChild(avatarCanvas);

const avatarScene = new THREE.Scene();
const avatarCamera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);

const avatarSpark = new SparkRenderer({ renderer: avatarRenderer });
avatarScene.add(avatarSpark);

const avatarSplat = new SplatMesh({ url: "./flowers.sog" });
const avatarTarget = new THREE.Vector3(0, 0, 0);
avatarSplat.rotation.x = Math.PI;
avatarSplat.position.y = -0.5;
avatarScene.add(avatarSplat);

let isDragging = false;
let lastX = 0;
let lastY = 0;
let yaw = 0;
let pitch = 0;

avatarCanvas.addEventListener("mousedown", (e) => {
  isDragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
});

// touch support
avatarCanvas.addEventListener("touchstart", (e) => {
  isDragging = true;
  lastX = e.touches[0].clientX;
  lastY = e.touches[0].clientY;
}, { passive: true });

window.addEventListener("mouseup", () => { isDragging = false; });
window.addEventListener("touchend", () => { isDragging = false; });

window.addEventListener("mousemove", (e) => {
  if (!isDragging) return;
  yaw  -= (e.clientX - lastX) * 0.005;
  pitch -= (e.clientY - lastY) * 0.005;
  pitch  = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, pitch));
  lastX = e.clientX;
  lastY = e.clientY;
});

window.addEventListener("touchmove", (e) => {
  if (!isDragging) return;
  yaw  -= (e.touches[0].clientX - lastX) * 0.005;
  pitch -= (e.touches[0].clientY - lastY) * 0.005;
  pitch  = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, pitch));
  lastX = e.touches[0].clientX;
  lastY = e.touches[0].clientY;
}, { passive: true });

avatarRenderer.setAnimationLoop(() => {
  if (!isDragging) yaw += 0.002;
  const radius = 1.5;
  avatarCamera.position.x = Math.cos(pitch) * Math.sin(yaw) * radius;
  avatarCamera.position.y = Math.sin(pitch) * radius;
  avatarCamera.position.z = Math.cos(pitch) * Math.cos(yaw) * radius;
  avatarCamera.lookAt(avatarTarget);
  avatarRenderer.render(avatarScene, avatarCamera);
});

// =====================
// LANGUAGE TOGGLE
// =====================
const toggle = document.getElementById("lang-toggle");
let lang = "fr";

toggle.addEventListener("click", () => {
  lang = lang === "fr" ? "en" : "fr";
  toggle.textContent = lang === "fr" ? "EN" : "FR";
  document.querySelectorAll("[data-fr]").forEach(el => {
    el.textContent = el.dataset[lang];
  });
});

// init text
document.querySelectorAll("[data-fr]").forEach(el => {
  el.textContent = el.dataset["fr"];
});

// =====================
// SCROLL REVEAL
// =====================
const revealEls = document.querySelectorAll(".reveal");
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });

revealEls.forEach(el => revealObserver.observe(el));

// =====================
// ACTIVE NAV HIGHLIGHT
// =====================
const sections = document.querySelectorAll("section[id]");
const navLinks = document.querySelectorAll(".nav-item");

const navObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      navLinks.forEach(link => {
        link.classList.toggle("active",
          link.getAttribute("href") === `#${entry.target.id}`
        );
      });
    }
  });
}, { threshold: 0.0, rootMargin: "-10% 0px -85% 0px" });

sections.forEach(s => navObserver.observe(s));

// =====================
// RESIZE
// =====================
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

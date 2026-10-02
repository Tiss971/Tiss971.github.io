import * as THREE from 'three';

const vertical = new THREE.Vector3(0, 1, 0);

// Start with the source camera at the origin, looking down -Z. Rotate only as
// much as needed to keep the chosen source ray aimed at the same scene point.
export function moveCamera(camera, pivot, trajectory, amount) {
  const depth = -pivot.z;
  if (trajectory === 'arc') {
    camera.position.copy(pivot).add(pivot.clone().negate().applyAxisAngle(vertical, .11 * amount));
  } else if (trajectory === 'dolly') {
    camera.position.set(depth * .05 * amount, depth * .025 * amount, -depth * .10 * amount);
  } else {
    camera.position.set(depth * .09 * amount, 0, 0);
  }
  camera.quaternion.setFromUnitVectors(
    pivot.clone().normalize(), pivot.clone().sub(camera.position).normalize());
}

export function anchorProjection(camera, zoom, u, v) {
  camera.zoom = zoom;
  camera.updateProjectionMatrix();
  // Zoom/crop around the selected subject, preserving its original screen
  // location rather than recentering the image when a pivot is picked.
  camera.projectionMatrix.elements[8] = (2 * u - 1) * (zoom - 1);
  camera.projectionMatrix.elements[9] = (1 - 2 * v) * (zoom - 1);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const get = (id) => document.getElementById(id);
const viewport = get('canvas');
const status = get('status');
const runInput = get('run');
const viewInput = get('view');
const pointSourceInput = get('point-source');
const confidenceInput = get('confidence');
const confidenceValue = get('confidence-value');
const confidenceNote = get('confidence-note');
const sizeInput = get('size');
const sizeValue = get('size-value');
const colorInput = get('point-color');
const confidenceOption = colorInput.querySelector('[value="confidence"]');
const confidenceLegend = get('confidence-legend');
const frustumInput = get('frustums');
const controlsPanel = get('controls');
const navigation = get('navigation');
const pointCount = get('point-count');
const matchingCount = get('matching-count');
const availableCount = get('available-count');
const fitButton = get('fit');
const numberFormat = new Intl.NumberFormat();
const srgbToLinear = Float32Array.from({ length: 256 }, (_, value) => {
  const srgb = value / 255;
  return srgb <= 0.04045 ? srgb / 12.92 : ((srgb + 0.055) / 1.055) ** 2.4;
});
const confidenceStops = [
  '#440154', '#472d7b', '#3b528b', '#2c728e', '#21918c', '#28ae80', '#5ec962', '#addc30', '#fde725',
];
confidenceLegend.style.setProperty('--confidence-colors', confidenceStops.join(', '));
const confidenceColors = confidenceStops.map((color) => new THREE.Color(color));
const confidencePalette = Array.from({ length: 101 }, (_, percentile) => {
  const position = (percentile / 100) * (confidenceColors.length - 1);
  const index = Math.min(Math.floor(position), confidenceColors.length - 2);
  return new THREE.Color().lerpColors(confidenceColors[index], confidenceColors[index + 1], position - index);
});
const unknownConfidenceColor = new THREE.Color('#778078');

function showStatus(message, error = false) {
  status.textContent = message;
  status.hidden = !message;
  status.classList.toggle('error', error);
}

function makeViewport() {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.domElement.setAttribute('aria-label', 'Interactive 3D point cloud');
  viewport.append(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#10140f');
  const camera = new THREE.PerspectiveCamera(48, 1, 0.001, 1000);
  camera.position.set(1, 0.7, 1.2);
  const orbit = new OrbitControls(camera, renderer.domElement);
  const draw = () => renderer.render(scene, camera);
  orbit.addEventListener('change', draw);

  function resize() {
    const width = viewport.clientWidth;
    const height = viewport.clientHeight;
    if (width === 0 || height === 0) return;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    draw();
  }

  new ResizeObserver(resize).observe(viewport);
  resize();
  return { scene, camera, orbit, draw };
}

async function fetchData(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    const { error } = await response.json();
    throw new Error(error || `Request failed (${response.status}).`);
  }
  return response;
}

function parseCloud(buffer) {
  const headerLength = new DataView(buffer).getUint32(0, true);
  const metadata = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerLength)));
  const offset = 4 + Math.ceil(headerLength / 4) * 4;
  const positions = new Float32Array(buffer, offset, metadata.count * 3);
  const rgb = new Uint8Array(buffer, offset + positions.byteLength, metadata.count * 3);
  const confidenceRanks = metadata.has_confidence
    ? new Uint8Array(buffer, rgb.byteOffset + rgb.byteLength, metadata.count)
    : null;
  return { metadata, positions, rgb, confidenceRanks };
}

function updatePointColors(cloud) {
  const { pointMaterial, points, rgb, confidenceRanks } = cloud;
  const vertexColors = colorInput.value !== 'solid';
  const byConfidence = colorInput.value === 'confidence' && confidenceRanks !== null;
  confidenceLegend.hidden = !byConfidence;
  if (pointMaterial.vertexColors !== vertexColors) {
    pointMaterial.vertexColors = vertexColors;
    pointMaterial.needsUpdate = true;
  }
  pointMaterial.color.set(vertexColors ? '#ffffff' : '#e4e9de');
  if (!vertexColors) return;

  const attribute = points.geometry.getAttribute('color');
  const colors = attribute.array;
  if (byConfidence) {
    for (let index = 0; index < confidenceRanks.length; index++) {
      const color = confidencePalette[confidenceRanks[index]] ?? unknownConfidenceColor;
      color.toArray(colors, index * 3);
    }
  } else {
    for (let index = 0; index < rgb.length; index++) colors[index] = srgbToLinear[rgb[index]];
  }
  attribute.needsUpdate = true;
}

function buildCloud(data) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  const colors = new THREE.BufferAttribute(new Float32Array(data.rgb.length), 3);
  geometry.setAttribute('color', colors.setUsage(THREE.DynamicDrawUsage));
  if (data.metadata.count) geometry.computeBoundingSphere();
  const pointMaterial = new THREE.PointsMaterial({
    size: Number(sizeInput.value),
    sizeAttenuation: false,
  });
  const points = new THREE.Points(geometry, pointMaterial);
  points.visible = data.metadata.count > 0;

  const cameraMaterial = new THREE.LineBasicMaterial({
    color: '#efbd78', opacity: 0.85, transparent: true, depthTest: false,
  });
  const edges = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1];
  const cameras = data.metadata.views.map(({ frustum }) => {
    const lines = new THREE.BufferGeometry();
    lines.setAttribute('position', new THREE.Float32BufferAttribute(edges.flatMap((index) => frustum[index]), 3));
    return new THREE.LineSegments(lines, cameraMaterial);
  });
  const group = new THREE.Group();
  group.add(points, ...cameras);
  return { ...data, group, points, cameras, pointMaterial, cameraMaterial };
}

function disposeCloud(scene, cloud) {
  if (!cloud) return;
  scene.remove(cloud.group);
  cloud.points.geometry.dispose();
  cloud.pointMaterial.dispose();
  for (const camera of cloud.cameras) camera.geometry.dispose();
  cloud.cameraMaterial.dispose();
}

function fitView(viewportState, positions) {
  if (!positions.length) return;
  const { camera, orbit, draw } = viewportState;
  const bounds = new THREE.Box3().setFromArray(positions);
  const { center, radius } = bounds.getBoundingSphere(new THREE.Sphere());
  const vertical = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const horizontal = vertical * camera.aspect;
  const sceneRadius = Math.max(radius, 0.001);
  const direction = new THREE.Vector3(1, 0.7, 1.2).normalize();
  const right = new THREE.Vector3().crossVectors(camera.up, direction).normalize();
  const up = new THREE.Vector3().crossVectors(direction, right);
  // Fit the visible points against the actual horizontal and vertical fields of view.
  let distance = sceneRadius * 0.1;
  const point = new THREE.Vector3();
  for (let index = 0; index < positions.length; index += 3) {
    point.fromArray(positions, index).sub(center);
    const reach = Math.max(Math.abs(point.dot(right)) / horizontal, Math.abs(point.dot(up)) / vertical);
    distance = Math.max(distance, point.dot(direction) + reach * 1.12);
  }
  distance += sceneRadius * 0.02;

  orbit.target.copy(center);
  camera.position.copy(center).addScaledVector(direction, distance);
  camera.near = sceneRadius / 1000;
  camera.far = distance + sceneRadius * 64;
  camera.updateProjectionMatrix();
  orbit.minDistance = sceneRadius / 100;
  orbit.maxDistance = distance + sceneRadius * 50;
  orbit.update();
  draw();
}

async function initialize() {
  let viewportState;
  try {
    viewportState = makeViewport();
  } catch {
    throw new Error('The 3D viewer needs a browser with WebGL 2 enabled.');
  }

  const { scene, draw } = viewportState;
  let cloud;
  let request;
  let refreshTimer;
  let needsFit = true;

  function updateConfidenceLabel(hasConfidence) {
    const percentile = Number(confidenceInput.value);
    confidenceInput.disabled = !hasConfidence;
    confidenceValue.value = hasConfidence ? `${percentile}%` : '—';
    confidenceNote.textContent = hasConfidence
      ? `Keep the top ${100 - percentile}% across this run.`
      : 'Confidence scores are unavailable for this run.';
  }

  function fitSelection() {
    if (cloud) fitView(viewportState, cloud.positions);
  }

  async function loadCloud(newRun = false) {
    clearTimeout(refreshTimer);
    request?.abort();
    const pending = new AbortController();
    request = pending;
    if (newRun) {
      controlsPanel.disabled = true;
      navigation.hidden = true;
      pointCount.value = '—';
      matchingCount.value = '—';
      availableCount.hidden = true;
      showStatus('Loading point cloud…');
      disposeCloud(scene, cloud);
      cloud = undefined;
      needsFit = true;
      draw();
    }
    const query = new URLSearchParams({
      run: runInput.value,
      percentile: confidenceInput.value,
      view: newRun ? '-1' : viewInput.value,
      point_source: pointSourceInput.value,
    });

    try {
      const response = await fetchData(`/api/cloud?${query}`, pending.signal);
      const buffer = await response.arrayBuffer();
      if (pending.signal.aborted) return;
      const next = buildCloud(parseCloud(buffer));
      disposeCloud(scene, cloud);
      cloud = next;
      scene.add(cloud.group);
      const { count, matching, available, has_confidence: hasConfidence, views } = cloud.metadata;
      if (newRun) {
        viewInput.replaceChildren(
          new Option(`All ${views.length} source images`, '-1'),
          ...views.map(({ name }, index) => new Option(name, String(index))),
        );
      }
      for (const camera of cloud.cameras) camera.visible = frustumInput.checked;
      controlsPanel.disabled = false;
      navigation.hidden = false;
      pointSourceInput.value = cloud.metadata.point_source;
      pointSourceInput.disabled = !cloud.metadata.has_direct_points;
      updateConfidenceLabel(hasConfidence);
      confidenceOption.disabled = !hasConfidence;
      if (!hasConfidence && colorInput.value === 'confidence') colorInput.value = 'image';
      updatePointColors(cloud);
      pointCount.value = numberFormat.format(count);
      matchingCount.value = numberFormat.format(matching);
      availableCount.textContent = `Of ${numberFormat.format(available)} valid points`;
      availableCount.hidden = false;
      fitButton.disabled = count === 0;
      const empty = available ? 'No points meet the confidence cutoff.' : 'This source image has no valid points.';
      showStatus(count ? '' : empty);
      if (needsFit && count) {
        needsFit = false;
        fitSelection();
      } else {
        draw();
      }
    } catch (error) {
      if (!pending.signal.aborted) {
        if (cloud) pointSourceInput.value = cloud.metadata.point_source;
        showStatus(error.message, true);
      }
    }
  }

  function queueCloud(delay) {
    clearTimeout(refreshTimer);
    request?.abort();
    showStatus('Updating point cloud…');
    refreshTimer = setTimeout(() => loadCloud(), delay);
  }

  runInput.addEventListener('change', () => loadCloud(true));
  pointSourceInput.addEventListener('change', () => queueCloud(0));
  viewInput.addEventListener('change', () => queueCloud(0));
  confidenceInput.addEventListener('input', () => {
    updateConfidenceLabel(cloud.metadata.has_confidence);
    queueCloud(150);
  });
  colorInput.addEventListener('change', () => {
    if (!cloud) return;
    updatePointColors(cloud);
    draw();
  });
  frustumInput.addEventListener('change', () => {
    if (!cloud) return;
    for (const camera of cloud.cameras) camera.visible = frustumInput.checked;
    draw();
  });
  sizeInput.addEventListener('input', () => {
    const size = Number(sizeInput.value);
    sizeValue.value = `${size.toFixed(1)} px`;
    if (cloud) {
      cloud.pointMaterial.size = size;
      draw();
    }
  });
  fitButton.addEventListener('click', fitSelection);

  const response = await fetchData('/api/runs');
  const { runs } = await response.json();
  if (!runs.length) {
    showStatus('No dvlt_predictions.npz files were found. Start the viewer with --root pointing to the prediction directory.');
    return;
  }
  runInput.replaceChildren(...runs.map((run) => new Option(run === '.' ? 'Root run' : run, run)));
  runInput.value = runs.includes('all') ? 'all' : runs[0];
  runInput.disabled = false;
  await loadCloud(true);
}

initialize().catch((error) => showStatus(error.message, true));

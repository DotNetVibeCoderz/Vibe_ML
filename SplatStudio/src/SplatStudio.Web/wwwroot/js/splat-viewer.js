/* SplatStudio — minimal Gaussian-splat point-cloud viewer.
 *
 * Parses the project's own 32-byte-per-point binary .splat format (see
 * SplatFileWriter.cs):
 *   float32 x, y, z        (position, 12 bytes)
 *   float32 sx, sy, sz     (per-axis scale, 12 bytes)
 *   uint8   r, g, b, a     (color, 4 bytes)
 *   uint8   qx, qy, qz, qw (rotation quaternion encoded 0-255 <-> -1..1, 4 bytes)
 *
 * Renders every point as a camera-facing soft circular billboard sized by
 * its scale and tinted by its color/alpha. The per-point rotation quaternion
 * is intentionally not applied to the billboard orientation — a screen-facing
 * sprite looks identical regardless of the splat's stored rotation, and
 * decoding it here would add complexity with no visible benefit for this
 * lightweight renderer.
 *
 * Known simplification: points are not depth-sorted back-to-front before
 * blending (would mean re-sorting tens of thousands of points every frame).
 * For the soft, mostly-opaque billboards this engine produces, the visual
 * difference is minor — consistent with this project's "good enough, fully
 * offline" approach rather than a physically exact splat renderer.
 */
(function () {
  'use strict';

  const SPLAT_BYTES = 32;

  function parseSplatBuffer(arrayBuffer) {
    const dv = new DataView(arrayBuffer);
    const count = Math.floor(arrayBuffer.byteLength / SPLAT_BYTES);
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 4);
    const sizes = new Float32Array(count);

    for (let i = 0; i < count; i++) {
      const o = i * SPLAT_BYTES;
      positions[i * 3 + 0] = dv.getFloat32(o + 0, true);
      positions[i * 3 + 1] = dv.getFloat32(o + 4, true);
      positions[i * 3 + 2] = dv.getFloat32(o + 8, true);

      const sx = dv.getFloat32(o + 12, true);
      const sy = dv.getFloat32(o + 16, true);
      const sz = dv.getFloat32(o + 20, true);

      colors[i * 4 + 0] = dv.getUint8(o + 24) / 255;
      colors[i * 4 + 1] = dv.getUint8(o + 25) / 255;
      colors[i * 4 + 2] = dv.getUint8(o + 26) / 255;
      colors[i * 4 + 3] = dv.getUint8(o + 27) / 255;

      sizes[i] = Math.max(sx, sy, sz, 0.002);
    }

    return { count, positions, colors, sizes };
  }

  const VERTEX_SHADER = [
    'attribute float pointSize;',
    'attribute vec4 pointColor;',
    'varying vec4 vColor;',
    'void main() {',
    '  vColor = pointColor;',
    '  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);',
    '  gl_PointSize = pointSize * (480.0 / max(-mvPosition.z, 0.001));',
    '  gl_Position = projectionMatrix * mvPosition;',
    '}'
  ].join('\n');

  const FRAGMENT_SHADER = [
    'varying vec4 vColor;',
    'void main() {',
    '  vec2 uv = gl_PointCoord - vec2(0.5);',
    '  float d = length(uv);',
    '  if (d > 0.5) discard;',
    '  float falloff = smoothstep(0.5, 0.05, d);',
    '  gl_FragColor = vec4(vColor.rgb, vColor.a * falloff);',
    '}'
  ].join('\n');

  class Viewer {
    constructor(container) {
      this.container = container;
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      container.innerHTML = '';
      container.appendChild(this.renderer.domElement);

      this.points = null;
      this.target = new THREE.Vector3(0, 0, 0);
      this.radius = 2.2;
      this.theta = Math.PI / 2.3;
      this.phi = -Math.PI / 4;
      this._updateCamera();

      this._disposed = false;
      this._bindControls();
      this._resize();

      this._resizeObserver = new ResizeObserver(() => this._resize());
      this._resizeObserver.observe(container);

      this._animate = this._animate.bind(this);
      this._raf = requestAnimationFrame(this._animate);
    }

    async load(url) {
      const response = await fetch(url, { cache: 'force-cache' });
      if (!response.ok) throw new Error('Failed to fetch splat file: ' + response.status);
      const buffer = await response.arrayBuffer();
      if (this._disposed) return;

      const { positions, colors, sizes } = parseSplatBuffer(buffer);

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute('pointColor', new THREE.BufferAttribute(colors, 4));
      geometry.setAttribute('pointSize', new THREE.BufferAttribute(sizes, 1));

      const material = new THREE.ShaderMaterial({
        vertexShader: VERTEX_SHADER,
        fragmentShader: FRAGMENT_SHADER,
        transparent: true,
        depthWrite: false,
        blending: THREE.NormalBlending
      });

      this._clearPoints();
      this.points = new THREE.Points(geometry, material);
      this.scene.add(this.points);

      geometry.computeBoundingSphere();
      const bs = geometry.boundingSphere;
      if (bs && bs.radius > 0) {
        this.target.copy(bs.center);
        this.radius = Math.max(bs.radius * 2.6, 0.6);
        this._updateCamera();
      }
    }

    _clearPoints() {
      if (!this.points) return;
      this.scene.remove(this.points);
      this.points.geometry.dispose();
      this.points.material.dispose();
      this.points = null;
    }

    _bindControls() {
      const el = this.renderer.domElement;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      const down = (x, y) => { dragging = true; lastX = x; lastY = y; };
      const move = (x, y) => {
        if (!dragging) return;
        const dx = x - lastX;
        const dy = y - lastY;
        lastX = x;
        lastY = y;
        this.phi -= dx * 0.006;
        this.theta = Math.min(Math.max(this.theta - dy * 0.006, 0.12), Math.PI - 0.12);
        this._updateCamera();
      };
      const up = () => { dragging = false; };

      this._onMouseDown = (e) => down(e.clientX, e.clientY);
      this._onMouseMove = (e) => move(e.clientX, e.clientY);
      this._onMouseUp = up;
      this._onWheel = (e) => {
        e.preventDefault();
        this.radius = Math.min(Math.max(this.radius * (1 + e.deltaY * 0.001), 0.2), 15);
        this._updateCamera();
      };
      this._onTouchStart = (e) => {
        if (e.touches.length === 1) down(e.touches[0].clientX, e.touches[0].clientY);
      };
      this._onTouchMove = (e) => {
        if (e.touches.length === 1) move(e.touches[0].clientX, e.touches[0].clientY);
      };
      this._onTouchEnd = up;

      el.addEventListener('mousedown', this._onMouseDown);
      window.addEventListener('mousemove', this._onMouseMove);
      window.addEventListener('mouseup', this._onMouseUp);
      el.addEventListener('wheel', this._onWheel, { passive: false });
      el.addEventListener('touchstart', this._onTouchStart, { passive: true });
      el.addEventListener('touchmove', this._onTouchMove, { passive: true });
      el.addEventListener('touchend', this._onTouchEnd);
    }

    _unbindControls() {
      const el = this.renderer.domElement;
      el.removeEventListener('mousedown', this._onMouseDown);
      window.removeEventListener('mousemove', this._onMouseMove);
      window.removeEventListener('mouseup', this._onMouseUp);
      el.removeEventListener('wheel', this._onWheel);
      el.removeEventListener('touchstart', this._onTouchStart);
      el.removeEventListener('touchmove', this._onTouchMove);
      el.removeEventListener('touchend', this._onTouchEnd);
    }

    _updateCamera() {
      const x = this.target.x + this.radius * Math.sin(this.theta) * Math.cos(this.phi);
      const y = this.target.y + this.radius * Math.cos(this.theta);
      const z = this.target.z + this.radius * Math.sin(this.theta) * Math.sin(this.phi);
      this.camera.position.set(x, y, z);
      this.camera.lookAt(this.target);
    }

    _resize() {
      const w = Math.max(this.container.clientWidth, 1);
      const h = Math.max(this.container.clientHeight, 1);
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(w, h, false);
    }

    _animate() {
      if (this._disposed) return;
      this.renderer.render(this.scene, this.camera);
      this._raf = requestAnimationFrame(this._animate);
    }

    dispose() {
      this._disposed = true;
      cancelAnimationFrame(this._raf);
      this._resizeObserver.disconnect();
      this._unbindControls();
      this._clearPoints();
      this.renderer.dispose();
      const el = this.renderer.domElement;
      if (el && el.parentNode) el.parentNode.removeChild(el);
    }
  }

  const instances = new WeakMap();

  window.splatViewer = {
    init(container, url) {
      if (!container || typeof THREE === 'undefined') return;
      const existing = instances.get(container);
      if (existing) existing.dispose();

      const viewer = new Viewer(container);
      instances.set(container, viewer);
      viewer.load(url).catch((err) => {
        console.error('SplatStudio viewer: failed to load splat file', err);
      });
    },
    dispose(container) {
      if (!container) return;
      const viewer = instances.get(container);
      if (viewer) {
        viewer.dispose();
        instances.delete(container);
      }
    }
  };
})();

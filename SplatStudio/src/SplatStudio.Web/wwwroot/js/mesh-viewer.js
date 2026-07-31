/* SplatStudio — glTF mesh viewer.
 *
 * Mode 3 (hosted image-to-3D: TRELLIS, Hunyuan3D, Rodin) produces a textured mesh rather
 * than a point cloud, which needs real lighting and a glTF parser — neither of which the
 * splat viewer has or should grow. So this is a separate, deliberately small viewer that
 * shares only the orbit interaction model, so both artifact types feel the same to drive.
 *
 * Loaded lazily: most deployments never enable mode 3, and GLTFLoader is ~96 KB that those
 * deployments should not pay for on every page.
 */
(function () {
  'use strict';

  const prefersReducedMotion = () =>
    window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function showFailure(container, message) {
    if (!container) return;
    container.innerHTML =
      '<div class="viewer-failure"><p><strong>This model could not be rendered.</strong></p>' +
      '<p class="muted small">' + message + '</p></div>';
  }

  /* GLTFLoader is a separate script from the three.js build, fetched on first use. */
  let loaderPromise = null;
  function ensureLoader() {
    if (typeof THREE !== 'undefined' && THREE.GLTFLoader) return Promise.resolve();
    if (loaderPromise) return loaderPromise;

    loaderPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'js/vendor/GLTFLoader.js';
      script.onload = () => resolve();
      script.onerror = () => reject(new Error('GLTFLoader failed to load'));
      document.head.appendChild(script);
    });
    return loaderPromise;
  }

  class MeshViewer {
    constructor(container, options) {
      const opts = options || {};
      this.container = container;

      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
      this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      this.renderer.outputEncoding = THREE.sRGBEncoding;
      container.innerHTML = '';
      container.appendChild(this.renderer.domElement);

      /* Three-point-ish lighting. Generated meshes ship with baked albedo textures and no
       * scene lighting of their own, so without this they render as silhouettes. */
      this.scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.DirectionalLight(0xfff0e0, 1.15);
      key.position.set(2.5, 3, 2);
      this.scene.add(key);
      const fill = new THREE.DirectionalLight(0x9fb4ff, 0.5);
      fill.position.set(-2.5, 0.5, -1.5);
      this.scene.add(fill);

      this.model = null;
      this.target = new THREE.Vector3(0, 0, 0);
      this.radius = 3;
      this.theta = Math.PI / 2.3;
      this.phi = -Math.PI / 2 + 0.5;
      this._basePhi = this.phi;

      this.autoOrbit = !!opts.autoOrbit && !prefersReducedMotion();
      this.autoOrbitSpeed = typeof opts.autoOrbitSpeed === 'number' ? opts.autoOrbitSpeed : 0.35;
      this.autoOrbitArc = typeof opts.autoOrbitArc === 'number' ? opts.autoOrbitArc : 0.5;
      this._elapsed = 0;
      this._lastFrame = 0;

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
      const loader = new THREE.GLTFLoader();
      const gltf = await new Promise((resolve, reject) => {
        loader.load(url, resolve, undefined, reject);
      });
      if (this._disposed) return;

      this.model = gltf.scene;
      this.scene.add(this.model);

      /* Generated models arrive at arbitrary scale and origin. Normalise to a unit box at
       * the origin so the camera framing below works for any provider's output. */
      const box = new THREE.Box3().setFromObject(this.model);
      const size = box.getSize(new THREE.Vector3());
      const centre = box.getCenter(new THREE.Vector3());
      const longest = Math.max(size.x, size.y, size.z) || 1;

      this.model.position.sub(centre);
      this.model.scale.setScalar(1 / longest);

      // Pull back far enough that the whole unit box fits the vertical field of view.
      const fovRad = (this.camera.fov * Math.PI) / 180;
      this.radius = (0.5 / Math.tan(fovRad / 2)) * 2.4;
      this._updateCamera();
    }

    _bindControls() {
      const el = this.renderer.domElement;
      let dragging = false;
      let lastX = 0;
      let lastY = 0;

      const down = (x, y) => { dragging = true; lastX = x; lastY = y; this.autoOrbit = false; };
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
        this.autoOrbit = false;
        this.radius = Math.min(Math.max(this.radius * (1 + e.deltaY * 0.001), 0.6), 20);
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

    _animate(now) {
      if (this._disposed) return;

      if (this.autoOrbit) {
        const delta = this._lastFrame ? Math.min((now - this._lastFrame) / 1000, 0.1) : 0;
        this._elapsed += delta;
        this.phi = this._basePhi + Math.sin(this._elapsed * this.autoOrbitSpeed) * this.autoOrbitArc;
        this._updateCamera();
      }
      this._lastFrame = now;

      this.renderer.render(this.scene, this.camera);
      this._raf = requestAnimationFrame(this._animate);
    }

    dispose() {
      this._disposed = true;
      cancelAnimationFrame(this._raf);
      this._resizeObserver.disconnect();
      this._unbindControls();

      /* glTF scenes own geometries, materials and textures; three.js does not free any of
       * them on scene.remove, so walk the graph and release each one. */
      if (this.model) {
        this.model.traverse((node) => {
          if (node.geometry) node.geometry.dispose();
          if (!node.material) return;
          const materials = Array.isArray(node.material) ? node.material : [node.material];
          materials.forEach((material) => {
            Object.values(material).forEach((value) => {
              if (value && value.isTexture) value.dispose();
            });
            material.dispose();
          });
        });
        this.scene.remove(this.model);
        this.model = null;
      }

      this.renderer.dispose();
      const el = this.renderer.domElement;
      if (el && el.parentNode) el.parentNode.removeChild(el);
    }
  }

  const instances = new WeakMap();

  window.meshViewer = {
    init(container, url, options) {
      if (!container) return;
      if (typeof THREE === 'undefined') {
        showFailure(container, 'The 3D library did not load. Try reloading the page.');
        return;
      }

      ensureLoader()
        .then(() => {
          if (!container.isConnected) return;

          const existing = instances.get(container);
          if (existing) existing.dispose();

          const viewer = new MeshViewer(container, options);
          instances.set(container, viewer);

          return viewer.load(url).catch((err) => {
            console.error('SplatStudio mesh viewer: failed to load model', err);
            viewer.dispose();
            instances.delete(container);
            showFailure(container, 'The model file could not be downloaded or parsed.');
          });
        })
        .catch((err) => {
          console.error('SplatStudio mesh viewer: loader unavailable', err);
          showFailure(container, 'The glTF loader could not be fetched.');
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

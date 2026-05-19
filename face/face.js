import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { GlitchEngine } from './glitch.js';

(function () {
  const canvas = document.getElementById('face-canvas');
  const W = canvas.width  = window.innerWidth;
  const H = canvas.height = window.innerHeight;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = false;

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, W / H, 0.1, 100);
  camera.position.set(0, 0, 2.2);

  // ── HDRI environment ───────────────────────────────────────────
  const pmrem = new THREE.PMREMGenerator(renderer);
  pmrem.compileEquirectangularShader();
  new THREE.TextureLoader().load(
    'hdris/studio_small_03_4k copia.jpg',
    (tex) => {
      tex.mapping = THREE.EquirectangularReflectionMapping;
      scene.environment = pmrem.fromEquirectangular(tex).texture;
      pmrem.dispose();
      tex.dispose();
      console.log('[FACE] HDRI loaded OK');
    },
    undefined,
    (err) => console.error('[FACE] HDRI load error:', err),
  );

  // ── State ──────────────────────────────────────────────────────
  let subtitleColor = new THREE.Color(0x00ffff);
  let targetSubtitleColor = new THREE.Color(0x00ffff);

  // Blend shape layers (merged in priority order, higher = wins)
  const BASE_SHAPES   = { browDownLeft: 0.3, browDownRight: 0.3 };
  let amplitudeShapes = {};
  let idleShapes      = {};
  let gazeShapes      = {};
  let targetShapes    = {};
  let blinkShapes     = {};
  const currentShapes = {};

  // Viseme watchdog: clear targetShapes if no viseme received for 500ms
  let lastVisemeAt = 0;

  // Timers
  let nextBlink   = Date.now() + 2000 + Math.random() * 3000;
  let blinkActive = false;
  let nextSaccade = Date.now() + 1000 + Math.random() * 2000;
  let saccadeActive = false;
  let nextMicro   = Date.now() + 3000 + Math.random() * 5000;
  let microActive = false;

  // Model references
  let modelRoot   = null;
  let headMesh    = null;
  let morphMeshes = [];
  let glitchMat   = null;

  // ── Load model ─────────────────────────────────────────────────
  const loader = new GLTFLoader();

  const texLoader = new THREE.TextureLoader();
  function loadTex(path, colorSpace = THREE.SRGBColorSpace) {
    const t = texLoader.load(path, undefined, undefined, (e) => console.warn('[FACE] tex missing:', path));
    t.colorSpace = colorSpace;
    t.flipY = false;
    return t;
  }

  loader.load('models/52shapes_v1.glb', (gltf) => {
    const albedoTex = loadTex('models/JPG/Albedo.jpg');
    const normalTex = loadTex('models/JPG/Normal.jpg', THREE.LinearSRGBColorSpace);
    const cavityTex = loadTex('models/JPG/Cavity.jpg', THREE.LinearSRGBColorSpace);
    const teethTex  = loadTex('models/JPG/Teeth.jpg');
    const eyeTex    = loadTex('models/JPG/Eye Colour.jpg');

    const headMat = new THREE.MeshStandardMaterial({
      map:             albedoTex,
      normalMap:       normalTex,
      aoMap:           cavityTex,
      aoMapIntensity:  1.0,
      metalness:       0.0,
      roughness:       0.6,
      envMapIntensity: 1.75,
    });
    const darkMat = new THREE.MeshStandardMaterial({
      color:     0xd4a017,
      metalness: 1.0,
      roughness: 0.1,
      envMapIntensity: 1.0,
    });
    const teethMat = new THREE.MeshStandardMaterial({
      color:           0x000000,
      metalness:       0.0,
      roughness:       0.0,
      envMapIntensity: 2.0,
    });
    const eyeMat = new THREE.MeshStandardMaterial({
      color:           0x000000,
      metalness:       0.0,
      roughness:       0.0,
      envMapIntensity: 2.0,
    });

    gltf.scene.traverse(obj => {
      if (!obj.isMesh) return;
      obj.castShadow    = false;
      obj.receiveShadow = false;

      const n = obj.name;
      if (n === 'teeth_ORIGINAL') {
        obj.material = teethMat.clone();
      } else if (n === 'eyeLeft_ORIGINAL' || n === 'eyeRight_ORIGINAL') {
        obj.material = eyeMat.clone();
      } else {
        obj.material = headMat.clone();
        if (!glitchMat) glitchMat = obj.material;
      }

      if (obj.morphTargetDictionary && Object.keys(obj.morphTargetDictionary).length > 0) {
        morphMeshes.push(obj);
        if (n === 'head_lod0_ORIGINAL') headMesh = obj;
      }
    });

    const box    = new THREE.Box3().setFromObject(gltf.scene);
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = box.getSize(new THREE.Vector3()).length();
    const scale  = 2.8 / maxDim;
    gltf.scene.scale.setScalar(scale);
    gltf.scene.position.set(
      -center.x * scale,
      -center.y * scale - 0.12,
      -center.z * scale,
    );

    scene.add(gltf.scene);
    modelRoot = gltf.scene;

    if (headMesh) {
      console.log('[FACE] Morph targets:', Object.keys(headMesh.morphTargetDictionary));
    }
  }, undefined, err => console.error('[FACE] Load error:', err));

  // ── Blend shape helpers ────────────────────────────────────────
  function lerpMorphTargets() {
    if (!headMesh?.morphTargetDictionary) return;
    const dict = headMesh.morphTargetDictionary;

    // Watchdog: clear stale visemes
    if (Object.keys(targetShapes).length > 0 && Date.now() - lastVisemeAt > 2000) {
      targetShapes = {};
    }

    const merged = { ...BASE_SHAPES, ...amplitudeShapes, ...idleShapes, ...gazeShapes, ...targetShapes, ...blinkShapes };

    for (const key of Object.keys(dict)) {
      const target = isFinite(merged[key]) ? merged[key] : 0;
      const cur    = isFinite(currentShapes[key]) ? currentShapes[key] : 0;
      currentShapes[key] = cur + (target - cur) * 0.25;
      const val = currentShapes[key];
      for (const mesh of morphMeshes) {
        const idx = mesh.morphTargetDictionary[key];
        if (idx !== undefined) mesh.morphTargetInfluences[idx] = val;
      }
    }
  }

  // ── Blink ──────────────────────────────────────────────────────
  function tickBlink() {
    if (blinkActive || Date.now() < nextBlink) return;
    blinkActive = true;
    blinkShapes = { eyeBlinkLeft: 1.0, eyeBlinkRight: 1.0 };
    setTimeout(() => {
      blinkShapes  = {};
      blinkActive  = false;
      nextBlink    = Date.now() + 2500 + Math.random() * 4000;
    }, 120);
  }

  // ── Eye saccades ───────────────────────────────────────────────
  function tickSaccade() {
    if (saccadeActive || Date.now() < nextSaccade) return;
    saccadeActive = true;
    const dirs = [
      { eyeLookInLeft: 0.5, eyeLookOutRight: 0.5 },
      { eyeLookOutLeft: 0.5, eyeLookInRight: 0.5 },
      { eyeLookUpLeft: 0.35, eyeLookUpRight: 0.35 },
      { eyeLookDownLeft: 0.4, eyeLookDownRight: 0.4 },
      {}, {},
    ];
    gazeShapes = dirs[Math.floor(Math.random() * dirs.length)];
    setTimeout(() => {
      gazeShapes    = {};
      saccadeActive = false;
      nextSaccade   = Date.now() + 1500 + Math.random() * 3000;
    }, 200 + Math.random() * 800);
  }

  // ── Micro-expressions ──────────────────────────────────────────
  function tickMicroExpression() {
    if (microActive || Date.now() < nextMicro) return;
    microActive = true;
    const exprs = [
      { browDownLeft: 0.9, browDownRight: 0.9, noseSneerLeft: 0.5 },
      { eyeSquintLeft: 0.8, eyeSquintRight: 0.8, browDownLeft: 0.6, browDownRight: 0.6 },
      { browInnerUp: 0.9, eyeWideLeft: 0.5, eyeWideRight: 0.5 },
      { noseSneerLeft: 0.7, noseSneerRight: 0.4, mouthFrownLeft: 0.5, mouthFrownRight: 0.5 },
      { browDownLeft: 0.7, browDownRight: 0.7, eyeSquintLeft: 0.4, eyeSquintRight: 0.4 },
      { cheekSquintLeft: 0.5, cheekSquintRight: 0.5, noseSneerLeft: 0.4, noseSneerRight: 0.4 },
    ];
    idleShapes = exprs[Math.floor(Math.random() * exprs.length)];
    setTimeout(() => {
      idleShapes  = {};
      microActive = false;
      nextMicro   = Date.now() + 2000 + Math.random() * 6000;
    }, 400 + Math.random() * 1400);
  }

  // ── WebSocket ──────────────────────────────────────────────────
  const WS_URL   = `ws://${location.host}/ws`;
  const subtitle = document.getElementById('subtitle');
  let ws, subtitleTimer;

  function connect() {
    ws = new WebSocket(WS_URL);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.type === 'amplitude') {
        amplitudeShapes = { jawOpen: Math.min(1.0, msg.value * 1.1) };
      }

      if (msg.type === 'speaking' && !msg.value) {
        amplitudeShapes = {};
        targetShapes    = {};
      }

      if (msg.type === 'viseme') {
        const raw = msg.shapes || {};
        if (Object.keys(raw).length === 0) {
          targetShapes = {};
        } else {
          const SCALE = 1.3;
          targetShapes = Object.fromEntries(
            Object.entries(raw).map(([k, v]) => [k, Math.min(1.0, v * SCALE)])
          );
          lastVisemeAt = Date.now();
        }
      }

      if (msg.type === 'mood_change') {
        targetSubtitleColor.set(msg.color);
        GlitchEngine.setIntensity(msg.glitch);
      }

      if (msg.type === 'text') {
        subtitle.textContent = msg.value;
        subtitle.style.opacity = '1';
        clearTimeout(subtitleTimer);
        subtitleTimer = setTimeout(() => { subtitle.style.opacity = '0'; }, 4000);
      }
    };
    ws.onclose = () => setTimeout(connect, 2000);
  }

  connect();

  // ── Render loop ────────────────────────────────────────────────
  function animate() {
    requestAnimationFrame(animate);

    if (!modelRoot) { renderer.render(scene, camera); return; }

    lerpMorphTargets();
    tickBlink();
    tickSaccade();
    tickMicroExpression();

    // Subtitle color lerp
    subtitleColor.lerp(targetSubtitleColor, 0.06);
    subtitle.style.color = `#${subtitleColor.getHexString()}`;

    modelRoot.rotation.y = Math.sin(Date.now() * 0.0003) * 0.08;

    if (glitchMat) GlitchEngine.tick(modelRoot, glitchMat);

    renderer.render(scene, camera);
  }

  animate();

  window.addEventListener('resize', () => {
    const w = window.innerWidth, h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
})();

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { GlitchEngine } from './glitch.js';

(function () {
  const canvas = document.getElementById('face-canvas');
  // In vertical mode the screen is portrait but we render LANDSCAPE (swapped dims)
  // and CSS rotates #stage 90° — so the face is framed correctly on a turned TV.
  function renderDims() {
    return document.body.classList.contains('vertical')
      ? { w: window.innerHeight, h: window.innerWidth }
      : { w: window.innerWidth,  h: window.innerHeight };
  }
  let { w: W, h: H } = renderDims();
  canvas.width = W; canvas.height = H;

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
  let currentMood = 'hostile';
  let lastAudioAt = 0;
  const _freqBuf = new Uint8Array(256);

  // Blend shape layers (merged in priority order, higher = wins)
  const BASE_SHAPES    = { browDownLeft: 0.3, browDownRight: 0.3 };
  let amplitudeShapes  = {};
  let idleShapes       = {};
  let expressionShapes = {};   // sustained emotional expression from LLM response
  let gazeShapes       = {};
  let blinkShapes      = {};
  const currentShapes  = {};

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
  loader.setMeshoptDecoder(MeshoptDecoder);

  const texLoader = new THREE.TextureLoader();
  function loadTex(path, colorSpace = THREE.SRGBColorSpace) {
    const t = texLoader.load(path, undefined, undefined,
      (e) => { console.warn('[FACE] tex error:', path, e); window.__showErr && window.__showErr('TEX FAIL: ' + path); });
    t.colorSpace = colorSpace;
    t.flipY = false;
    return t;
  }

  // ── ORACLE caption auto-hide ───────────────────────────────────
  // Driven by ACTUAL audio playback (window._playbackEndsAt, set in index.html),
  // not the flaky speaking event: when playback truly ends, fade the caption.
  let captionWasPlaying = false;
  function isOraclePlaying() {
    return Date.now() < (window._playbackEndsAt || 0);
  }
  function captionTick() {
    if (!isOraclePlaying() && captionWasPlaying) {
      if (subtitleOracle) subtitleOracle.style.opacity = '0';  // hide caption when audio ends
    }
    captionWasPlaying = isOraclePlaying();
    setTimeout(captionTick, 120);   // keep polling for the next utterance
  }

  loader.load('models/52shapes_v1_meshopt.glb', (gltf) => {
    const albedoTex = loadTex('models/textures_comp/albedo_comp.jpg');
    const normalTex = loadTex('models/textures_comp/normal_comp.jpg', THREE.LinearSRGBColorSpace);
    const cavityTex = loadTex('models/textures_comp/cavity_comp.jpg', THREE.LinearSRGBColorSpace);

    const headMat = new THREE.MeshStandardMaterial({
      map:             albedoTex,
      normalMap:       normalTex,
      aoMap:           cavityTex,
      aoMapIntensity:  1.0,
      metalness:       0.0,
      roughness:       0.6,
      envMapIntensity: 2.4,
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
      color: 0x000000, metalness: 0.0, roughness: 0.0, envMapIntensity: 2.0,
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
    const scale  = 3.3 / maxDim;   // a bit bigger on screen
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

    captionTick();             // start the playback-driven caption auto-hide loop
  }, undefined, err => { console.error('[FACE] Load error:', err); window.__showErr && window.__showErr('MODEL LOAD FAIL: ' + (err && err.message || err)); });

  // ── Blend shape helpers ────────────────────────────────────────
  // Mouth/jaw shapes need faster lerp to track per-syllable viseme timing (~100ms/syllable)
  const FAST_SHAPES = new Set([
    'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthPucker',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthShrugLower', 'mouthShrugUpper',
    'cheekSquintLeft', 'cheekSquintRight',
  ]);

  function lerpMorphTargets() {
    if (!headMesh?.morphTargetDictionary) return;
    const dict = headMesh.morphTargetDictionary;

    const merged = { ...BASE_SHAPES, ...idleShapes, ...expressionShapes, ...gazeShapes, ...amplitudeShapes, ...blinkShapes };

    for (const key of Object.keys(dict)) {
      const target = isFinite(merged[key]) ? merged[key] : 0;
      const cur    = isFinite(currentShapes[key]) ? currentShapes[key] : 0;
      const rate   = FAST_SHAPES.has(key) ? 0.40 : 0.22;
      currentShapes[key] = cur + (target - cur) * rate;
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

  // ── Mood expressions ───────────────────────────────────────────
  const MOOD_EXPRESSIONS = {
    hostile: [
      { browDownLeft: 1.0, browDownRight: 1.0, noseSneerLeft: 0.7, eyeSquintLeft: 0.5, eyeSquintRight: 0.5 },
      { browDownLeft: 0.8, noseSneerLeft: 0.9, mouthFrownLeft: 0.6, mouthFrownRight: 0.3 },
      { eyeSquintLeft: 0.9, eyeSquintRight: 0.9, browDownLeft: 0.7, browDownRight: 0.6 },
      { noseSneerLeft: 0.8, noseSneerRight: 0.4, browDownLeft: 0.9, browDownRight: 0.7 },
      { mouthFrownLeft: 0.7, mouthFrownRight: 0.7, browDownLeft: 0.6, eyeSquintLeft: 0.4 },
    ],
    friendly: [
      { mouthSmileLeft: 0.5, mouthSmileRight: 0.5, cheekSquintLeft: 0.4, cheekSquintRight: 0.4 },
      { browInnerUp: 0.4, eyeWideLeft: 0.25, eyeWideRight: 0.25, mouthSmileLeft: 0.3, mouthSmileRight: 0.3 },
      { cheekSquintLeft: 0.5, cheekSquintRight: 0.5, mouthSmileLeft: 0.2, mouthSmileRight: 0.2 },
      { browOuterUpLeft: 0.5, browOuterUpRight: 0.5, mouthSmileLeft: 0.4, mouthSmileRight: 0.4 },
    ],
    surreal: [
      { browInnerUp: 1.0, eyeWideLeft: 0.8, eyeWideRight: 0.7, mouthFunnel: 0.3 },
      { eyeLookUpLeft: 0.9, eyeLookUpRight: 0.8, browInnerUp: 0.7 },
      { mouthFunnel: 0.5, eyeSquintLeft: 0.3, noseSneerLeft: 0.4, browInnerUp: 0.5 },
      { jawForward: 0.3, browInnerUp: 0.8, eyeWideLeft: 0.6, eyeWideRight: 0.4 },
      { eyeLookInLeft: 0.6, eyeLookInRight: 0.6, browDownLeft: 0.3, mouthFunnel: 0.2 },
    ],
    paranoid: [
      { eyeWideLeft: 0.9, eyeWideRight: 0.9, browInnerUp: 0.7, mouthFrownLeft: 0.3 },
      { eyeLookInLeft: 0.8, eyeLookInRight: 0.8, browDownLeft: 0.4, mouthFrownLeft: 0.4 },
      { eyeLookOutLeft: 0.9, browInnerUp: 0.9, mouthFrownRight: 0.3, eyeWideLeft: 0.5 },
      { browDownLeft: 0.6, browDownRight: 0.7, eyeSquintLeft: 0.5, eyeSquintRight: 0.4 },
    ],
    dismissive: [
      { browDownLeft: 0.6, browDownRight: 0.2, mouthFrownLeft: 0.5, eyeSquintLeft: 0.6 },
      { eyeSquintLeft: 0.8, eyeSquintRight: 0.3, noseSneerLeft: 0.5, browDownLeft: 0.5 },
      { browOuterUpLeft: 0.7, browDownRight: 0.6, mouthFrownRight: 0.4 },
      { browDownLeft: 0.9, browDownRight: 0.5, mouthFrownLeft: 0.4, eyeSquintLeft: 0.3 },
    ],
    philosophical: [
      { browInnerUp: 0.8, eyeLookUpLeft: 0.5, eyeLookUpRight: 0.5, mouthFrownLeft: 0.2 },
      { browInnerUp: 1.0, eyeWideLeft: 0.4, eyeWideRight: 0.4, mouthFunnel: 0.2 },
      { eyeLookDownLeft: 0.5, eyeLookDownRight: 0.5, browInnerUp: 0.6 },
      { cheekSquintLeft: 0.3, eyeSquintLeft: 0.4, browDownLeft: 0.3, browInnerUp: 0.4 },
    ],
  };

  // ── Micro-expressions ──────────────────────────────────────────
  function tickMicroExpression() {
    if (microActive || Date.now() < nextMicro) return;
    microActive = true;
    const exprs = MOOD_EXPRESSIONS[currentMood] || MOOD_EXPRESSIONS.hostile;
    idleShapes = exprs[Math.floor(Math.random() * exprs.length)];
    const hold = 350 + Math.random() * 1000;
    setTimeout(() => {
      idleShapes  = {};
      microActive = false;
      nextMicro   = Date.now() + 800 + Math.random() * 4000;
    }, hold);
  }

  // ── WebSocket ──────────────────────────────────────────────────
  const WS_PROTO = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const WS_URL   = `${WS_PROTO}//${location.host}/ws`;
  const subtitle = document.getElementById('subtitle');               // user (blue)
  const subtitleOracle = document.getElementById('subtitle-oracle');  // ORACLE (green)
  let ws, subtitleTimer, subtitleOracleTimer;

  function connect() {
    ws = new WebSocket(WS_URL);
    // Keep-alive ping every 25s so iOS/proxies don't drop idle WebSocket
    const pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
      else clearInterval(pingInterval);
    }, 25000);
    ws.onclose = () => { clearInterval(pingInterval); setTimeout(connect, 2000); };
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.type === 'speaking') {
        window.isMuted = msg.value;
        if (!msg.value) {
          // Speech ended → clear sustained expression so face returns to baseline.
          // (Caption hide is driven by real playback in glitchTick, not this flaky event.)
          expressionShapes = {};
        }
      }

      if (msg.type === 'expression') {
        expressionShapes = msg.shapes || {};
      }

      if (msg.type === 'mood_change') {
        currentMood = msg.mood || 'hostile';
        targetSubtitleColor.set(msg.color);
        GlitchEngine.setIntensity(msg.glitch);
      }

      if (msg.type === 'text') {
        subtitle.textContent = msg.value;
        subtitle.style.opacity = '1';
        clearTimeout(subtitleTimer);
        subtitleTimer = setTimeout(() => { subtitle.style.opacity = '0'; }, 4000);
      }

      // What the face SAYS (ORACLE's own words) — green line above. Stays up
      // while ORACLE talks; the speaking=false handler fades it when speech ends.
      // Long fallback in case that event is ever missed.
      if (msg.type === 'caption') {
        subtitleOracle.textContent = msg.value;
        subtitleOracle.style.opacity = '1';
        clearTimeout(subtitleOracleTimer);
        subtitleOracleTimer = setTimeout(() => { subtitleOracle.style.opacity = '0'; }, 30000);
      }
    };
  }

  connect();


  // ── Render loop ────────────────────────────────────────────────
  function _band(lo, hi) {
    const s = Math.max(0, Math.floor(lo / 46.875));
    const e = Math.min(255, Math.ceil(hi / 46.875));
    let sum = 0;
    for (let i = s; i <= e; i++) sum += _freqBuf[i];
    return sum / ((e - s + 1) * 255);
  }

  function tickLocalVoice() {
    const analyser = window._ttsAnalyser;
    if (!analyser) return;
    analyser.getByteFrequencyData(_freqBuf);

    const fund  = _band(80,   300);
    const f1    = _band(300,  900);
    const f2    = _band(900,  2500);
    const fric  = _band(2500, 7000);
    const total = fund * 0.5 + f1 * 0.8 + f2 * 0.3 + fric * 0.2;

    // Prefer server visemes (per-character, accurate). A fresh {} from the
    // schedule closes mouth between words/sentences; truthy in JS so it wins.
    const viseme = window._currentViseme;
    if (viseme !== null && Date.now() - window._lastVisemeAt < 250) {
      amplitudeShapes = viseme;
      if (total > 0.04) lastAudioAt = Date.now();
      return;
    }

    if (total > 0.04) {
      lastAudioAt = Date.now();
      // FFT fallback (only when server visemes haven't arrived yet)
      const jaw   = Math.min(0.42, (fund * 0.5 + f1 * 1.4) * 0.70);
      const f2n   = Math.min(1.0, f2 * 2.8) * Math.max(0, 1 - fund);
      const round = Math.min(1.0, f1 * 2.5) * Math.max(0, 1 - f2 * 1.5);
      amplitudeShapes = {
        jawOpen:             jaw,
        mouthSmileLeft:      f2n  * 0.45,
        mouthSmileRight:     f2n  * 0.45,
        mouthFunnel:         round * 0.50,
        mouthPucker:         round * 0.30,
        mouthStretchLeft:    Math.min(0.30, fric * 3.5),
        mouthStretchRight:   Math.min(0.30, fric * 3.5),
        mouthLowerDownLeft:  jaw  * 0.40,
        mouthLowerDownRight: jaw  * 0.40,
        mouthUpperUpLeft:    jaw  * 0.20,
        mouthUpperUpRight:   jaw  * 0.20,
      };
    } else if (Date.now() - lastAudioAt > 300) {
      // Real audio silence for 300ms → close mouth and reset viseme state
      amplitudeShapes = {};
      window._currentViseme = null;
    }
  }

  function animate() {
    requestAnimationFrame(animate);

    if (!modelRoot) { renderer.render(scene, camera); return; }

    tickLocalVoice();
    lerpMorphTargets();
    tickBlink();
    tickSaccade();
    tickMicroExpression();

    // Subtitle colors are fixed in CSS now (user=blue, ORACLE=green).

    modelRoot.rotation.y = Math.sin(Date.now() * 0.0003) * 0.08;

    if (glitchMat) GlitchEngine.tick(modelRoot, glitchMat);

    renderer.render(scene, camera);
  }

  animate();

  function applyOrientation() {
    const { w, h } = renderDims();
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window._applyOrientation = applyOrientation;   // called by setup when orientation is chosen
  window.addEventListener('resize', applyOrientation);
})();

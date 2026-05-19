(function () {
  const canvas   = document.getElementById('face-canvas');
  const W = canvas.width  = window.innerWidth;
  const H = canvas.height = window.innerHeight;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false });
  renderer.setSize(W, H);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 100);
  camera.position.set(0, 0.2, 3.5);

  // ── Materials ──────────────────────────────────────────────────
  const lineMat = new THREE.LineBasicMaterial({ color: 0x00ffff });

  function makeWireframe(geo) {
    return new THREE.LineSegments(new THREE.WireframeGeometry(geo), lineMat.clone());
  }

  // ── Head (skull) ───────────────────────────────────────────────
  const headGeo = new THREE.SphereGeometry(1, 10, 7);
  const head    = makeWireframe(headGeo);
  head.scale.set(1, 1.25, 0.88);
  head.position.y = 0.3;
  scene.add(head);

  // ── Eyes ───────────────────────────────────────────────────────
  const eyeGeo = new THREE.SphereGeometry(0.18, 6, 4);
  [-0.38, 0.38].forEach(x => {
    const eye = makeWireframe(eyeGeo);
    eye.position.set(x, 0.45, 0.82);
    head.add(eye);
  });

  // ── Jaw (pivot at top of jaw so it rotates open from hinge) ────
  const jawPivot = new THREE.Group();
  jawPivot.position.set(0, -0.55, 0);
  scene.add(jawPivot);

  const jawGeo = new THREE.BoxGeometry(1.4, 0.45, 0.9);
  const jaw    = makeWireframe(jawGeo);
  jaw.position.y = -0.22;
  jawPivot.add(jaw);

  // ── Neck ───────────────────────────────────────────────────────
  const neckGeo = new THREE.CylinderGeometry(0.28, 0.34, 0.5, 6, 1, true);
  const neck    = makeWireframe(neckGeo);
  neck.position.y = -1.25;
  scene.add(neck);

  // ── State ──────────────────────────────────────────────────────
  let targetJawAngle  = 0;
  let currentJawAngle = 0;
  let targetColor     = new THREE.Color(0x00ffff);
  let currentColor    = new THREE.Color(0x00ffff);

  function setColor(hex) {
    targetColor.set(hex);
  }

  function applyColorToScene(color) {
    [head, jawPivot, neck].forEach(obj => {
      obj.traverse(child => {
        if (child.material) child.material.color.copy(color);
      });
    });
    document.getElementById('subtitle').style.color = `#${color.getHexString()}`;
  }

  // ── WebSocket ──────────────────────────────────────────────────
  const WS_URL = `ws://${location.host}/ws`;
  let ws;
  const subtitle = document.getElementById('subtitle');
  let subtitleTimer;

  function connect() {
    ws = new WebSocket(WS_URL);

    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.type === 'amplitude') {
        targetJawAngle = msg.value * 0.55;
      }

      if (msg.type === 'speaking' && !msg.value) {
        targetJawAngle = 0;
      }

      if (msg.type === 'mood_change') {
        setColor(msg.color);
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

    // Smooth jaw
    currentJawAngle += (targetJawAngle - currentJawAngle) * 0.35;
    jawPivot.rotation.x = currentJawAngle;

    // Smooth color
    currentColor.lerp(targetColor, 0.06);
    applyColorToScene(currentColor);

    // Subtle idle rotation
    head.rotation.y = Math.sin(Date.now() * 0.0003) * 0.12;
    jawPivot.rotation.y = head.rotation.y;

    GlitchEngine.tick(head, lineMat);

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

export const GlitchEngine = (function () {
  let intensity = 0.1;
  let glitchTimer = 0;
  const GLITCH_INTERVAL_BASE = 2500; // ms between glitch bursts at intensity 1.0

  const originalPositions = new WeakMap();

  function saveOriginal(mesh) {
    if (!originalPositions.has(mesh)) {
      const pos = mesh.position.clone();
      originalPositions.set(mesh, pos);
    }
  }

  function restorePosition(mesh) {
    const orig = originalPositions.get(mesh);
    if (orig) mesh.position.copy(orig);
  }

  function triggerGlitch(mesh) {
    const orig = originalPositions.get(mesh);
    if (!orig) return;

    const burst = 3 + Math.floor(Math.random() * 4);
    for (let i = 0; i < burst; i++) {
      const delay = i * (30 + Math.random() * 40);
      const restore = delay + 20 + Math.random() * 30;

      setTimeout(() => {
        mesh.position.set(
          orig.x + (Math.random() - 0.5) * intensity * 0.25,
          orig.y + (Math.random() - 0.5) * intensity * 0.08,
          orig.z,
        );
      }, delay);

      setTimeout(() => {
        restorePosition(mesh);
      }, restore);
    }
  }

  function triggerColorGlitch(mat) {
    const orig = mat.color.clone();
    const flashCount = 2 + Math.floor(Math.random() * 3);
    for (let i = 0; i < flashCount; i++) {
      setTimeout(() => {
        mat.color.set(Math.random() > 0.5 ? 0xffffff : 0xff0000);
      }, i * 40);
      setTimeout(() => {
        mat.color.copy(orig);
      }, i * 40 + 20);
    }
  }

  return {
    setIntensity(v) {
      intensity = Math.max(0, Math.min(1, v));
    },

    tick(mesh, mat) {
      if (intensity < 0.05) return;
      saveOriginal(mesh);

      glitchTimer += 16; // ~60fps
      const interval = GLITCH_INTERVAL_BASE / intensity;

      if (glitchTimer >= interval) {
        glitchTimer = 0;
        if (Math.random() < intensity) {
          triggerGlitch(mesh);
        }
        if (Math.random() < intensity * 0.4) {
          triggerColorGlitch(mat);
        }
      }
    },
  };
})();

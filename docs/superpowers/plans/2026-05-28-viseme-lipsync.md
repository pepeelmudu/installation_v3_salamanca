# Viseme Lip Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace broken FFT-based mouth animation with accurate per-character server visemes from ElevenLabs alignment data, so the face closes its mouth for consonants (P/B/M/T/D/S) and opens correctly for each vowel (A/E/I/O/U).

**Architecture:** ElevenLabs `stream_with_timestamps` already returns per-character start times. `tts_client.py` already has `CHAR_VISEMES` (ARKit blend shapes per Spanish character) and already calls `_schedule_visemes` → `on_viseme` → `broadcast({type:"viseme", shapes})`. The only missing piece is that `face.js` ignores these `viseme` events. We add the handler, remove the FFT approach, and add a timing offset on the server to compensate for browser audio buffering.

**Tech Stack:** JavaScript (Web Audio API, Three.js morph targets), Python (asyncio, ElevenLabs SDK)

---

### Task 1: Add viseme WebSocket handler in face.js

The server already sends `{type: "viseme", shapes: {"jawOpen": 0.5, ...}}` via `/ws`. `face.js` handles `speaking`, `mood_change`, `text` but silently ignores `viseme`. This task wires it up.

**Files:**
- Modify: `face/face.js` — ws.onmessage block (around line 273)

- [ ] **Step 1: Add viseme handler in ws.onmessage**

In `face/face.js`, find the `ws.onmessage` block and add the viseme case:

```javascript
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);

      if (msg.type === 'speaking') {
        window.isMuted = msg.value;
      }

      if (msg.type === 'viseme') {
        amplitudeShapes = msg.shapes || {};
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
    };
```

- [ ] **Step 2: Remove FFT logic from tickLocalVoice — replace with isMuted gate only**

The FFT approach conflicts with server visemes (both write to `amplitudeShapes`). Replace `tickLocalVoice` entirely:

```javascript
  function tickLocalVoice() {
    // Server visemes drive amplitudeShapes via ws.onmessage.
    // Clear shapes immediately when entity stops speaking.
    if (!window.isMuted) {
      amplitudeShapes = {};
    }
  }
```

Also remove the now-unused variables. Find and remove these lines near the top of the IIFE:
```javascript
  const _freqBuf = new Uint8Array(256); // FFT frequency bins
```
And remove the `_band()` helper function (it's only used by the old tickLocalVoice):
```javascript
  // ── Render loop ────────────────────────────────────────────────
  // FFT bands at 24000 Hz sample rate, fftSize 512, 256 bins, ~47 Hz/bin
  function _band(lo, hi) {
    const s = Math.max(0, Math.floor(lo / 46.875));
    const e = Math.min(255, Math.ceil(hi / 46.875));
    let sum = 0;
    for (let i = s; i <= e; i++) sum += _freqBuf[i];
    return sum / ((e - s + 1) * 255);
  }
```

- [ ] **Step 3: Add FAST_SHAPES lerp — mouth shapes respond faster than brow/eye shapes**

Visemes fire per character (~100ms apart at normal speech rate). With lerp 0.25 at 60fps, a jaw transition takes ~200ms to reach 75% — too slow to track syllable timing. Mouth shapes need a faster rate (~0.40) while expression shapes (browDown, eyeSquint) stay at 0.22 for natural blending.

Replace the single `rate = 0.25` in `lerpMorphTargets` with per-category rates:

```javascript
  // Mouth/jaw shapes need faster lerp to track per-syllable viseme timing
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

    const merged = { ...BASE_SHAPES, ...idleShapes, ...gazeShapes, ...amplitudeShapes, ...blinkShapes };

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
```

`FAST_SHAPES` must be declared **above** `lerpMorphTargets` (not inside it) so it's computed once, not every frame.

- [ ] **Step 4: Commit**

```bash
git add face/face.js
git commit -m "feat: wire server visemes to face, remove FFT lip sync"
```

---

### Task 2: Add browser audio offset to server viseme timing

The server fires viseme events at the moment it sends audio bytes. The browser receives audio → queues it with `Math.max(playCtx.currentTime + 0.05, scheduledUntil)` — so audio plays ~50-150ms AFTER it arrives. Visemes arrive almost simultaneously with audio but fire immediately, appearing early.

Fix: add a constant offset to all `loop.call_later` delays in `_schedule_visemes`.

**Files:**
- Modify: `tts_client.py` — `_do_schedule` inner function (around line 271)

- [ ] **Step 1: Add the offset constant and apply it**

In `tts_client.py`, find `_do_schedule` inside `_schedule_visemes`:

```python
    async def _do_schedule() -> None:
        loop = asyncio.get_running_loop()
        for delay, shapes in events:
            loop.call_later(
                delay,
                lambda s=shapes: asyncio.run_coroutine_threadsafe(
                    self._on_viseme(s), self._loop
                ),
            )
```

Replace with:

```python
    _BROWSER_AUDIO_OFFSET = 0.12  # 120ms: compensates for browser audio buffer + network

    async def _do_schedule() -> None:
        loop = asyncio.get_running_loop()
        for delay, shapes in events:
            loop.call_later(
                delay + _BROWSER_AUDIO_OFFSET,
                lambda s=shapes: asyncio.run_coroutine_threadsafe(
                    self._on_viseme(s), self._loop
                ),
            )
```

`_BROWSER_AUDIO_OFFSET` is a local variable inside `_schedule_visemes`, which is fine — it's only used here. If timing feels early or late after testing, adjust this value: +50ms if visemes appear before the sound, -50ms if they appear after.

- [ ] **Step 2: Commit and push**

```bash
git add tts_client.py
git commit -m "fix: add 120ms browser audio offset to viseme timing"
git push
```

---

### Task 3: Verify and tune

- [ ] **Step 1: Deploy and test on device**

Wait for Render to redeploy. Open the face URL. Speak to it. Watch for:
- **Mouth closes** for P, B, M sounds (e.g. "muy bien", "para", "más")
- **Wide jaw open** for A sounds ("habla", "para")
- **Smile shape** for I/E sounds ("sí", "te")
- **Rounded** for O/U sounds ("cómo", "tu")
- **Mouth closes between sentences** (not hanging open)

- [ ] **Step 2: Tune offset if needed**

If visemes look **too early** (mouth shape appears before the sound): increase `_BROWSER_AUDIO_OFFSET` to `0.18`.
If visemes look **too late** (sound plays before mouth moves): decrease to `0.06`.

Edit `tts_client.py`, commit, push.

- [ ] **Step 3: Tune CHAR_VISEMES if specific sounds look wrong**

Current values in `tts_client.py` CHAR_VISEMES to check:

| Character | Current | Notes |
|-----------|---------|-------|
| `'a'` | `jawOpen: 0.8` | May be too wide — try 0.65 if it looks extreme |
| `'p','b'` | `mouthClose: 0.9` | Good — bilabial closure |
| `'m'` | `mouthClose: 1.0` | Good — full lip seal |
| `'s','z'` | `jawOpen: 0.2, mouthStretch: 0.2` | Good |
| `'i'` | `jawOpen: 0.15, mouthSmile: 0.6` | Good |

If `jawOpen: 0.8` for 'a' looks dislocated, change to `0.60` in `tts_client.py`.

```bash
git add tts_client.py
git commit -m "tune: adjust CHAR_VISEMES values for 'a'"
git push
```

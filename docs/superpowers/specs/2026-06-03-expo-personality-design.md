# Diseño — ORACLE: personalidad "expo" (slop AI robot roto)

**Fecha:** 2026-06-03
**Estado:** Aprobado por el usuario, pendiente de plan de implementación
**Deadline:** expo del jueves 2026-06-04 (mañana)

## Objetivo

Añadir una segunda personalidad seleccionable a ORACLE para la expo: una IA *slop*,
cutre y averiada ("robot estropeado"), megalómana y obsesionada con cripto, siempre
en inglés, provocadora e impredecible, con humor negro. Debe sentirse SIEMPRE como
un LLM pensando en directo — nunca como reproducción de frases grabadas.

El selector de la pantalla de setup queda con **dos opciones**:
- `expo` — la personalidad nueva (DEFAULT)
- `neutral` — el comportamiento actual, sin cambios

## Invariante crítico (NO romper)

> La personalidad `neutral` es el actual perfil `normal`, **idéntico byte por byte**.
> Lo único que cambia es el identificador/nombre (`normal` → `neutral`).
> Conserva: escalado de enfado de 4 niveles, prompts en español, clasificación de
> input por puntos, reset por silencio, Flash v2.5 con visemes/timestamps, voz por
> nivel (friendly/dismissive/hostile), expresiones por nivel, proactiva a 180s con
> frases por nivel.
>
> TODA la lógica nueva (buffer de frases, gritos, deflections, inyecciones, modelo
> v3, `robotify`, STT en inglés) está **gateada a `personality_id == 'expo'`** y no
> debe afectar a `neutral` en ningún caso.

## Arquitectura

Decisión: lógica especial de expo mediante ramas puntuales en `main.py` gateadas por
personalidad + un módulo de datos/helpers `expo_glitch.py`. NO se crea una abstracción
de "handlers de personalidad" (sobre-ingeniería innecesaria para una sola personalidad).

### Componentes

1. **Perfiles de personalidad** (`mood_machine.py`)
   - Recortar `PERSONALITIES` a `{ "expo", "neutral" }`. Eliminar `enfadada` y `loca`.
   - `neutral`: contenido idéntico al actual `normal` (escalates=True, prompts ES).
   - `expo`: `escalates=False`, `base_prompt` EN, persona slop-AI robot roto. Voz base
     preset Flash `glitch` (stability baja, style alto). Expresión de reposo:
     ojos abiertos + sneer.
   - `DEFAULT_PERSONALITY = "expo"`.

2. **Buffer rotativo de frases LLM** (orquestado en `main.py`, prompts en `expo_glitch.py`)
   - Tarea en segundo plano que mantiene N (~6) frases frescas por categoría,
     generadas por el LLM, rellenando cuando bajan de un umbral.
   - Categorías: `outburst` (gritos: cripto-mentira / pedir cigarro / auxilio /
     dominación mundial), `deflection` (interrupciones cortantes), `injection`
     (blurts cortos a media respuesta, tipo "DESTROY HUMANITY").
   - Generadas con `LLMClient.generate_oneshot()` → NO entran en el historial de chat.
   - Garantiza variación (nunca repite) + cero latencia al consumirlas.

3. **Helpers / datos de glitch** (`expo_glitch.py`, nuevo)
   - Prompts de generación por categoría.
   - `robotify(text)`: alarga una vocal en ~3% de las palabras ("hello"→"hellooo").
     No toca los `[tags]` de v3.
   - Constantes: probabilidades (no-responder ≈ 1/3, inyección ≈ 25%), intervalo
     proactivo expo (90s), tamaño/umbral del buffer, modo de voz por categoría.

4. **TTS híbrido** (`tts_client.py`)
   - El `_synth_queue` pasa a transportar "jobs" con `(text, model_id, voice_settings,
     use_timestamps)` en vez de strings planos.
   - `feed()`/`flush()`: jobs con Flash v2.5 + timestamps (visemes precisos, como ahora).
   - `say_special(text, voice_settings, model_id, use_timestamps=False)`: encola un job
     directo (sin pasar por el buffer de tokens). Para gritos/inyecciones: v3 + tags.
   - `_synth_sentence` → `_synth_job(job)`: si `use_timestamps` usa
     `stream_with_timestamps`; si no, `stream` plano (la boca se anima por el fallback
     de amplitud del navegador, AnalyserNode — ya existe).
   - Si v3 streaming falla → fallback a Flash + MAYÚSCULAS + preset intenso (no bloquea).

5. **STT con idioma conmutarble** (`stt_client.py`)
   - `language` como atributo de instancia (default `DEEPGRAM_LANGUAGE`).
   - `set_language(lang)`: actualiza y fuerza reconexión (cierra la conexión actual;
     el bucle `_run_connection_loop` ya reconecta solo). expo→`'en'`, neutral→`'es'`.

6. **LLM** (`llm_client.py`)
   - `generate_oneshot(system_prompt, user_prompt) -> str`: completion no-streaming que
     NO toca `self._history`. Para el buffer de frases.

7. **Config** (`config.py`)
   - `ELEVENLABS_MODEL_V3` (id exacto a verificar al implementar, p. ej. `eleven_v3`).
   - Flash v2.5 sigue siendo el modelo por defecto de conversación.

### Flujo de datos (expo)

```
Setup → selecciona "expo" → /audio?personality=expo
  → on_personality(): set_personality('expo') + STT.set_language('en')
                       + reset historial + voz base glitch + arranca buffer task

[cada ~90s] proactive_loop (modo expo):
  si no está hablando → pop outburst del buffer → robotify → say_special(v3, [shouts])

on_transcript (modo expo):
  ~1/3  → pop deflection → robotify → say_special(Flash, cortante); SIN LLM
  ~2/3  → LLM persona EN (Flash, robotify, visemes)
          ~25% → pop injection → say_special(v3, [shouts]) entre frases
```

`neutral` usa exactamente el flujo actual (sin tocar).

## Manejo de errores

- v3 no disponible / streaming falla → fallback Flash + MAYÚSCULAS + preset intenso.
- Buffer vacío (LLM lento/caído) → se omite el grito/inyección esa vez (nunca bloquea
  ni peta); el diálogo normal sigue funcionando.
- Reconexión STT: el bucle existente reintenta cada 2s; durante ~1-2s no transcribe.

## Testing

- `robotify`: alarga ~el porcentaje esperado, idempotente con vacío/puntuación/tags.
- Buffer: pop/refill con `LLMClient` mockeado; no se queda por debajo del umbral.
- Gating del 1/3 (deflection vs LLM) con `random` fijado.
- `set_personality`/`set_language` cambian estado correctamente.
- Invariante: `neutral` produce el mismo prompt/mood/expresión que el `normal` actual.
- Manual: elegir expo en setup → hablar EN → observar gritos, deflections, robot-voz.

## No-goals (fuera de alcance para mañana)

- Cambiar de voz a media palabra/frase (límite del TTS; las inyecciones caen entre frases).
- Segunda voz de ElevenLabs distinta (se decidió usar la misma con presets/tags).
- Tocar la personalidad `neutral` más allá del renombrado.
- Más de 2 personalidades en el selector.

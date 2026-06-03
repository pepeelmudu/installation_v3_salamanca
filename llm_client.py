from groq import Groq
from typing import Iterator
from config import MAX_HISTORY_MESSAGES, GROQ_MAX_TOKENS


class LLMClient:
    def __init__(self, api_key: str, model: str, fallback_models: list[str] | None = None):
        self._groq = Groq(api_key=api_key)
        self._model = model
        # Ordered model chain: primary first, then fallbacks tried on rate-limit.
        self._models = [model] + list(fallback_models or [])
        self._history: list[dict] = []

    @staticmethod
    def _is_rate_limit(e: Exception) -> bool:
        return getattr(e, "status_code", None) == 429 or "rate_limit" in str(e).lower()

    def _build_messages(self, user_text: str, system_prompt: str) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._history[-MAX_HISTORY_MESSAGES:])
        messages.append({"role": "user", "content": user_text})
        return messages

    def reset_history(self) -> None:
        """Wipe conversation history. Call when personality tone changes mid-conversation."""
        self._history.clear()

    def _save_exchange(self, user_text: str, assistant_text: str) -> None:
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})
        if len(self._history) > MAX_HISTORY_MESSAGES:
            self._history = self._history[-MAX_HISTORY_MESSAGES:]

    def stream(self, user_text: str, system_prompt: str,
               temperature: float = 1.1, max_tokens: int | None = None) -> Iterator[str]:
        messages = self._build_messages(user_text, system_prompt)
        for idx, model in enumerate(self._models):
            try:
                response = self._groq.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens or GROQ_MAX_TOKENS,
                    temperature=temperature,
                )
            except Exception as e:
                if self._is_rate_limit(e) and idx < len(self._models) - 1:
                    print(f"[LLM] {model} rate-limited → falling back to next model", flush=True)
                    continue
                print(f"[LLM ERROR] {e!r}", flush=True)
                return
            # Committed to this model — stream its tokens.
            full_response = ""
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:
                    full_response += token
                    yield token
            if full_response:
                self._save_exchange(user_text, full_response)
            return

    def generate_oneshot(self, system_prompt: str, user_prompt: str,
                         temperature: float = 1.3, max_tokens: int = 40) -> str:
        """Single non-streaming completion that does NOT touch conversation
        history. Used to pre-generate glitch lines for the expo personality."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for idx, model in enumerate(self._models):
            try:
                resp = self._groq.chat.completions.create(
                    model=model, messages=messages, stream=False,
                    max_tokens=max_tokens, temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if self._is_rate_limit(e) and idx < len(self._models) - 1:
                    continue
                raise
        return ""

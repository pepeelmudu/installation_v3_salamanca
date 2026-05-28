from groq import Groq
from typing import Iterator
from config import MAX_HISTORY_MESSAGES, GROQ_MAX_TOKENS


class LLMClient:
    def __init__(self, api_key: str, model: str):
        self._groq = Groq(api_key=api_key)
        self._model = model
        self._history: list[dict] = []

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

    def stream(self, user_text: str, system_prompt: str) -> Iterator[str]:
        messages = self._build_messages(user_text, system_prompt)
        response = self._groq.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=1.1,
        )
        full_response = ""
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                yield token
        if full_response:
            self._save_exchange(user_text, full_response)

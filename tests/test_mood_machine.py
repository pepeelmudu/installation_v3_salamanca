# tests/test_mood_machine.py
import asyncio
import pytest
from mood_machine import MoodMachine, MOODS

def test_initial_mood_is_valid():
    mm = MoodMachine()
    assert mm.current_mood in MOODS

def test_get_current_prompt_contains_base():
    mm = MoodMachine()
    prompt = mm.get_current_prompt()
    assert "escultura de arte" in prompt
    assert "español" in prompt

def test_get_current_prompt_contains_mood_fragment():
    mm = MoodMachine()
    for mood_id in MOODS:
        mm.current_mood = mood_id
        prompt = mm.get_current_prompt()
        assert MOODS[mood_id]["prompt_fragment"] in prompt

def test_get_current_state_returns_color_and_glitch():
    mm = MoodMachine()
    state = mm.get_current_state()
    assert "color" in state
    assert "glitch" in state
    assert isinstance(state["glitch"], float)

def test_force_change_picks_different_mood():
    mm = MoodMachine()
    original = mm.current_mood
    # Force 20 changes — at least one must differ
    seen = set()
    for _ in range(20):
        mm._pick_next_mood()
        seen.add(mm.current_mood)
    assert len(seen) > 1

@pytest.mark.asyncio
async def test_on_change_callback_called():
    mm = MoodMachine()
    received = []
    async def cb(mood_id, state):
        received.append(mood_id)
    mm._on_change_cb = cb
    mm._pick_next_mood()
    await mm._notify_change()
    assert len(received) == 1
    assert received[0] in MOODS

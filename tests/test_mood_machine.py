# tests/test_mood_machine.py
from mood_machine import (
    AnnoyanceState,
    classify_user_input,
    detect_expression,
    LEVEL_PROMPTS,
    LEVEL_EXPRESSIONS,
    LEVEL_TO_MOOD,
)


def test_initial_state_is_friendly():
    a = AnnoyanceState()
    assert a.points == 0
    assert a.level == 0
    assert a.mood_id == "friendly"


def test_prompt_contains_base_and_level_fragment():
    a = AnnoyanceState()
    prompt = a.get_prompt()
    assert "entidad digital" in prompt
    assert "español" in prompt
    assert LEVEL_PROMPTS[0] in prompt


def test_insult_escalates_one_level():
    a = AnnoyanceState()
    delta, reason = classify_user_input("Eres un gilipollas")
    assert delta == 3
    assert reason == "insulto a la entidad"
    level_change = a.apply(delta, reason)
    assert a.level == 1
    assert level_change == 1


def test_three_insults_reach_enraged():
    a = AnnoyanceState()
    for _ in range(3):
        d, r = classify_user_input("Eres una mierda")
        a.apply(d, r)
    assert a.level == 3
    assert a.mood_id == "hostile"


def test_apology_calms_down():
    a = AnnoyanceState()
    a.apply(*classify_user_input("Eres tonto"))  # +3 → level 1
    a.apply(*classify_user_input("Eres tonto"))  # +3 → level 2
    assert a.level == 2
    a.apply(*classify_user_input("Lo siento mucho"))  # -2 → level 1
    assert a.level == 1


def test_politeness_decrements():
    a = AnnoyanceState()
    a.apply(*classify_user_input("Eres tonto"))   # +3 → 3
    a.apply(*classify_user_input("Por favor"))    # -1 → 2
    assert a.points == 2


def test_points_clamped_to_range():
    a = AnnoyanceState()
    for _ in range(10):
        a.apply(3, "x")
    assert a.points == 10
    for _ in range(20):
        a.apply(-2, "x")
    assert a.points == 0


def test_reset_returns_to_zero():
    a = AnnoyanceState()
    a.apply(5, "x")
    assert a.points > 0
    a.reset()
    assert a.points == 0
    assert a.level == 0


def test_level_to_mood_mapping_covers_all_levels():
    for lvl in range(4):
        assert lvl in LEVEL_TO_MOOD
        assert lvl in LEVEL_PROMPTS
        assert lvl in LEVEL_EXPRESSIONS


def test_neutral_input_no_change():
    a = AnnoyanceState()
    d, r = classify_user_input("Hola, ¿cómo estás?")
    assert d == 0
    assert r == "neutral"


def test_detect_expression_returns_dict():
    # Empty for normal responses (level expression takes over)
    assert detect_expression("Hola.") == {}
    # Philosophical bonus when content matches
    long_phil = "Pienso mucho en la existencia y el tiempo y el sentido de todo."
    out = detect_expression(long_phil)
    assert 'browInnerUp' in out

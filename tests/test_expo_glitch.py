import random
from expo_glitch import robotify


def test_robotify_elongates_when_prob_one():
    rng = random.Random(0)
    out = robotify("hello world", prob=1.0, rng=rng)
    assert "oo" in out  # at least one elongated vowel run
    assert len(out) > len("hello world")


def test_robotify_noop_when_prob_zero():
    assert robotify("hello world", prob=0.0) == "hello world"


def test_robotify_handles_empty_and_punctuation():
    assert robotify("", prob=1.0) == ""
    assert robotify("...", prob=1.0) == "..."   # no vowels → unchanged


def test_robotify_skips_audio_tags():
    rng = random.Random(1)
    out = robotify("[shouts] bitcoin", prob=1.0, rng=rng)
    assert out.startswith("[shouts]")           # tag untouched


from expo_glitch import (
    split_sentences, CATEGORIES, GLITCH_SYSTEM, GLITCH_PROMPTS,
    CATEGORY_VOICE, DEFLECT_PROB, INJECT_PROB, EXPO_PROACTIVE_INTERVAL,
)


def test_split_sentences_basic():
    assert split_sentences("Hi there. Bye now!") == ["Hi there.", "Bye now!"]


def test_split_sentences_single():
    assert split_sentences("just one") == ["just one"]


def test_categories_have_prompts_and_voice():
    assert set(CATEGORIES) == {"outburst", "deflection", "injection"}
    for c in CATEGORIES:
        assert c in GLITCH_PROMPTS and GLITCH_PROMPTS[c]
        assert CATEGORY_VOICE[c] in {"shout", "whisper", "normal"}


def test_probability_constants_in_range():
    assert 0.0 <= DEFLECT_PROB <= 1.0
    assert 0.0 <= INJECT_PROB <= 1.0
    assert EXPO_PROACTIVE_INTERVAL > 0


from expo_glitch import GlitchBuffer


def test_buffer_pop_is_fifo_and_empty_safe():
    b = GlitchBuffer(min_size=2)
    assert b.pop("outburst") is None          # empty → None, never raises
    b.add("outburst", "one")
    b.add("outburst", "two")
    assert b.pop("outburst") == "one"
    assert b.pop("outburst") == "two"


def test_buffer_low_categories():
    b = GlitchBuffer(min_size=2)
    assert set(b.low_categories()) == set(CATEGORIES)   # all empty → all low
    b.add("outburst", "x")
    b.add("outburst", "y")
    assert "outburst" not in b.low_categories()

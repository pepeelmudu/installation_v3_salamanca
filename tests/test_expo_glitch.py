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

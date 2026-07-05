from __future__ import annotations

from video_lance.models import TranscriptWord


def test_transcript_word_basic() -> None:
    w = TranscriptWord(word="hello", start=0.0, end=0.5)
    assert w.word == "hello"
    assert w.start == 0.0
    assert w.end == 0.5


def test_is_sentence_end_period() -> None:
    assert TranscriptWord(word="end.", start=0.0, end=1.0).is_sentence_end


def test_is_sentence_end_question() -> None:
    assert TranscriptWord(word="end?", start=0.0, end=1.0).is_sentence_end


def test_is_sentence_end_exclamation() -> None:
    assert TranscriptWord(word="end!", start=0.0, end=1.0).is_sentence_end


def test_is_sentence_end_false_for_plain_word() -> None:
    assert not TranscriptWord(word="middle", start=0.0, end=1.0).is_sentence_end


def test_is_sentence_end_strips_whitespace() -> None:
    assert TranscriptWord(word="end. ", start=0.0, end=1.0).is_sentence_end
    assert TranscriptWord(word="  end!\n", start=0.0, end=1.0).is_sentence_end

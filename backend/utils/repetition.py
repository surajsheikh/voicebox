"""
Phrase-repetition detector for TTS output.

Fazmo fork addition — catches a distinct failure mode from
``has_tts_runaway`` (utils/audio.py). That detector only recognizes
speech -> long silence gap -> garbled noise, the classic EOS-miss
pattern. Confirmed live (2026-08-09/10): Qwen3-TTS separately gets
stuck looping a short phrase verbatim 15-25+ times with continuous
fluent speech and no silence gap at all — a different failure shape
that has_tts_runaway structurally cannot see, and that a pure audio
duration/RMS heuristic can't reliably catch either without also
flagging legitimately slow, deliberate narration as false positives.

This detector transcribes the chunk (Whisper, already vendored in this
image) and compares word n-gram frequency against the chunk's own
source text. A phrase repeated far more often in the transcript than
it actually appears in the source is a precise, low-false-positive
signal — genuine source text that happens to repeat a phrase (e.g.
intentional rhetorical repetition) is not penalized, since only the
*excess* beyond the source's own count is flagged.

Wired into the same detect-and-retry path as has_tts_runaway in
chunked_tts.py's generate_chunked() — reuses that already-proven retry
mechanism rather than introducing a new one.
"""

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger("voicebox.repetition")

_NGRAM_SIZE = 6
_EXCESS_THRESHOLD = 3
_WHISPER_MODEL = "openai/whisper-base"

_pipe = None


def _get_pipeline():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline as hf_pipeline

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=_WHISPER_MODEL,
            chunk_length_s=30,
            stride_length_s=5,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    return _pipe


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngram_counts(words: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i : i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def has_phrase_repetition(
    audio: np.ndarray,
    sample_rate: int,
    source_text: str,
) -> bool:
    """Detect a phrase repeated far more in the generated audio than it
    actually appears in the source text it was asked to speak.

    Fails safe: any error transcribing or comparing returns False rather
    than raising, so a problem in this detector itself can never trigger
    a retry loop or block generation.
    """
    try:
        pipe = _get_pipeline()
        # Whisper expects float32 mono at whatever sample rate the
        # pipeline's feature extractor resamples internally — passing
        # the raw array + explicit sampling_rate lets it handle that.
        result = pipe(
            {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": sample_rate},
            generate_kwargs={"language": "english"},
        )
        transcript = result["text"] if isinstance(result, dict) else str(result)

        source_words = _words(source_text)
        transcript_words = _words(transcript)
        if len(transcript_words) < _NGRAM_SIZE:
            return False

        source_counts = _ngram_counts(source_words, _NGRAM_SIZE)
        transcript_counts = _ngram_counts(transcript_words, _NGRAM_SIZE)

        for gram, t_count in transcript_counts.items():
            s_count = source_counts.get(gram, 0)
            if t_count - s_count >= _EXCESS_THRESHOLD:
                logger.warning(
                    "Detected phrase repetition: %r appears %d times in output "
                    "vs %d times in source text",
                    " ".join(gram),
                    t_count,
                    s_count,
                )
                return True

        return False
    except Exception:
        logger.exception("Phrase-repetition check failed — treating as not repeated")
        return False

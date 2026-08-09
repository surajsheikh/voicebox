"""
Qwen3-TTS CustomVoice backend implementation.

Wraps the Qwen3-TTS-12Hz CustomVoice model for preset-speaker TTS with
instruction-based style control. Uses the same qwen_tts library as the
Base model (pytorch_backend.py) but loads a different checkpoint and
calls generate_custom_voice() instead of generate_voice_clone().

Key differences from the Base engine:
  - Uses preset speakers (9 built-in voices) instead of zero-shot cloning
  - Supports instruct parameter for tone/emotion/prosody control
  - Two model sizes: 1.7B and 0.6B

Languages supported: zh, en, ja, ko, de, fr, ru, pt, es, it
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import torch
from accelerate.hooks import attach_align_device_hook

from . import TTSBackend, LANGUAGE_CODE_TO_NAME
from .base import (
    is_model_cached,
    get_torch_device,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# ── Preset speakers ──────────────────────────────────────────────────

# (speaker_id, display_name, gender, native_language_code, description)
QWEN_CUSTOM_VOICES = [
    ("Vivian", "Vivian", "female", "zh", "Bright, slightly edgy young female voice"),
    ("Serena", "Serena", "female", "zh", "Warm, gentle young female voice"),
    ("Uncle_Fu", "Uncle Fu", "male", "zh", "Seasoned male voice with a low, mellow timbre"),
    ("Dylan", "Dylan", "male", "zh", "Youthful Beijing male voice with a clear, natural timbre"),
    ("Eric", "Eric", "male", "zh", "Lively Chengdu male voice with a slightly husky brightness"),
    ("Ryan", "Ryan", "male", "en", "Dynamic male voice with strong rhythmic drive"),
    ("Aiden", "Aiden", "male", "en", "Sunny American male voice with a clear midrange"),
    ("Ono_Anna", "Ono Anna", "female", "ja", "Playful Japanese female voice with a light, nimble timbre"),
    ("Sohee", "Sohee", "female", "ko", "Warm Korean female voice with rich emotion"),
]

QWEN_CV_DEFAULT_SPEAKER = "Ryan"

# HuggingFace repo IDs per model size
QWEN_CV_HF_REPOS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}


class QwenCustomVoiceBackend:
    """Qwen3-TTS CustomVoice backend — preset speakers with instruct control."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = self._get_device()
        self._current_model_size: Optional[str] = None

    def _get_device(self) -> str:
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        if model_size not in QWEN_CV_HF_REPOS:
            raise ValueError(f"Unknown model size: {model_size}")
        return QWEN_CV_HF_REPOS[model_size]

    def _is_model_cached(self, model_size: Optional[str] = None) -> bool:
        size = model_size or self.model_size
        return is_model_cached(self._get_model_path(size))

    async def load_model_async(self, model_size: Optional[str] = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility with the TTSBackend protocol
    load_model = load_model_async

    def _load_model_sync(self, model_size: str) -> None:
        model_name = f"qwen-custom-voice-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from qwen_tts import Qwen3TTSModel

            model_path = self._get_model_path(model_size)
            logger.info("Loading Qwen CustomVoice %s on %s...", model_size, self.device)

            if self.device == "cpu":
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
            else:
                # See pytorch_backend.py's _load_model_sync for why
                # device_map is avoided here — Qwen3TTSModel's submodules
                # aren't accelerate-dispatch-aware and are left stranded on
                # the meta device ("Cannot copy out of meta tensor").
                # See pytorch_backend.py's identical fix for the full story
                # — a plain string attn_implementation="flash_attention_2"
                # propagates into the Mimi codec submodule too (not one of
                # this model's registered sub_configs, but still inherited
                # via HF's generic recursive config propagation) and crashes
                # it with "RuntimeError: unsupported scalarType" in ICL mode.
                # Scoping to just the two real sub_configs avoids that.
                try:
                    import flash_attn  # noqa: F401
                    attn_kwargs = {"attn_implementation": {
                        "talker_config": "flash_attention_2",
                        "speaker_encoder_config": "flash_attention_2",
                        "": "eager",
                    }}
                except ImportError:
                    attn_kwargs = {}
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                    **attn_kwargs,
                )
                # Qwen3TTSModel itself is a plain wrapper with no .to() of
                # its own — the real nn.Module is the inner .model.
                self.model.model = self.model.model.to(self.device)
                # See pytorch_backend.py's identical fix for why this hook
                # is needed — without device_map, upstream's generate_*
                # methods that tokenize/build CPU tensors internally have
                # no way to reach the model's device, since they were
                # relying on device_map's own hook to do that silently.
                attach_align_device_hook(self.model.model, execution_device=self.device)
                # See pytorch_backend.py's identical fix for the full story
                # — this is THE actual root cause of multi-minute
                # generations, not the attention implementation. speech_
                # tokenizer is a plain Python wrapper, not an nn.Module, so
                # the .to(self.device) above never reaches its inner .model
                # (confirmed live: left on cpu, silently). Its own .device
                # attribute is also a stale plain attribute from
                # construction, not something that tracks the real device.
                self.model.model.speech_tokenizer.model = self.model.model.speech_tokenizer.model.to(self.device)
                self.model.model.speech_tokenizer.device = self.device

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen CustomVoice %s loaded successfully", model_size)

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Qwen CustomVoice unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> tuple[dict, bool]:
        """
        Create voice prompt for CustomVoice.

        CustomVoice doesn't use reference audio — it uses preset speakers.
        When called for a cloned profile (fallback), uses the default speaker.
        For preset profiles, the voice_prompt dict is built by the profile
        service and bypasses this method entirely.
        """
        return {
            "voice_type": "preset",
            "preset_engine": "qwen_custom_voice",
            "preset_voice_id": QWEN_CV_DEFAULT_SPEAKER,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio using Qwen CustomVoice.

        Args:
            text: Text to synthesize
            voice_prompt: Dict with preset_voice_id (speaker name)
            language: Language code (zh, en, ja, ko, etc.)
            seed: Random seed for reproducibility
            instruct: Natural language instruction for style control
                      (e.g. "Speak in an angry tone", "Very happy")

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model_async(None)

        speaker = voice_prompt.get("preset_voice_id") or QWEN_CV_DEFAULT_SPEAKER

        def _generate_sync():
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

            lang_name = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            kwargs = {
                "text": text,
                "language": lang_name.capitalize() if lang_name != "auto" else "Auto",
                "speaker": speaker,
            }

            # Only pass instruct if non-empty
            if instruct:
                kwargs["instruct"] = instruct

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state. Forcing offline here (issue #462) regressed online
            # users whose libraries issue legitimate metadata lookups
            # during generation.
            #
            # See pytorch_backend.py's generate() for the full rationale —
            # same single-parameter repetition_penalty nudge (1.05 -> 1.15)
            # applied here for consistency, since this backend shares the
            # same qwen_tts library and underlying repetition-loop failure
            # mode. Everything else stays at proven-stable defaults after
            # the earlier broader attempt broke generation in production.
            kwargs.setdefault("repetition_penalty", 1.15)
            wavs, sample_rate = self.model.generate_custom_voice(**kwargs)
            return wavs[0], sample_rate

        audio, sample_rate = await asyncio.to_thread(_generate_sync)
        return audio, sample_rate

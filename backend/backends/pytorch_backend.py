"""
PyTorch backend implementation for TTS and STT.
"""

from typing import Optional, List, Tuple
import asyncio
import logging
import torch
import numpy as np
from accelerate.hooks import attach_align_device_hook

logger = logging.getLogger(__name__)

from . import TTSBackend, STTBackend, LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import (
    is_model_cached,
    get_torch_device,
    empty_device_cache,
    manual_seed,
    combine_voice_prompts as _combine_voice_prompts,
    model_load_progress,
)
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.audio import load_audio


class PyTorchTTSBackend:
    """PyTorch-based TTS backend using Qwen3-TTS."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self.device = self._get_device()
        self._current_model_size = None

    def _get_device(self) -> str:
        """Get the best available device."""
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the HuggingFace Hub model ID.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID
        """
        hf_model_map = {
            "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        }

        if model_size not in hf_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        return hf_model_map[model_size]

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(self._get_model_path(model_size))

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the TTS model with automatic downloading from HuggingFace Hub.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size

        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return

        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from qwen_tts import Qwen3TTSModel

            model_path = self._get_model_path(model_size)
            logger.info("Loading TTS model %s on %s...", model_size, self.device)

            # Route both HF Hub and Transformers through a single cache root.
            # On Windows local setups, model assets can otherwise split between
            # .hf-cache/hub and .hf-cache/transformers, causing speech_tokenizer
            # and preprocessor_config.json to fail to resolve during load.
            from huggingface_hub import constants as hf_constants
            tts_cache_dir = hf_constants.HF_HUB_CACHE

            if self.device == "cpu":
                self.model = Qwen3TTSModel.from_pretrained(
                    model_path,
                    cache_dir=tts_cache_dir,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=False,
                )
            else:
                # device_map=self.device routes loading through accelerate's
                # meta-device dispatch, but Qwen3TTSModel constructs its
                # talker/code_predictor/speaker_encoder submodules directly
                # ("Initializing talker model with default values" in logs)
                # rather than through accelerate's checkpoint-dispatch path,
                # so those submodules are left on the meta device with no
                # data — accelerate never materializes them, and any later
                # .to()/inference call fails with "Cannot copy out of meta
                # tensor; no data!". Loading with real tensors up front
                # (mirroring the CPU branch) sidesteps accelerate's dispatch
                # path entirely, so nothing ends up on meta.
                # Confirmed live: installing flash-attn alone doesn't make
                # anything use it — HF's from_pretrained defaults to eager/
                # sdpa attention unless explicitly told otherwise, and this
                # custom model class doesn't auto-detect it. In x-vector
                # mode (no ref_text), a plain string attn_implementation=
                # "flash_attention_2" cut generate_voice_clone() from ~30min
                # to ~4.4min live. But ICL mode (real profiles - the ref_text/
                # ref_code path this backend actually uses in production)
                # additionally runs reference audio through the Mimi codec
                # (transformers.models.mimi), which is NOT one of this
                # model's registered sub_configs (only talker_config and
                # speaker_encoder_config are, confirmed live via
                # config.sub_configs) but still inherits the plain string via
                # HF's generic recursive config propagation. Mimi's own RoPE
                # forward does `torch.autocast(device_type=x.device.type,
                # enabled=False)` to force float32 - something about flash
                # attention's own dtype/autocast state corrupts that call,
                # confirmed live: "RuntimeError: unsupported scalarType"
                # inside modeling_mimi.py, immediately crashing every ICL
                # generation on this same image that worked fine in x-vector
                # mode. Scoping attn_implementation as a dict to only the
                # two real sub_configs avoids ever touching Mimi's own
                # attention implementation - confirmed live this fixes the
                # crash and still gets a real (smaller, ~30min->~20min)
                # speedup from flash attention on the talker's autoregressive
                # loop, which dominates generation cost either way.
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
                    cache_dir=tts_cache_dir,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                    **attn_kwargs,
                )
                # Qwen3TTSModel is a plain wrapper (model/processor/
                # generate_defaults), not an nn.Module itself — it has no
                # .to() of its own ("'Qwen3TTSModel' object has no attribute
                # 'to'", confirmed live). The actual PyTorch model requiring
                # device placement is the inner .model
                # (Qwen3TTSForConditionalGeneration, a real nn.Module).
                self.model.model = self.model.model.to(self.device)
                # Loading without device_map means we also lose the
                # accelerate pre-forward hook that device_map installs,
                # which is what silently moved caller-supplied CPU tensors
                # onto the model's device on every forward call. Confirmed
                # live: upstream's generate_voice_clone() tokenizes text via
                # self._tokenize_texts(...) (a plain CPU tensor, HF
                # tokenizers always return CPU by default) and passes it
                # straight into self.model.generate(input_ids=...) with no
                # device handling of its own anywhere in that function - it
                # was relying entirely on device_map's hook to paper over
                # that gap. Without device_map, that call hits "Expected
                # all tensors to be on the same device, but got index is on
                # cpu, different from other tensors on cuda:0". Explicitly
                # attaching the same hook accelerate would have installed
                # restores automatic CPU->device tensor alignment on every
                # forward call, without going through device_map's
                # meta-device loading path (which is what caused the
                # earlier "Cannot copy out of meta tensor" failure this
                # whole branch exists to avoid). Verified live: a CPU index
                # tensor fed into a CUDA nn.Embedding after this call
                # resolves correctly instead of raising.
                attach_align_device_hook(self.model.model, execution_device=self.device)
                # THE actual root cause of ~30-50min generations (not the
                # attention implementation, not max_new_tokens — those were
                # real but secondary): speech_tokenizer (the Mimi-style
                # codec used for both reference-audio encoding and final
                # audio decoding) is never moved to GPU at all by
                # self.model.model.to(self.device) above. Confirmed live:
                # `speech_tokenizer.model.decoder` params were still on cpu
                # after that .to() call — speech_tokenizer is a plain Python
                # wrapper class (Qwen3TTSTokenizer), not an nn.Module
                # itself, so PyTorch's device-recursion (which only follows
                # registered nn.Module submodules) never reaches its inner
                # .model at all, unlike talker/speaker_encoder which ARE
                # real nn.Module children and moved correctly. A 114M-param
                # decoder running on CPU for even a few hundred codec frames
                # is entirely sufficient to explain 3000+ second decode
                # calls. Separately, the wrapper's own .device attribute
                # (used internally by its decode()/encode() methods to move
                # *input* tensors before calling into the model) is a plain
                # attribute set once at construction, not something that
                # tracks the inner model's real device — it stays "cpu"
                # even after the line below moves the actual weights,
                # so it needs setting explicitly too, or those methods keep
                # moving inputs to the wrong place and hit "Expected all
                # tensors to be on the same device" against the now-GPU
                # weights. Verified live: fixing both took total generation
                # time from ~3100s to ~5.5s for a comparable ICL-mode call -
                # this single fix accounts for the overwhelming majority of
                # the speedup this whole investigation was chasing, far more
                # than flash-attention or the max_new_tokens cap combined.
                self.model.model.speech_tokenizer.model = self.model.model.speech_tokenizer.model.to(self.device)
                self.model.model.speech_tokenizer.device = self.device

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("TTS model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None

            empty_device_cache(self.device)

            logger.info("TTS model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)

        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key, map_location=self.device)
            if cached_prompt is not None:
                # Cache stores as torch.Tensor but actual prompt is dict
                # Convert if needed
                if isinstance(cached_prompt, dict):
                    # For PyTorch backend, the dict should contain tensors, not file paths
                    # So we can safely return it
                    return cached_prompt, True
                elif isinstance(cached_prompt, torch.Tensor):
                    # Legacy cache format - convert to dict
                    # This shouldn't happen in practice, but handle it
                    return {"prompt": cached_prompt}, True

        def _create_prompt_sync():
            """Run synchronous voice prompt creation in thread pool."""
            # Inference runs with the process's default HF_HUB_OFFLINE
            # state. Forcing offline here (issue #462) regressed online
            # users whose libraries issue legitimate metadata lookups
            # during voice-prompt creation.
            return self.model.create_voice_clone_prompt(
                ref_audio=str(audio_path),
                ref_text=reference_text,
                x_vector_only_mode=False,
            )

        # Run blocking operation in thread pool
        voice_prompt_items = await asyncio.to_thread(_create_prompt_sync)

        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)

        return voice_prompt_items, False

    async def combine_voice_prompts(
        self,
        audio_paths: List[str],
        reference_texts: List[str],
    ) -> Tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary from create_voice_prompt
            language: Language code (en or zh)
            seed: Random seed for reproducibility
            instruct: Natural language instruction for speech delivery control

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Load model
        await self.load_model_async(None)

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            # Set seed if provided
            if seed is not None:
                manual_seed(seed, self.device)

            # Confirmed live: generate_voice_clone()'s default max_new_tokens
            # (8192, from this model's own generate_config.json) is treated
            # as a target, not a safety ceiling - a real ICL-mode call for
            # an 11-word, 12-second-output test generated 8191/8192 frames
            # (torch.Size([8191, 16])), taking 797s for what should need
            # roughly a couple hundred frames. This model's EOS prediction
            # is unreliable (the same underlying issue documented for
            # runaway generation elsewhere in this fork - see
            # utils/chunked_tts.py's has_tts_runaway) - it likely finishes
            # the real content quickly, then keeps emitting near-silent
            # filler until hitting the cap, and something downstream trims
            # that trailing silence, so the wasted generation never showed
            # up as a correctness bug, only a massive time cost. Capping
            # max_new_tokens to the text's own actual likely duration can't
            # lose real content - those extra frames were being generated
            # and then discarded either way. This model generates ~12 codec
            # frames/sec; ~2 words/sec is a conservative (slow) spoken pace,
            # and ICL mode also replays the reference audio before the new
            # content, so a flat buffer covers that. 2x multiplier on top
            # for safety margin (pacing variance, instruct-driven slower
            # delivery) - still enormously tighter than the unconditional
            # 8192 default.
            words = len(text.split())
            estimated_content_seconds = words / 2.0
            reference_replay_buffer_seconds = 20
            max_new_tokens = min(8192, max(200, int((estimated_content_seconds + reference_replay_buffer_seconds) * 12 * 2)))

            # See _create_prompt_sync comment — inference runs with the
            # process's default HF_HUB_OFFLINE state (issue #462).
            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                voice_clone_prompt=voice_prompt,
                language=LANGUAGE_CODE_TO_NAME.get(language, "auto"),
                instruct=instruct,
                max_new_tokens=max_new_tokens,
            )
            return wavs[0], sample_rate

        # Run blocking inference in thread pool to avoid blocking event loop
        audio, sample_rate = await asyncio.to_thread(_generate_sync)

        return audio, sample_rate


class PyTorchSTTBackend:
    """PyTorch-based STT backend using Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.processor = None
        self.model_size = model_size
        self.device = self._get_device()

    def _get_device(self) -> str:
        """Get the best available device."""
        return get_torch_device(allow_xpu=True, allow_directml=True)

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo)

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self.model_size == model_size:
            return

        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading Whisper model %s on %s...", model_size, self.device)

            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        self.model.to(self.device)
        self.model_size = model_size
        logger.info("Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            empty_device_cache(self.device)

            logger.info("Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        await self.load_model_async(model_size)

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # Load audio
            audio, _sr = load_audio(audio_path, sample_rate=16000)

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state — forcing offline here (issue #462) broke online users
            # whose `get_decoder_prompt_ids` / tokenizer calls issue
            # legitimate metadata lookups.
            # Process audio
            inputs = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate transcription
            # If language is provided, force it; otherwise let Whisper auto-detect
            generate_kwargs = {}
            if language:
                forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                    language=language,
                    task="transcribe",
                )
                generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs["input_features"],
                    **generate_kwargs,
                )

            # Decode
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
            )[0]

            return transcription.strip()

        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)

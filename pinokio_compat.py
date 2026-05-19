import os
from typing import Any


def _hf_hub_cache() -> str:
    """Resolve the canonical HF hub cache root ($HF_HUB_CACHE → $HF_HOME/hub)."""
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        return os.path.abspath(explicit)
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.abspath(os.path.join(hf_home, "hub"))


def patch_hf_downloads_for_pinokio(model_downloader: Any) -> None:
    """Route Hugging Face downloads through the standard HF hub cache layout.

    Upstream's model_downloader passes only ``cache_dir=$HF_HOME/.cache/dramabox``
    which produces the right blob/snapshot tree but at a non-standard root.
    Earlier versions of this shim additionally passed ``local_dir=...`` which
    forced huggingface_hub to keep both the cache tree AND a flat duplicate
    under ``local/...`` (a full second copy on Windows since symlinks are off
    by default), doubling disk usage.

    We point ``cache_dir`` at the canonical ``$HF_HUB_CACHE`` (i.e.
    ``$HF_HOME/hub``) and drop ``local_dir`` entirely. Result: one copy per
    weight, at the expected location:
        cache/HF_HOME/hub/models--<org>--<repo>/snapshots/<sha>/...
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    hub_cache = _hf_hub_cache()

    def get_model_path(name: str, cache_dir: str = None) -> str:
        if name not in model_downloader.MODEL_FILES:
            raise ValueError(f"Unknown model: {name}. Choose from: {list(model_downloader.MODEL_FILES.keys())}")
        repo_path = model_downloader.MODEL_FILES[name]
        model_downloader.logger.info(f"Fetching {name} from {model_downloader.DRAMABOX_REPO}/{repo_path}...")
        local_path = hf_hub_download(
            repo_id=model_downloader.DRAMABOX_REPO,
            filename=repo_path,
            cache_dir=cache_dir or hub_cache,
            token=os.environ.get("HF_TOKEN"),
        )
        model_downloader.logger.info(f"  -> {local_path}")
        return local_path

    def get_gemma_path(cache_dir: str = None) -> str:
        model_downloader.logger.info(f"Fetching Gemma from {model_downloader.GEMMA_REPO}...")
        local_dir = snapshot_download(
            repo_id=model_downloader.GEMMA_REPO,
            cache_dir=cache_dir or hub_cache,
            token=os.environ.get("HF_TOKEN"),
            max_workers=1,
        )
        model_downloader.logger.info(f"  -> {local_dir}")
        return local_dir

    def get_all_paths(cache_dir: str = None) -> dict:
        paths = {}
        for name in model_downloader.MODEL_FILES:
            paths[name] = get_model_path(name)
        paths["gemma_root"] = get_gemma_path()
        return paths

    model_downloader.get_model_path = get_model_path
    model_downloader.get_gemma_path = get_gemma_path
    model_downloader.get_all_paths = get_all_paths


def _mps_is_available(torch_module: Any) -> bool:
    return bool(
        hasattr(torch_module.backends, "mps")
        and torch_module.backends.mps.is_available()
    )


def select_torch_device(torch_module: Any) -> str:
    """Pick the best available accelerator for local Pinokio launches."""
    if _mps_is_available(torch_module):
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda"
    return "cpu"


def _device_type(torch_module: Any, device: Any) -> str:
    try:
        return torch_module.device(device).type
    except Exception:
        return str(device).split(":", 1)[0]


def _choose_runtime_device(torch_module: Any, requested_device: Any) -> str:
    override = os.environ.get("DRAMABOX_DEVICE") or os.environ.get("LTX_DEVICE")
    if override:
        return override

    requested_type = _device_type(torch_module, requested_device)
    if requested_type == "cuda" and not torch_module.cuda.is_available():
        return select_torch_device(torch_module)
    return str(requested_device)


def patch_tts_server_device(inference_server: Any, torch_module: Any = None) -> None:
    """Replace upstream's CUDA-only default with local CUDA/MPS/CPU selection."""
    torch_module = torch_module or inference_server.torch
    original_cls = inference_server.TTSServer
    if getattr(original_cls, "_pinokio_device_patched", False):
        return

    class PinokioTTSServer(original_cls):
        _pinokio_device_patched = True
        _pinokio_original_cls = original_cls

        def __init__(self, *args, **kwargs):
            args = list(args)
            if "device" in kwargs:
                requested_device = kwargs["device"]
                selected_device = _choose_runtime_device(torch_module, requested_device)
                kwargs["device"] = selected_device
            elif len(args) >= 4:
                requested_device = args[3]
                selected_device = _choose_runtime_device(torch_module, requested_device)
                args[3] = selected_device
            else:
                requested_device = "cuda"
                selected_device = _choose_runtime_device(torch_module, requested_device)
                kwargs["device"] = selected_device

            if str(selected_device) != str(requested_device):
                inference_server.logging.info(
                    "Pinokio selected %s instead of requested %s",
                    selected_device,
                    requested_device,
                )

            on_mps = _device_type(torch_module, selected_device) == "mps"
            if on_mps:
                kwargs.setdefault("compile_model", False)

            super().__init__(*args, **kwargs)

            if on_mps:
                _cast_audio_stack_to_fp32(self, torch_module, inference_server.logging)

    inference_server.TTSServer = PinokioTTSServer


def patch_mps_vocoder_dtype(torch_module: Any) -> None:
    """Force the audio decoder + vocoder to run in fp32 on MPS.

    Why this exists:
      - VocoderWithBWE.forward() wraps itself in
        ``torch.autocast(device_type=mel_spec.device.type, dtype=torch.float32)``
        to escape bf16/fp16 underflow inside the STFT/BWE path.
      - On MPS, ``torch.autocast`` only accepts bf16/fp16 — passing float32
        emits ``"In MPS autocast, but the target dtype is not supported.
        Disabling autocast"`` and the entire vocoder ends up running in bf16.
      - bf16 ``sqrt(real**2 + imag**2)`` inside ``_STFTFn`` underflows, NaNs
        propagate, and torchaudio saves a non-finite waveform as a silent
        (or empty) WAV. Symptom: correct-length file, total silence,
        "Audio buffer is not finite everywhere" warning from the Perth
        watermark stage.

    Fix: at TTSServer construction time on MPS, cast the warm AudioDecoder
    (nn.Module) and every VocoderWithBWE sub-module to float32, and replace
    ``VocoderWithBWE.forward`` with a version that does the dtype handling
    explicitly instead of relying on autocast.
    """
    from ltx_core.model.audio_vae import vocoder as vocoder_module
    import torch.nn.functional as F

    if getattr(vocoder_module, "_pinokio_mps_fp32_patched", False):
        return

    def _vocoder_with_bwe_forward_fp32(self, mel_spec):
        """Drop-in replacement for VocoderWithBWE.forward that runs in fp32.

        Mirrors upstream logic but skips the autocast context (no-op on MPS)
        and casts inputs/outputs explicitly.
        """
        input_dtype = mel_spec.dtype
        mel_spec = mel_spec.float()

        x = self.vocoder(mel_spec)
        _, _, length_low_rate = x.shape
        output_length = (
            length_low_rate * self.output_sampling_rate // self.input_sampling_rate
        )

        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            x = F.pad(x, (0, self.hop_length - remainder))

        mel = self._compute_mel(x)
        # _compute_mel applies log to the mel power; log(0) = -inf for silent
        # frames.  bwe_generator's convolutions turn -inf into NaN, and NaN
        # propagates into the final waveform where torch.clamp cannot remove it
        # (NaN comparisons are always False, so clamp is a no-op on NaN).
        # Floor to -80 dB (a standard mel silence level) before bwe_generator.
        mel = torch_module.nan_to_num(mel, nan=-80.0, posinf=0.0, neginf=-80.0)
        mel_for_bwe = mel.transpose(2, 3)
        residual = self.bwe_generator(mel_for_bwe)
        skip = self.resampler(x)

        out = torch_module.clamp(residual + skip, -1, 1)[..., :output_length]
        # Second safety net: any remaining non-finite samples (NaN/±Inf from
        # unusual inputs) become silence so Perth watermarking never skips.
        out = torch_module.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        return out.to(input_dtype)

    vocoder_module._pinokio_mps_fp32_patched = True
    vocoder_module._pinokio_vocoder_with_bwe_forward_fp32 = _vocoder_with_bwe_forward_fp32

    # Patch the class directly so every instantiation (warm, cold, lazy) gets
    # the fp32 path — instance-level MethodType binding in _cast_audio_stack_to_fp32
    # only works if the object already exists at TTSServer.__init__ time.
    if hasattr(vocoder_module, "VocoderWithBWE"):
        vocoder_module.VocoderWithBWE.forward = _vocoder_with_bwe_forward_fp32


def _cast_audio_stack_to_fp32(server: Any, torch_module: Any, logger: Any) -> None:
    """Cast the warm AudioDecoder + VocoderWithBWE to fp32 (MPS only)."""
    import types
    from ltx_core.model.audio_vae import vocoder as vocoder_module

    audio_decoder = getattr(server, "_audio_decoder", None)
    if audio_decoder is None:
        return

    warm_decoder = getattr(audio_decoder, "_warm_decoder", None)
    warm_vocoder = getattr(audio_decoder, "_warm_vocoder", None)
    if warm_decoder is None and warm_vocoder is None:
        return  # cold mode — modules are built per call; not patched here

    if warm_decoder is not None:
        warm_decoder.float()
        # DiT outputs bf16; cast input latent to fp32 before the decoder's conv_in.
        warm_decoder.register_forward_pre_hook(
            lambda mod, inp: tuple(
                x.float() if isinstance(x, torch_module.Tensor) else x for x in inp
            )
        )

    if warm_vocoder is not None:
        # Cast ALL parameters and buffers to fp32 — this covers .vocoder,
        # .bwe_generator, .mel_stft, .resampler, and any other sub-modules,
        # without relying on an explicit attribute name list.
        warm_vocoder.float()

        # Replace VocoderWithBWE.forward with an explicit fp32 implementation.
        # Use duck-typing rather than isinstance() to handle the common case where
        # warm_vocoder's class was imported via a different sys.path entry than
        # vocoder_module (e.g. "ltx2.ltx_core..." vs "ltx_core..."), which makes
        # isinstance() return False even though the classes are structurally identical.
        has_bwe = hasattr(warm_vocoder, "bwe_generator") and hasattr(warm_vocoder, "resampler")
        replacement = getattr(vocoder_module, "_pinokio_vocoder_with_bwe_forward_fp32", None)
        if has_bwe and replacement is not None:
            # Patch the actual runtime class so any future instances (cold-mode or
            # lazy) also get the fp32 forward path without needing this function.
            actual_cls = type(warm_vocoder)
            if not getattr(actual_cls, "_pinokio_mps_fp32_patched", False):
                actual_cls.forward = replacement
                actual_cls._pinokio_mps_fp32_patched = True
            warm_vocoder.forward = types.MethodType(replacement, warm_vocoder)

    logger.info("MPS: cast AudioDecoder + Vocoder to float32 (autocast(fp32) is a no-op on MPS)")


def apply_runtime_patches(model_downloader: Any, inference_server: Any = None, torch_module: Any = None) -> None:
    patch_hf_downloads_for_pinokio(model_downloader)

    if inference_server is None:
        return

    torch_module = torch_module or inference_server.torch
    patch_mps_vocoder_dtype(torch_module)
    patch_tts_server_device(inference_server, torch_module)

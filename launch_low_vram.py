import os
import sys
import time

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(ROOT_DIR, "app")
SRC_DIR = os.path.join(APP_DIR, "src")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, APP_DIR)

import torch  # noqa: E402

if not torch.cuda.is_available():
    sys.stderr.write(
        "\n"
        "================================================================\n"
        " Low VRAM mode requires an NVIDIA GPU with CUDA.\n"
        " MMGP (Memory Management for the GPU Poor) can only offload to a\n"
        " CUDA device, and this machine does not have CUDA-enabled PyTorch.\n"
        "\n"
        " Use the regular 'Start' option instead.\n"
        "================================================================\n"
        "\n"
    )
    sys.exit(1)

from mmgp import offload, profile_type  # noqa: E402
import model_downloader  # noqa: E402
import inference_server  # noqa: E402
from pinokio_compat import apply_runtime_patches  # noqa: E402


apply_runtime_patches(model_downloader)


def _profile_from_env():
    profiles = {
        "1": profile_type.HighRAM_HighVRAM,
        "2": profile_type.HighRAM_LowVRAM,
        "3": profile_type.LowRAM_HighVRAM,
        "4": profile_type.LowRAM_LowVRAM,
        "5": profile_type.VerylowRAM_LowVRAM,
        "highram_highvram": profile_type.HighRAM_HighVRAM,
        "highram_lowvram": profile_type.HighRAM_LowVRAM,
        "lowram_highvram": profile_type.LowRAM_HighVRAM,
        "lowram_lowvram": profile_type.LowRAM_LowVRAM,
        "verylowram_lowvram": profile_type.VerylowRAM_LowVRAM,
    }
    key = os.environ.get("MMGP_PROFILE", "5").strip().lower()
    if key and key != "auto":
        return profiles.get(key, profile_type.VerylowRAM_LowVRAM)
    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return profile_type.VerylowRAM_LowVRAM
    if total_gb >= 24:
        return profile_type.HighRAM_HighVRAM
    if total_gb >= 17:
        return profile_type.HighRAM_LowVRAM
    if total_gb >= 13:
        return profile_type.LowRAM_LowVRAM
    return profile_type.VerylowRAM_LowVRAM


def _load_all_with_mmgp(self):
    """Launcher-side low-VRAM loader.

    The upstream warm loader builds the DiT directly on CUDA, then MMGP offloads
    it after the peak has already happened. Keep the app checkout untouched, but
    load the DiT on CPU here so MMGP owns the first GPU transfer.
    """
    t0_total = time.time()

    t0 = time.time()
    self._prompt_encoder = inference_server.PromptEncoder(
        checkpoint_path=self.full_checkpoint,
        gemma_root=self.gemma_root,
        dtype=self.dtype,
        device=self.device,
        warm=True,
        use_bnb_4bit=self.bnb_4bit,
        audio_only=True,
    )
    inference_server.logging.info(f"  PromptEncoder (warm): {time.time()-t0:.1f}s")

    t0 = time.time()
    self._audio_conditioner = inference_server.AudioConditioner(
        checkpoint_path=self.full_checkpoint,
        dtype=self.dtype,
        device=self.device,
        warm=True,
    )
    inference_server.logging.info(f"  AudioConditioner (warm): {time.time()-t0:.1f}s")

    t0 = time.time()
    class AudioOnlyConfigurator(inference_server.ModelConfigurator[inference_server.LTXModel]):
        @classmethod
        def from_config(cls, cfg):
            t = cfg.get("transformer", {})
            cp = None
            if not t.get("caption_proj_before_connector", False):
                with torch.device("meta"):
                    cp = inference_server.create_caption_projection(t, audio=True)
            return inference_server.LTXModel(
                model_type=inference_server.LTXModelType.AudioOnly,
                audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                audio_in_channels=t.get("audio_in_channels", 128),
                audio_out_channels=t.get("audio_out_channels", 128),
                num_layers=t.get("num_layers", 48),
                audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                norm_eps=t.get("norm_eps", 1e-6),
                attention_type=inference_server.AttentionFunction(t.get("attention_type", "default")),
                positional_embedding_theta=10000.0,
                audio_positional_embedding_max_pos=[20.0],
                timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                rope_type=inference_server.LTXRopeType(t.get("rope_type", "interleaved")),
                double_precision_rope=t.get("frequencies_precision", False) == "float64",
                apply_gated_attention=t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=t.get("cross_attention_adaln", False),
            )

    audio_sd_ops = inference_server.SDOps("AO").with_matching(
        prefix="model.diffusion_model."
    ).with_replacement("model.diffusion_model.", "")
    builder = inference_server.Builder(
        model_path=self.checkpoint,
        model_class_configurator=AudioOnlyConfigurator,
        model_sd_ops=audio_sd_ops,
        registry=inference_server.DummyRegistry(),
    )
    self._velocity_model = builder.build(device=torch.device("cpu"), dtype=self.dtype).eval()
    n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
    model_gb = sum(p.numel() * p.element_size() for p in self._velocity_model.parameters()) / 1e9
    inference_server.logging.info(
        f"  Transformer: {time.time()-t0:.1f}s "
        f"({n_params:.1f}B params, {model_gb:.1f}GB CPU, {self.dtype})"
    )

    t0 = time.time()
    self._audio_decoder = inference_server.AudioDecoder(
        checkpoint_path=self.full_checkpoint,
        dtype=self.dtype,
        device=self.device,
        warm=True,
    )
    inference_server.logging.info(f"  AudioDecoder (warm): {time.time()-t0:.1f}s")

    # Hand every offload-safe warm component to MMGP, not just the DiT.
    #
    # Upstream keeps the Gemma text encoder, the audio VAE encoder, and the
    # audio VAE decoder + vocoder all resident on CUDA (warm=True). On a 12GB
    # card they coexist with the transformer's working set, and the OOM in
    # issue #3 lands *during prompt encoding* — when the audio VAE pieces are
    # idle but still pinned to the GPU. Registering them with MMGP lets it park
    # them on CPU and page each back only for its own forward pass, freeing
    # VRAM exactly at the peak that currently overflows.
    #
    # The Gemma text encoder is deliberately left out: with bnb_4bit=True (the
    # upstream default, inference_server TTSServer.__init__) its weights are
    # bitsandbytes 4-bit tensors that do not survive a round-trip to CPU, so
    # MMGP must not manage it. We only add plain bf16 nn.Modules that are
    # invoked through their forward(). getattr guards keep this a no-op if a
    # future upstream sync renames the warm attributes.
    pipe = {"transformer": self._velocity_model}
    audio_encoder = getattr(self._audio_conditioner, "_warm_encoder", None)
    if audio_encoder is not None:
        pipe["audio_encoder"] = audio_encoder
    audio_decoder = getattr(self._audio_decoder, "_warm_decoder", None)
    if audio_decoder is not None:
        pipe["audio_decoder"] = audio_decoder
    vocoder = getattr(self._audio_decoder, "_warm_vocoder", None)
    if vocoder is not None:
        pipe["vocoder"] = vocoder
    inference_server.logging.info(f"  MMGP offload pipe: {', '.join(pipe.keys())}")
    offload.profile(pipe, _profile_from_env())

    inference_server.logging.info(f"All models loaded in {time.time()-t0_total:.1f}s - ready for requests")


inference_server.TTSServer._load_all = _load_all_with_mmgp

import app as dramabox_app  # noqa: E402


if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    # MMGP mutates/offloads modules at runtime. Keep low-VRAM generations
    # serialized so two requests cannot race the same CPU/GPU module state.
    dramabox_app.app.queue(max_size=2, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
        ssr_mode=False,
        show_api=False,
    )

import os
from typing import Any


def patch_hf_downloads_for_pinokio(model_downloader: Any) -> None:
    """Keep Hugging Face downloads inside the Pinokio app cache."""
    from huggingface_hub import hf_hub_download, snapshot_download

    cache_root = os.path.abspath(model_downloader.DEFAULT_CACHE)
    local_root = os.path.join(cache_root, "local")
    dramabox_dir = os.path.join(local_root, "ResembleAI--Dramabox")
    gemma_dir = os.path.join(local_root, "unsloth--gemma-3-12b-it-bnb-4bit")

    def get_model_path(name: str, cache_dir: str = None) -> str:
        if name not in model_downloader.MODEL_FILES:
            raise ValueError(f"Unknown model: {name}. Choose from: {list(model_downloader.MODEL_FILES.keys())}")
        repo_path = model_downloader.MODEL_FILES[name]
        model_downloader.logger.info(f"Fetching {name} from {model_downloader.DRAMABOX_REPO}/{repo_path}...")
        # Use only local_dir so files land directly in one predictable folder.
        # Passing both cache_dir and local_dir causes huggingface_hub to keep
        # a copy in the HF cache tree AND materialize a second copy in
        # local_dir (full file on Windows, since symlinks are off by default),
        # doubling disk usage.
        local_path = hf_hub_download(
            repo_id=model_downloader.DRAMABOX_REPO,
            filename=repo_path,
            local_dir=dramabox_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        model_downloader.logger.info(f"  -> {local_path}")
        return local_path

    def get_gemma_path(cache_dir: str = None) -> str:
        model_downloader.logger.info(f"Fetching Gemma from {model_downloader.GEMMA_REPO}...")
        # See note in get_model_path: avoid the cache_dir + local_dir double-copy.
        local_dir = snapshot_download(
            repo_id=model_downloader.GEMMA_REPO,
            local_dir=gemma_dir,
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

            if _device_type(torch_module, selected_device) == "mps":
                kwargs.setdefault("compile_model", False)

            super().__init__(*args, **kwargs)

    inference_server.TTSServer = PinokioTTSServer


def patch_mps_vocoder_dtype(torch_module: Any) -> None:
    """Make the vocoder input match MPS weights instead of forcing float32."""
    from ltx_core.model.audio_vae import vocoder as vocoder_module

    original_forward = vocoder_module.Vocoder.forward
    if getattr(original_forward, "_pinokio_mps_dtype_patched", False):
        return

    def forward_with_mps_dtype(self, x):
        try:
            ref = next(self.parameters())
        except StopIteration:
            ref = None

        if ref is not None and ref.device.type == "mps" and (x.device != ref.device or x.dtype != ref.dtype):
            x = x.to(device=ref.device, dtype=ref.dtype)

        return original_forward(self, x)

    forward_with_mps_dtype._pinokio_mps_dtype_patched = True
    forward_with_mps_dtype._pinokio_original_forward = original_forward
    vocoder_module.Vocoder.forward = forward_with_mps_dtype


def apply_runtime_patches(model_downloader: Any, inference_server: Any = None, torch_module: Any = None) -> None:
    patch_hf_downloads_for_pinokio(model_downloader)

    if inference_server is None:
        return

    torch_module = torch_module or inference_server.torch
    patch_tts_server_device(inference_server, torch_module)
    patch_mps_vocoder_dtype(torch_module)

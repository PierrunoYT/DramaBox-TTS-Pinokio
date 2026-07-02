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
            paths[name] = get_model_path(name, cache_dir=cache_dir)
        paths["gemma_root"] = get_gemma_path(cache_dir=cache_dir)
        return paths

    model_downloader.get_model_path = get_model_path
    model_downloader.get_gemma_path = get_gemma_path
    model_downloader.get_all_paths = get_all_paths


def apply_runtime_patches(model_downloader: Any) -> None:
    patch_hf_downloads_for_pinokio(model_downloader)

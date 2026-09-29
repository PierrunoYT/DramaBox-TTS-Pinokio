import concurrent.futures
import logging
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

from pinokio_compat import patch_hf_downloads_for_pinokio, require_cuda, serialize_generation


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_cuda_rejects_cpu_and_rocm(self):
        for available, hip in [(False, None), (True, "6.3")]:
            torch = types.SimpleNamespace(
                cuda=types.SimpleNamespace(is_available=lambda: available),
                version=types.SimpleNamespace(hip=hip),
            )
            with self.assertRaisesRegex(SystemExit, "NVIDIA"):
                require_cuda(torch)
        torch.version.hip = None
        require_cuda(torch)

    def test_cache_routing_and_explicit_override(self):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = Mock(return_value="model")
        hub.snapshot_download = Mock(return_value="gemma")
        constants = types.ModuleType("huggingface_hub.constants")
        constants.HF_HUB_CACHE = "~/test-hub"
        downloader = types.SimpleNamespace(
            MODEL_FILES={"transformer": "dit.safetensors"},
            DRAMABOX_REPO="org/model", GEMMA_REPO="org/gemma",
            logger=logging.getLogger("test"),
        )
        with patch.dict(sys.modules, {"huggingface_hub": hub,
                                      "huggingface_hub.constants": constants}):
            patch_hf_downloads_for_pinokio(downloader)
            expected = os.path.abspath(os.path.expanduser("~/test-hub"))
            self.assertEqual(downloader.DEFAULT_CACHE, expected)
            self.assertEqual(downloader.get_all_paths(),
                             {"transformer": "model", "gemma_root": "gemma"})
            self.assertEqual(hub.hf_hub_download.call_args.kwargs["cache_dir"], expected)
            self.assertNotIn("local_dir", hub.snapshot_download.call_args.kwargs)
            downloader.get_all_paths(cache_dir="custom")
            for download in (hub.hf_hub_download, hub.snapshot_download):
                self.assertEqual(download.call_args.kwargs["cache_dir"], "custom")
            with self.assertRaises(ValueError):
                downloader.get_model_path("missing")

    def test_generation_serialized_and_lock_released_after_failure(self):
        active = 0
        peak = 0
        barrier = threading.Barrier(4)

        def generate(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                time.sleep(0.01)
                if value == "fail":
                    raise ValueError("generation failed")
                return value
            finally:
                active -= 1

        server = types.SimpleNamespace(generate_to_file=generate)
        serialize_generation(server)

        def request(value):
            barrier.wait(timeout=5)
            return server.generate_to_file(value)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(list(pool.map(request, range(4))), list(range(4)))
        self.assertEqual(peak, 1)
        with self.assertRaises(ValueError):
            server.generate_to_file("fail")
        self.assertEqual(server.generate_to_file("ok"), "ok")

    def test_both_launchers_work_from_an_unrelated_directory(self):
        for launcher in ("launch.py", "launch_low_vram.py"):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "app" / "src").mkdir(parents=True)
                shutil.copyfile(ROOT / launcher, root / launcher)
                ui = Mock()
                ui.queue.return_value = ui
                modules = {
                    "torch": types.SimpleNamespace(),
                    "model_downloader": types.SimpleNamespace(),
                    "inference_server": types.SimpleNamespace(TTSServer=type("Server", (), {})),
                    "mmgp": types.SimpleNamespace(offload=Mock(), profile_type=Mock()),
                    "app": types.SimpleNamespace(app=ui, tts=Mock()),
                    "pinokio_compat": types.SimpleNamespace(
                        apply_runtime_patches=Mock(), require_cuda=Mock(), serialize_generation=Mock()),
                }
                old_cwd, old_path = os.getcwd(), sys.path[:]
                try:
                    with patch.dict(sys.modules, modules), patch.dict(
                        os.environ, {"GRADIO_SERVER_PORT": "8123", "GRADIO_SHARE": "0"}
                    ):
                        runpy.run_path(str(root / launcher), run_name="__main__")
                    self.assertEqual(Path.cwd(), root / "app")
                    self.assertTrue((root / "app" / "output").is_dir())
                    self.assertEqual(ui.launch.call_args.kwargs["server_name"], "127.0.0.1")
                    self.assertEqual(ui.launch.call_args.kwargs["server_port"], 8123)
                    self.assertFalse(ui.launch.call_args.kwargs["share"])
                    modules["pinokio_compat"].serialize_generation.assert_called_once_with(
                        modules["app"].tts)
                finally:
                    os.chdir(old_cwd)
                    sys.path[:] = old_path


if __name__ == "__main__":
    unittest.main()

import os
import sys


port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "app")
src_dir = os.path.join(app_dir, "src")

os.chdir(app_dir)
sys.path.insert(0, src_dir)
sys.path.insert(0, app_dir)

import model_downloader  # noqa: E402
import inference_server  # noqa: E402
import torch  # noqa: E402
from pinokio_compat import apply_runtime_patches  # noqa: E402


apply_runtime_patches(model_downloader, inference_server, torch)

import app as dramabox_app  # noqa: E402


dramabox_app.app.queue(max_size=10).launch(
    server_name="127.0.0.1",
    server_port=port,
    share=os.environ.get("GRADIO_SHARE", "0") == "1",
    ssr_mode=False,
    show_api=False,
)

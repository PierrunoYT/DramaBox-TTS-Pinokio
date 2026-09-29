import os
import sys


port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "app")
src_dir = os.path.join(app_dir, "src")

os.chdir(app_dir)
os.makedirs("output", exist_ok=True)
sys.path.insert(0, src_dir)
sys.path.insert(0, app_dir)

import torch  # noqa: E402
from pinokio_compat import apply_runtime_patches, require_cuda, serialize_generation  # noqa: E402

require_cuda(torch)

import model_downloader  # noqa: E402

apply_runtime_patches(model_downloader)

import app as dramabox_app  # noqa: E402

# Generate and example clicks are separate Gradio events with separate
# concurrency limits; share one lock so two generations never overlap on the GPU.
serialize_generation(dramabox_app.tts)

dramabox_app.app.queue(max_size=10).launch(
    server_name="127.0.0.1",
    server_port=port,
    share=os.environ.get("GRADIO_SHARE", "0") == "1",
    ssr_mode=False,
    show_api=False,
)

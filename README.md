<p align="center">
  <img src="icon.jpeg" alt="DramaBox logo" width="160" />
</p>

# DramaBox — Pinokio Launcher

A 1-click Pinokio launcher for [DramaBox](https://github.com/resemble-ai/DramaBox) by Resemble AI — expressive TTS with voice cloning, built on LTX-2.3.

## What It Does

DramaBox is a prompt-driven text-to-speech model with voice cloning. The prompt controls speaker identity, emotion, delivery style, laughs, sighs, pauses and transitions. An optional 10-second voice reference clones the target timbre. It is an IC-LoRA fine-tune of the LTX-2.3 3.3B audio-only model.

**Requirements:** ~24 GB VRAM (NVIDIA GPU), ~17 GB disk space for models.

> ⚠️ **macOS is not supported.** This launcher will not work on Mac (Apple Silicon or Intel).

## How to Use

1. Click **Install** to clone the repo and install all dependencies.
2. Click **Start** to launch the Gradio web UI.
3. Click **Open Web UI** to open DramaBox in your browser.

After upgrading an older launcher, run **Install** once to validate the existing
environment and record a successful install. This reuses the `app` checkout and
cached models. Start appears only after dependency installation and validation
succeed. If installation fails, fix the reported error and run Install again.

Updates use fast-forward-only Git pulls. If an update reports diverging history,
resolve the local changes before retrying.

### Low VRAM Mode

Click **Start Low VRAM** to try the experimental MMGP offload wrapper. The launcher builds the transformer on CPU before MMGP manages GPU transfers and offloads the audio VAE modules. The quantized Gemma encoder remains on CUDA. The upstream checkout stays untouched. The launcher selects an MMGP profile from detected CUDA VRAM; lower VRAM usage comes at the cost of speed and higher system RAM usage. Generate and example requests share a lock to prevent overlapping model transfers.

### Optional: RE-USE Voice Denoising

Click **Install RE-USE** (after a successful install) to add the optional [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE) voice-reference denoising addon. This installs `librosa`, `resampy`, and — on Linux only — the `mamba-ssm` and `causal-conv1d` Bi-Mamba kernels. Windows skips these kernels; denoising availability depends on the upstream fallback implementation.

**Requirements:** Linux kernel builds require `nvcc` and a CUDA toolkit matching the installed PyTorch build (CUDA 12.8). The installer supplies Python build tools and uses the existing environment for compilation. If compilation fails, installation stops; fix the reported toolchain error and retry. macOS is not supported.

### Prompt Writing Guide

**Structure:** `<speaker description>, "<dialogue>" <action direction> "<more dialogue>"`

**Inside quotes** (model produces actual sounds):
- Laughs: `"Hahaha"` `"Hehehe"`
- Sounds: `"Mmmmm"` `"Ugh"` `"Argh"` `"Ahhh"` `"Hmm"`

**Outside quotes** (stage directions):
- `She sighs deeply.` · `He gulps nervously.` · `A long pause.`

**Example prompt:**
```
A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is so good to see you!"
```

## API

### Python

Run from the launcher repository root using the installed `app/env` Python environment.
Resolve the downloaded checkpoints explicitly; the server's defaults refer to a
different set of local model filenames.

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path("app/src").resolve()))
import model_downloader
from pinokio_compat import apply_runtime_patches

apply_runtime_patches(model_downloader)
paths = model_downloader.get_all_paths()
from inference_server import TTSServer

server = TTSServer(
    checkpoint=paths["transformer"],
    full_checkpoint=paths["audio_components"],
    gemma_root=paths["gemma_root"],
    device="cuda",
    compile_model=False,
)
server.generate_to_file(
    prompt='A woman speaks warmly, "Hello, how are you today?"',
    output="output.wav",
    voice_ref="reference.wav",   # optional, 10+ seconds
)
```

### JavaScript

> The port is assigned dynamically. Replace `7860` with the URL shown in **Open Web UI**.

```javascript
import { Client } from "@gradio/client";

const client = await Client.connect("http://127.0.0.1:7860");
const result = await client.predict("/on_generate", [
  'A woman speaks warmly, "Hello, how are you today?"',
  null, 2.5, 1.5, 1.1, 0, 10, 42, false, 45, 37, 50
]);

console.log(result);
```

### Python Gradio Client

Once the server is running, use the URL from **Open Web UI** with the Gradio API:

```python
from gradio_client import Client

client = Client("http://127.0.0.1:7860")  # replace port with the one shown in Open Web UI
result = client.predict(
    prompt='A woman speaks warmly, "Hello, how are you today?"',
    audio_ref=None,
    cfg=2.5, stg=1.5, dur_mult=1.1,
    gen_dur=0, ref_dur=10, seed=42, denoise_ref=False,
    max_chunk_s=45, target_chunk_s=37, crossfade_ms=50,
    api_name="/on_generate",
)
print(result)
```

The inputs follow the upstream `on_generate` handler: prompt, optional reference,
CFG, STG, duration multiplier, target duration, reference duration, seed,
denoising, maximum chunk duration, target chunk duration, and crossfade.
Use `client.view_api()` to inspect the running version after upstream updates.

### Curl

```bash
# Replace 7860 with the port shown in Open Web UI
curl -X POST http://127.0.0.1:7860/gradio_api/call/on_generate \
  -H "Content-Type: application/json" \
  -d '{"data": ["A woman speaks warmly, \"Hello, how are you today?\"", null, 2.5, 1.5, 1.1, 0, 10, 42, false, 45, 37, 50]}'

# Copy event_id from the POST response into EVENT_ID below, then stream the result.
curl -N http://127.0.0.1:7860/gradio_api/call/on_generate/EVENT_ID
```

## Development checks

These checks use Python's standard library and Node.js; no models or GPU are needed:

```bash
python -m unittest discover -s tests -v
node --test tests/launcher.test.js
```

They cover startup configuration, cache routing, generation locking, hardware
guards, URL capture, and launcher state transitions. Full audio generation and
Linux CUDA extension builds require a separate test with the installed app and GPU.

## Models

Auto-downloaded on first run from [ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox):

| File | Size | Description |
|------|------|-------------|
| dramabox-dit-v1.safetensors | 6.6 GB | DiT transformer |
| dramabox-audio-components.safetensors | 1.9 GB | Audio embeddings + VAE + vocoder |
| gemma-3-12b-it-bnb-4bit | ~8 GB | Text encoder |

## License

DramaBox is distributed under the [LTX-2 Community License](https://github.com/resemble-ai/DramaBox/blob/master/LICENSE). Built on [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3).

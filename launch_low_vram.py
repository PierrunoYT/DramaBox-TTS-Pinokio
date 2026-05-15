import os
import subprocess
import sys
import tempfile
import time

import gradio as gr


port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
repo_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(repo_dir, "app")

os.chdir(app_dir)
sys.path.insert(0, app_dir)
sys.path.insert(0, os.path.join(app_dir, "src"))

from model_downloader import get_all_paths  # noqa: E402


PATHS = get_all_paths()
OUTPUT_DIR = os.path.join(app_dir, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return "..." + text[-max_chars:]


def generate_low_vram(
    prompt: str,
    audio_ref,
    cfg: float,
    stg: float,
    dur_mult: float,
    gen_dur: float,
    ref_dur: float,
    steps: int,
    seed: int,
):
    if not prompt or not prompt.strip():
        raise gr.Error("Prompt is empty.")

    output = tempfile.mktemp(suffix=".wav", prefix="dramabox_low_vram_", dir=OUTPUT_DIR)
    command = [
        sys.executable,
        "src/inference.py",
        "--prompt",
        prompt,
        "--output",
        output,
        "--checkpoint",
        PATHS["transformer"],
        "--full-checkpoint",
        PATHS["audio_components"],
        "--gemma-root",
        PATHS["gemma_root"],
        "--cfg-scale",
        str(float(cfg)),
        "--stg-scale",
        str(float(stg)),
        "--duration-multiplier",
        str(float(dur_mult)),
        "--gen-duration",
        str(float(gen_dur)),
        "--ref-duration",
        str(float(ref_dur)),
        "--steps",
        str(int(steps)),
        "--seed",
        str(int(seed)),
    ]

    ref_path = audio_ref if audio_ref and os.path.exists(str(audio_ref)) else None
    if ref_path:
        command.extend(["--voice-sample", str(ref_path)])
    else:
        command.append("--no-ref")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start = time.time()
    result = subprocess.run(
        command,
        cwd=app_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    log = "\n".join(part for part in (result.stdout, result.stderr) if part)

    if result.returncode != 0:
        raise gr.Error(
            "Low VRAM generation failed. Last log lines:\n\n"
            + _tail(log)
        )
    if not os.path.exists(output):
        raise gr.Error(
            "Low VRAM generation finished without an output file. Last log lines:\n\n"
            + _tail(log)
        )

    elapsed = time.time() - start
    print(f"Low VRAM generation completed in {elapsed:.2f}s -> {output}", flush=True)
    if log:
        print(_tail(log), flush=True)
    return output


EXAMPLES = [
    (
        "Warm greeting",
        'A woman speaks warmly, "Hello, how are you today?" She laughs, "Hahaha, it is good to see you!"',
    ),
    (
        "Short dramatic line",
        'A tired detective whispers, "I know who took the missing letter." He pauses, "And I know why."',
    ),
]


with gr.Blocks(
    title="DramaBox - Low VRAM",
    theme=gr.themes.Default(),
    analytics_enabled=False,
) as app:
    gr.Markdown("# DramaBox - Low VRAM Mode")
    gr.Markdown(
        "This mode is intended for GPUs below the normal 24 GB recommendation. "
        "It runs each generation through DramaBox's sequential CLI path so Gemma, "
        "the DiT, and the decoder are not kept warm on the GPU at the same time. "
        "Startup and each request are slower, but peak VRAM pressure is lower."
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt_box = gr.Textbox(
                label="Scene prompt",
                placeholder=EXAMPLES[0][1],
                lines=6,
            )
            audio_ref = gr.Audio(
                label="Voice reference (optional, 10+ seconds)",
                type="filepath",
            )
            gen_btn = gr.Button("Generate (Low VRAM)", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Accordion("Inference settings", open=True):
                cfg_slider = gr.Slider(1.0, 10.0, value=2.5, step=0.5, label="CFG scale")
                stg_slider = gr.Slider(0.0, 5.0, value=1.5, step=0.5, label="STG scale")
                dur_slider = gr.Slider(
                    0.8,
                    2.0,
                    value=1.0,
                    step=0.05,
                    label="Duration x (only used when target duration = 0)",
                )
                gen_dur_slider = gr.Slider(
                    0.0,
                    30.0,
                    value=0.0,
                    step=1.0,
                    label="Target duration (s) - 0 = auto from prompt",
                )
                ref_dur_slider = gr.Slider(
                    3.0,
                    15.0,
                    value=8.0,
                    step=1.0,
                    label="Reference duration (s)",
                )
                steps_slider = gr.Slider(
                    12,
                    30,
                    value=20,
                    step=1,
                    label="Denoising steps (lower = faster, lower quality)",
                )
                seed_input = gr.Number(value=42, label="Seed", precision=0)

            audio_out = gr.Audio(label="Generated audio", type="filepath")

    gen_btn.click(
        generate_low_vram,
        inputs=[
            prompt_box,
            audio_ref,
            cfg_slider,
            stg_slider,
            dur_slider,
            gen_dur_slider,
            ref_dur_slider,
            steps_slider,
            seed_input,
        ],
        outputs=[audio_out],
    )

    gr.Examples(
        label="Examples",
        examples=[[name, prompt] for name, prompt in EXAMPLES],
        inputs=[gr.Textbox(visible=False), prompt_box],
        cache_examples=False,
    )


app.queue(max_size=1).launch(
    server_name="127.0.0.1",
    server_port=port,
    share=os.environ.get("GRADIO_SHARE", "0") == "1",
    ssr_mode=False,
    show_api=False,
)

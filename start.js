module.exports = {
  daemon: true,
  run: [
    {
      when: "{{platform === 'darwin'}}",
      method: "notify",
      params: {
        html: "macOS is not supported. DramaBox requires Windows or Linux with an NVIDIA GPU."
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          GRADIO_SERVER_PORT: "{{port}}",
          // Reduce CUDA allocator fragmentation on tight-VRAM cards. Lets
          // PyTorch grow/shrink segments instead of stranding "reserved but
          // unallocated" memory that triggers spurious OOMs.
          PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"
        },
        path: "app",
        message: ["python ../launch.py"],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}

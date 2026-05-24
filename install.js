module.exports = {
  requires: {
    bundle: "ai"
  },
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
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: "git clone https://github.com/resemble-ai/DramaBox app"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: ["uv pip install -r ../requirements.txt", "uv pip install -r ../requirements_low_vram.txt"]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app"
        }
      }
    },
    {
      method: "notify",
      params: {
        html: "Installation finished! Click the 'Start' tab to launch DramaBox."
      }
    }
  ]
}

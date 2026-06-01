module.exports = {
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
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
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
        html: "Update complete! Click 'Start' to launch DramaBox."
      }
    }
  ]
}

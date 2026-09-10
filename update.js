module.exports = {
  run: [
    {
      when: "{{platform === 'darwin' || gpu !== 'nvidia'}}",
      method: "notify",
      params: {
        html: "DramaBox requires Windows or Linux with an NVIDIA GPU. CPU, AMD, and macOS are not supported."
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        message: "git pull --ff-only"
      }
    },
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull --ff-only"
      }
    },
    {
      when: "{{exists('app/.installed')}}",
      method: "fs.rm",
      params: { path: "app/.installed" }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install -c ../constraints.txt -r ../requirements.txt -r ../requirements_low_vram.txt"
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
      method: "shell.run",
      params: { venv: "env", path: "app", message: "uv pip check" }
    },
    {
      method: "fs.write",
      params: { path: "app/.installed", text: "Dependencies installed successfully.\n" }
    },
    {
      method: "notify",
      params: {
        html: "Update complete! Click 'Start' to launch DramaBox."
      }
    }
  ]
}

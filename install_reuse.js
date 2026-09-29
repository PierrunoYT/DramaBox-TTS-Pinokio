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
      // Upstream RE-USE imports mamba_ssm, which has no Windows build.
      when: "{{platform !== 'linux'}}",
      method: "notify",
      params: {
        html: "RE-USE voice denoising requires Linux: its mamba-ssm dependency has no Windows build. DramaBox works without it."
      },
      next: null
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: "uv pip install setuptools wheel packaging ninja"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install --no-build-isolation -c ../constraints.txt -r ../requirements.txt -r ../requirements_low_vram.txt -r ../requirements-reuse.txt",
          "uv pip check"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "RE-USE installed! Voice-reference denoising is now enabled."
      }
    }
  ]
}

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
        venv: "env",
        path: "app",
        message: "uv pip install -r ../requirements-reuse.txt"
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

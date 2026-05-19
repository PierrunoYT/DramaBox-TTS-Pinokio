module.exports = {
  run: [
    // nvidia windows
    {
      when: "{{gpu === 'nvidia' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps"
        ]
      },
      next: null
    },
    // nvidia linux
    {
      when: "{{gpu === 'nvidia' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        bluefairy: "off",
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall"
        ]
      },
      next: null
    },
    // amd windows
    {
      when: "{{gpu === 'amd' && platform === 'win32'}}",
      method: "shell.run",
      params: {
        bluefairy: "off",
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message:
          "uv pip install torch torch-directml torchaudio torchvision numpy==1.26.4 --force-reinstall"
      },
      next: null
    },
    // amd linux (rocm)
    {
      when: "{{gpu === 'amd' && platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message:
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall --no-deps"
      },
      next: null
    },
    // cpu (fallback when no supported GPU backend is detected)
    {
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : null}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message:
          "uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps"
      }
    }
  ]
}

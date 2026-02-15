module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        path: ".",
        message: [
          "uv python install 3.12",
          "uv venv --python 3.12 env",
        ],
      },
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: ".",
        },
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "uv pip install --upgrade pip setuptools wheel",
          "uv pip install opencv-python gradio",
          "uv pip install -e packages/ltx-core -e packages/ltx-pipelines",
        ],
      },
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "python -c \"from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ('inputs', 'models', 'outputs')]\"",
        ],
      },
    },
    {
      method: "notify",
      params: {
        html: "Install complete. Put checkpoints under <code>models/</code>, then click Start.",
      },
    },
  ],
};

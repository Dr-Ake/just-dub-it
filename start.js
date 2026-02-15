module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: ".",
        message: [
          "python pinokio_app.py --host 0.0.0.0 --port 7860",
        ],
        on: [
          {
            event: "/http:\\/\\/\\S+/",
            done: true,
          },
        ],
      },
    },
    {
      when: "{{input && input.event && Array.isArray(input.event) && input.event.length > 0}}",
      method: "local.set",
      params: {
        url: "{{input.event[0]}}",
      },
    },
  ],
};

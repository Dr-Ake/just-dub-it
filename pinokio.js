const normalizeUrl = (url) => {
  if (!url) {
    return url;
  }
  return url.includes("0.0.0.0") ? url.replace("0.0.0.0", "127.0.0.1") : url;
};

module.exports = {
  version: "1.0",
  title: "JustDubit",
  description: "Video dubbing with the JustDubit AV diffusion pipeline.",
  menu: async (kernel, info) => {
    const installed = info.exists("env");
    const running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      update: info.running("update.js"),
    };

    if (running.install) {
      return [
        {
          default: true,
          icon: "fa-solid fa-plug",
          text: "Installing",
          href: "install.js",
        },
      ];
    }

    if (!installed) {
      return [
        {
          default: true,
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        },
      ];
    }

    if (running.start) {
      const local = info.local("start.js");
      if (local && local.url) {
        return [
          {
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open UI",
            href: normalizeUrl(local.url),
          },
          {
            icon: "fa-solid fa-terminal",
            text: "Terminal",
            href: "start.js",
          },
        ];
      }
      return [
        {
          default: true,
          icon: "fa-solid fa-terminal",
          text: "Terminal",
          href: "start.js",
        },
      ];
    }

    if (running.update) {
      return [
        {
          default: true,
          icon: "fa-solid fa-rotate",
          text: "Updating",
          href: "update.js",
        },
      ];
    }

    return [
      {
        default: true,
        icon: "fa-solid fa-power-off",
        text: "Start",
        href: "start.js",
      },
      {
        icon: "fa-solid fa-rotate",
        text: "Update",
        href: "update.js",
      },
      {
        icon: "fa-solid fa-plug",
        text: "Install / Repair",
        href: "install.js",
      },
    ];
  },
};

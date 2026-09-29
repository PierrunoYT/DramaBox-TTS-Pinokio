module.exports = {
  version: "5.0",
  menu: async (kernel, info) => {
    let installed = info.exists("app/env") && info.exists("app/.installed")
    let running = {
      install: info.running("install.js"),
      start: info.running("start.js"),
      low_vram: info.running("start_low_vram.js"),
      update: info.running("update.js"),
      reset: info.running("reset.js"),
      link: info.running("link.js"),
      reuse: info.running("install_reuse.js")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.js",
      }]
    } else if (running.update || running.reset || running.link || running.reuse) {
      const operations = [
        [running.update, "Updating", "update.js"],
        [running.reset, "Resetting", "reset.js"],
        [running.link, "Deduplicating", "link.js"],
        [running.reuse, "Installing RE-USE", "install_reuse.js"]
      ]
      const [, text, href] = operations.find(([active]) => active)
      return [{ default: true, icon: "fa-solid fa-terminal", text, href }]
    } else if (installed) {
      if (running.start || running.low_vram) {
        let start_script = running.low_vram ? "start_low_vram.js" : "start.js"
        let local = info.local(start_script)
        if (local && local.url) {
          return [{
            default: true,
            icon: "fa-solid fa-rocket",
            text: "Open Web UI",
            href: local.url,
          }, {
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: start_script,
          }]
        } else {
          return [{
            default: true,
            icon: 'fa-solid fa-terminal',
            text: "Terminal",
            href: start_script,
          }]
        }
      } else {
        let items = [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.js",
        }]
        // MMGP can only offload to a CUDA device, so only show Low VRAM on NVIDIA.
        if (kernel.gpu === "nvidia") {
          items.push({
            icon: "fa-solid fa-memory",
            text: "<div><strong>Start Low VRAM</strong><div>Experimental MMGP offload mode</div></div>",
            href: "start_low_vram.js",
          })
        }
        items.push({
          icon: "fa-solid fa-download",
          text: "Model on HuggingFace",
          href: "https://huggingface.co/ResembleAI/Dramabox",
          popout: true
        }, {
          icon: "fa-solid fa-plug",
          text: "Update",
          href: "update.js",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.js",
        })
        // RE-USE needs mamba-ssm, which only builds on Linux.
        if (kernel.platform === "linux") {
          items.push({
            icon: "fa-solid fa-wand-magic-sparkles",
            text: "<div><strong>Install RE-USE</strong><div>Optional voice-reference denoising</div></div>",
            href: "install_reuse.js",
          })
        }
        return items.concat([{
          icon: "fa-solid fa-file-zipper",
          text: "<div><strong>Save Disk Space</strong><div>Deduplicates redundant library files</div></div>",
          href: "link.js",
        }, {
          icon: "fa-regular fa-circle-xmark",
          text: "<div><strong>Reset</strong><div>Revert to pre-install state</div></div>",
          href: "reset.js",
          confirm: "Are you sure you wish to reset the app?"
        }])
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.js",
      }]
    }
  }
}

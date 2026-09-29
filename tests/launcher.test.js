const assert = require('node:assert/strict')
const test = require('node:test')
const { menu } = require('../pinokio.js')

function info(files = [], active = '', url) {
  return {
    exists: path => files.includes(path),
    running: script => script === active,
    local: () => ({ url })
  }
}

test('partial installs never expose Start', async () => {
  for (const files of [[], ['app/env'], ['app/.installed']]) {
    const items = await menu({ gpu: 'nvidia' }, info(files))
    assert.equal(items[0].href, 'install.js')
    assert.equal(items.length, 1)
  }
})

test('maintenance stays visible after the environment or marker disappears', async () => {
  for (const active of ['install.js', 'update.js', 'reset.js', 'link.js', 'install_reuse.js']) {
    const items = await menu({}, info([], active))
    assert.equal(items[0].href, active)
    assert.equal(items[0].default, true)
    assert.equal(items.length, 1)
  }
})

test('ready and running menus select the correct launch mode', async () => {
  const files = ['app/env', 'app/.installed']
  const ready = await menu({ gpu: 'nvidia' }, info(files))
  assert.equal(ready[0].href, 'start.js')
  assert.equal(ready[0].default, true)
  assert(ready.some(item => item.href === 'start_low_vram.js'))
  for (const active of ['start.js', 'start_low_vram.js']) {
    const starting = await menu({}, info(files, active))
    assert.equal(starting[0].href, active)
    const running = await menu({}, info(files, active, 'http://127.0.0.1:8123'))
    assert.equal(running[0].href, 'http://127.0.0.1:8123')
    assert.equal(running[1].href, active)
  }
})

test('RE-USE is offered and installed only on Linux', async () => {
  const files = ['app/env', 'app/.installed']
  const offered = async platform => (await menu({ gpu: 'nvidia', platform }, info(files)))
    .some(item => item.href === 'install_reuse.js')
  assert.equal(await offered('linux'), true)
  assert.equal(await offered('win32'), false)
  const script = require('../install_reuse.js')
  const first = script.run.find(step => enabled(step, 'win32', 'nvidia', new Set()))
  assert.equal(first.method, 'notify')
  assert.equal(first.next, null)
})

function enabled(step, platform, gpu, files) {
  if (!step.when) return true
  const expression = step.when.slice(2, -2)
  return Function('platform', 'gpu', 'exists', `return (${expression})`)(
    platform, gpu, path => files.has(path))
}

test('unsupported devices stop before any shell command', () => {
  for (const file of ['install', 'update', 'install_reuse', 'start', 'start_low_vram', 'torch']) {
    const script = require(`../${file}.js`)
    for (const [platform, gpu] of [['darwin', 'apple'], ['win32', 'amd'], ['linux', 'cpu']]) {
      const first = script.run.find(step => enabled(step, platform, gpu, new Set()))
      assert.equal(first.method, 'notify', file)
      assert.equal(first.next, null, file)
    }
  }
})

test('only completed dependency installs set the readiness marker', () => {
  for (const file of ['install', 'update']) {
    const script = require(`../${file}.js`)
    for (const failAt of ['uv pip install', 'torch.js', 'uv pip check', null]) {
      const files = new Set(['app', 'app/env', 'app/.installed'])
      for (const step of script.run) {
        if (!enabled(step, 'win32', 'nvidia', files)) continue
        if (failAt && JSON.stringify(step.params).includes(failAt)) break
        if (step.method === 'fs.rm') files.delete(step.params.path)
        if (step.method === 'fs.write') files.add(step.params.path)
      }
      assert.equal(files.has('app/.installed'), failAt === null, `${file}: ${failAt}`)
    }
  }
})

test('both launchers capture the URL in group one and keep the server alive', () => {
  for (const file of ['start', 'start_low_vram']) {
    const script = require(`../${file}.js`)
    const shell = script.run.find(step => step.method === 'shell.run')
    const pattern = shell.params.on[0].event
    const match = new RegExp(pattern.slice(1, -1)).exec('Running on local URL:  http://127.0.0.1:8123')
    assert.equal(match[1], 'http://127.0.0.1:8123')
    assert.equal(script.run.at(-1).params.url, '{{input.event[1]}}')
    assert.equal(shell.params.on[0].done, true)
    assert.equal(script.daemon, true)
  }
})

// The model runs in one offscreen document, shared by every site; it reads the model files bundled in the extension package.
// This worker only makes sure that document exists.
let creating;

async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  creating ??= chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['DOM_PARSER'],
    justification: 'Parses the page HTML sent by the reader button and runs the local Neko.js model with WebGPU.',
  }).finally(() => { creating = undefined; });
  await creating;
}

chrome.runtime.onMessage.addListener((message, sender, reply) => {
  if (message?.type !== 'neko:ensure-offscreen' || sender.id !== chrome.runtime.id) return false;
  ensureOffscreen().then(() => reply({ ok: true }), (error) => reply({ ok: false, message: String(error?.message ?? error) }));
  return true;
});

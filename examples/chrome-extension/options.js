import { DEFAULT_MODEL_BASE } from './config.js';

const input = document.querySelector('#model-base');
const status = document.querySelector('#status');
input.placeholder = DEFAULT_MODEL_BASE;
input.value = (await chrome.storage.local.get({ modelBase: '' })).modelBase || DEFAULT_MODEL_BASE;

document.querySelector('#save').addEventListener('click', async () => {
  let url;
  try { url = new URL(input.value); } catch { status.textContent = '不是有效的 URL。'; return; }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) { status.textContent = '只接受不含帳密的 http(s) URL。'; return; }
  if (!url.pathname.endsWith('/')) url.pathname += '/';
  await chrome.storage.local.set({ modelBase: url.href === DEFAULT_MODEL_BASE ? '' : url.href });
  input.value = url.href; status.textContent = '已儲存。';
});
document.querySelector('#reset').addEventListener('click', async () => {
  await chrome.storage.local.set({ modelBase: '' });
  input.value = DEFAULT_MODEL_BASE; status.textContent = '已恢復預設。';
});

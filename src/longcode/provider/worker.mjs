// Model IO only. LongCode's Python runtime owns tools, permissions and task state.
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import {createModels, createProvider, envApiKeyAuth} from '@earendil-works/pi-ai';
import {openaiProvider} from '@earendil-works/pi-ai/providers/openai';
import {openaiCodexProvider} from '@earendil-works/pi-ai/providers/openai-codex';
import {anthropicProvider} from '@earendil-works/pi-ai/providers/anthropic';
import {openAICompletionsApi} from '@earendil-works/pi-ai/api/openai-completions.lazy';

// Python holds auth.lock for the worker's lifetime (including token refresh).
// Never run this internal worker as a standalone credential writer.
const controller = new AbortController();
const replies = new Map();
let request;
let secrets = [];
const output = (type, data = {}) => process.stdout.write(JSON.stringify({id: request?.id, type, ...data}) + '\n');
const safeError = (e) => {
  let text = String(e?.message || e);
  for (const secret of secrets) if (secret) text = text.split(secret).join('[REDACTED]');
  // Provider login errors can contain a response body with newly issued tokens.
  return text.replace(/(sk-[\w-]+|eyJ[\w.-]{30,})/g, '[REDACTED]').slice(0, 1500);
};

export class FileCredentials {
  constructor(home) { this.file = path.join(home, 'auth.json'); this.queue = Promise.resolve(); }
  async all() {
    try { return JSON.parse(await fs.readFile(this.file, 'utf8')); }
    catch (e) { if (e.code === 'ENOENT') return {}; throw e; }
  }
  async read(id) { return (await this.all())[id]; }
  async list() { return Object.entries(await this.all()).map(([providerId, v]) => ({providerId, type: v.type})); }
  async write(data) {
    await fs.mkdir(path.dirname(this.file), {recursive: true, mode: 0o700});
    const temp = this.file + '.' + process.pid + '.tmp';
    const file = await fs.open(temp, 'w', 0o600);
    try { await file.writeFile(JSON.stringify(data)); await file.sync(); } finally { await file.close(); }
    await fs.chmod(temp, 0o600);
    await fs.rename(temp, this.file);
  }
  async modify(id, fn) {
    const operation = this.queue.then(async () => {
      const data = await this.all();
      const value = await fn(data[id]);
      if (value !== undefined) { data[id] = value; await this.write(data); }
      return data[id];
    });
    this.queue = operation.catch(() => {});
    return operation;
  }
  async delete(id) {
    const operation = this.queue.then(async () => {
      const data = await this.all(); delete data[id]; await this.write(data);
    });
    this.queue = operation.catch(() => {}); return operation;
  }
}

async function main(r) {
  const credentials = new FileCredentials(r.home);
  for (const c of Object.values(await credentials.all())) secrets.push(c.key, c.access, c.refresh);
  const models = createModels({credentials, authContext: {env: async () => undefined, fileExists: async () => false}});
  models.setProvider(openaiProvider());
  models.setProvider(openaiCodexProvider());
  models.setProvider(anthropicProvider());
  const settings = r.settings || {};
  const providerId = r.provider || settings.provider || 'openai-codex';
  if (!['openai', 'openai-codex', 'anthropic', 'compatible'].includes(providerId)) throw Error('Unsupported provider');
  const custom = {
    id: settings.model || 'custom', name: settings.model || 'Custom endpoint', provider: 'compatible',
    api: 'openai-completions', baseUrl: settings.base_url || 'http://localhost:11434/v1',
    reasoning: false, input: ['text'], cost: {input: 0, output: 0, cacheRead: 0, cacheWrite: 0},
    contextWindow: settings.context_window || 64000, maxTokens: settings.max_output_tokens || 8192,
  };
  models.setProvider(createProvider({id: 'compatible', name: 'Compatible API', baseUrl: custom.baseUrl,
    auth: {apiKey: envApiKeyAuth('Compatible API key', [])}, models: [custom], api: openAICompletionsApi()}));
  if (r.op === 'status') {
    const stored = await credentials.list();
    output('result', {value: {credentials: stored, models: models.getModels(providerId).map(m => ({
      id: m.id, name: m.name, reasoning: m.reasoning, context_window: m.contextWindow,
    }))}}); return;
  }
  if (r.op === 'key') {
    if (providerId === 'openai-codex') throw Error('ChatGPT subscription requires OAuth');
    if (!r.key || typeof r.key !== 'string') throw Error('API key cannot be empty');
    secrets.push(r.key);
    await credentials.modify(providerId, async () => ({type: 'api_key', key: r.key}));
    output('result', {value: {saved: true}}); return;
  }
  if (r.op === 'logout') { await models.logout(providerId); output('result', {value: {logged_out: true}}); return; }
  if (r.op === 'login') {
    if (providerId !== 'openai-codex') throw Error('Use API key or Claude CLI login for this provider');
    await models.login(providerId, 'oauth', {
      signal: controller.signal,
      notify: event => output('auth_event', {event}),
      prompt: prompt => {
        const promptId = crypto.randomUUID();
        const {signal, ...display} = prompt;
        if (prompt.type === 'select' && r.method) return Promise.resolve(r.method);
        output('auth_prompt', {prompt_id: promptId, prompt: display});
        return new Promise((resolve, reject) => {
          const abort = () => { replies.delete(promptId); output('auth_prompt_cancelled', {prompt_id: promptId}); reject(Error('Login prompt cancelled')); };
          if (signal?.aborted) { abort(); return; }
          signal?.addEventListener('abort', abort, {once: true});
          replies.set(promptId, value => {signal?.removeEventListener('abort', abort); resolve(value);});
        });
      },
    });
    output('result', {value: {logged_in: true}}); return;
  }
  if (r.op !== 'model') throw Error('Unknown operation');
  if (!settings.model) throw Error('请先在设置中选择模型');
  if (providerId === 'compatible' && !settings.base_url) throw Error('兼容 API 需要填写服务地址');
  let model = models.getModel(providerId, settings.model);
  if (!model) throw Error('模型不在当前组件目录中；自定义模型请选择 compatible 并填写服务地址');
  if (settings.reasoning && !model.reasoning) throw Error('所选模型不支持推理强度设置');
  if (settings.base_url) {
    if (providerId === 'openai-codex') throw Error('不能把订阅凭据发送到自定义地址');
    model = {...model, baseUrl: settings.base_url};
  }
  const stream = models.streamSimple(model, r.context, {
    signal: controller.signal, maxTokens: Math.min(settings.max_output_tokens || 8192, model.maxTokens),
    ...(settings.reasoning ? {reasoning: settings.reasoning} : {}),
    sessionId: r.session_id, transport: 'sse',
  });
  for await (const event of stream) {
    if (event.type === 'text_delta') output('text_delta', {text: event.delta});
  }
  const message = await stream.result();
  if (message.stopReason === 'error' || message.stopReason === 'aborted') throw Error(message.errorMessage || message.stopReason);
  // No headers/credentials in the protocol. Provider-returned content is the only model trace.
  output('result', {value: message});
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(new URL(import.meta.url).pathname)) {
  const input = readline.createInterface({input: process.stdin});
  input.on('line', line => {
    let item;
    try { item = JSON.parse(line); } catch { output('error', {message: 'Invalid JSON request'}); return; }
    if (item.op === 'cancel') { controller.abort(); return; }
    if (item.op === 'reply') { const reply = replies.get(item.prompt_id); replies.delete(item.prompt_id); reply?.(item.value); return; }
    if (request) return;
    request = item;
    main(item).catch(e => output('error', {message: item.op === 'login' ? '登录未完成，请重试或检查账户访问权限' : safeError(e)}))
      .finally(() => {input.close(); process.stdin.pause(); process.exitCode = 0;});
  });
  input.on('close', () => { if (!request) controller.abort(); });
}

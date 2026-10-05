import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {FileCredentials} from './worker.mjs';

test('private atomic credentials: concurrent writes, refresh and logout', async () => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'longcode-auth-test-'));
  try {
    const store = new FileCredentials(dir);
    await Promise.all([
      store.modify('openai', async () => ({type:'api_key',key:'fixture-not-a-real-key'})),
      store.modify('openai-codex', async () => ({type:'oauth',access:'old',refresh:'r1',expires:0})),
    ]);
    const refreshed = await store.modify('openai-codex', async current => ({...current,access:'new',refresh:'r2',expires:999}));
    assert.equal(refreshed.refresh, 'r2');
    assert.equal((await store.list()).length, 2);
    assert.ok(!JSON.stringify(await store.list()).includes('fixture'));
    assert.equal((await fs.stat(path.join(dir,'auth.json'))).mode & 0o777, 0o600);
    await assert.rejects(store.modify('openai-codex', async () => {throw Error('refresh failed');}));
    assert.equal((await store.read('openai-codex')).access, 'new');
    await store.delete('openai');
    assert.equal(await store.read('openai'), undefined);
    assert.equal((await store.list()).length, 1);
  } finally {await fs.rm(dir, {recursive:true, force:true});}
});

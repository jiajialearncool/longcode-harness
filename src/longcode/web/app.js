'use strict';
const $ = id => document.getElementById(id);
let token = new URLSearchParams(location.hash.slice(1)).get('token') || sessionStorage.getItem('longcode-token');
if (token) {sessionStorage.setItem('longcode-token', token); history.replaceState(null, '', '/');}
let sid = sessionStorage.getItem('longcode-session'), cursor = 0, state = null, settings = null, authId = null, authSeen = 0;
const statusNames = {idle:'待命',running:'工作中',interrupted:'已中断',error:'需要处理',completed:'已完成',paused:'已暂停',blocked:'受阻',waiting_input:'等待补充信息',verified:'已验证',pending:'待办',in_progress:'执行中',failed:'检查未通过',needs_revalidation:'需要重新验证'};
const lines = value => value.split('\n').map(s => s.trim()).filter(Boolean);
function node(tag, text, className) {const n = document.createElement(tag); if(text !== undefined)n.textContent=text; if(className)n.className=className; return n;}
async function request(path, data, raw=false) {
  const r = await fetch('/api'+path, {method:data === undefined?'GET':'POST', headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json'},body:data === undefined?undefined:JSON.stringify(data)});
  if(!r.ok){const e=await r.json();throw Error(e.error || r.statusText);} return raw?r.text():r.json();
}
function guard(fn){return async (...args)=>{try{$('notice').textContent='';await fn(...args);}catch(e){$('notice').textContent=e.message;if($('settings-dialog').open)$('settings-notice').textContent=e.message;}};}
async function refreshSessions(){const sessions=await request('/sessions');$('sessions').replaceChildren();for(const s of sessions){const b=node('button',s.title,'session');b.append(node('small',statusNames[s.status]||s.status));b.onclick=guard(()=>select(s.id));$('sessions').append(b);}}
async function select(id){sid=id;sessionStorage.setItem('longcode-session',sid);cursor=0;state=null;$('events').replaceChildren();$('diffs').replaceChildren();$('live').textContent='';$('draft').classList.add('hidden');$('draft').dataset.source='';$('task-state').textContent='还没有启动长程任务。';$('task-events').textContent='';$('objective').value='';await poll();}
function displayMessages(messages){$('messages').replaceChildren();for(const m of messages){if(!['user','assistant'].includes(m.role))continue;const text=typeof m.content==='string'?m.content:m.content.filter(x=>x.type==='text').map(x=>x.text).join('\n');if(!text)continue;const box=node('div',undefined,'message '+m.role);box.append(node('div',m.role==='user'?'你':'LongCode','who'),node('div',text,'text'));$('messages').append(box);}}
function showQuestions(target, questions, base){
  const signature=JSON.stringify(questions);if(target.dataset.signature===signature)return;target.dataset.signature=signature;target.replaceChildren();
  for(const q of questions){const box=node('div',undefined,'question');box.append(node('p',q.message||'需要补充信息'));
    const extras={...q};for(const k of ['id','message','type','kind','placeholder'])delete extras[k];if(Object.keys(extras).length)box.append(node('pre',JSON.stringify(extras,null,2)));
    const answer=async value=>{await request(base+'/answer',{id:q.id,value});target.dataset.signature='';};
    if(q.kind==='permission'){for(const [label,value]of [['允许这一次','allow'],['拒绝','deny']]){const b=node('button',label);b.onclick=guard(()=>answer(value));box.append(b);}}
    else{const input=node('input');if(['secret','manual_code'].includes(q.type))input.type='password';input.placeholder=q.placeholder||'填写回答';const b=node('button','提交');b.onclick=guard(()=>answer(input.value));box.append(input,b);}target.append(box);
  }
}
function eventCard(event){const box=node('details',undefined,'event');box.append(node('summary',`${event.sequence} · ${event.type}`),node('small',event.timestamp),node('pre',JSON.stringify(event.data,null,2)));return box;}
async function poll(){if(!sid)return;const selected=sid;const current=await request('/sessions/'+sid);if(selected!==sid)return;const statusChanged=!state||state.status!==current.status;state=current;if(statusChanged)await refreshSessions();
  $('title').textContent=current.mode==='task'?'长程任务':'编程对话';$('workspace').textContent=current.workspace;$('status').textContent=statusNames[current.status]||current.status;
  const fingerprint=JSON.stringify(current.messages);if($('messages').dataset.fingerprint!==fingerprint){displayMessages(current.messages);$('messages').dataset.fingerprint=fingerprint;}
  showQuestions($('questions'),current.pending||[],'/sessions/'+sid);
  if(current.error)$('notice').textContent=current.error;
  if(current.draft&&$('draft').dataset.source!==JSON.stringify(current.draft)){const d=current.draft;$('draft').classList.remove('hidden');$('objective').value=d.objective;$('acceptance').value=d.acceptance.join('\n');$('checks').value=d.checks.join('\n');$('missing').textContent=(d.missing||[]).join('；');$('draft').dataset.source=JSON.stringify(d);}
  if(current.task_state){const ts=current.task_state;$('task-state').replaceChildren(node('p','任务状态：'+(statusNames[ts.status]||ts.status)+(ts.blocker?' · '+ts.blocker:'')));const table=node('table');for(const [id,item]of Object.entries(ts.criteria||{})){const row=node('tr');row.append(node('td',id),node('td',item.description||item.criterion||''),node('td',statusNames[item.status]||item.status));table.append(row);}$('task-state').append(table);$('task-events').textContent=JSON.stringify(await request('/sessions/'+sid+'/task-events'),null,2);}
  const events=await request('/sessions/'+sid+'/events?after='+cursor);if(selected!==sid)return;
  for(const e of events){cursor=e.sequence;if(e.type==='text_delta'){$('live').textContent=($('live').textContent+e.data.text).slice(-20000);continue;}$('events').append(eventCard(e));if(e.type==='file_changed'){$('diffs').append(node('h3',e.data.path),node('pre',e.data.diff));}}
  if(current.status!=='running')$('live').textContent='';
  $('send-form').querySelector('button').disabled=current.status==='running'||current.mode==='task';
}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tab').forEach(n=>n.classList.toggle('hidden',n.id!==b.dataset.tab));document.querySelectorAll('[data-tab]').forEach(n=>n.classList.toggle('active',n===b));});
$('new-session').onclick=()=> $('project-dialog').showModal();$('project-cancel').onclick=()=> $('project-dialog').close();
$('project-form').onsubmit=guard(async e=>{e.preventDefault();const s=await request('/sessions',{workspace:$('project-path').value});$('project-dialog').close();await select(s.id);await refreshSessions();});
$('send-form').onsubmit=guard(async e=>{e.preventDefault();if(!sid)throw Error('请先新建会话并选择项目');const text=$('prompt').value;await request('/sessions/'+sid+'/message',{text,request_id:crypto.randomUUID()});$('prompt').value='';await poll();});
$('stop').onclick=guard(async()=>{if(sid)await request('/sessions/'+sid+'/stop',{});});
$('prepare').onclick=guard(async()=>{if(!sid)throw Error('请先选择项目');await request('/sessions/'+sid+'/prepare',{objective:$('objective').value,request_id:crypto.randomUUID()});await poll();});
$('start-task').onclick=guard(async()=>{if(!sid)throw Error('请先选择项目');const draft={objective:$('objective').value,acceptance:lines($('acceptance').value),checks:lines($('checks').value)};if(!confirm('确认按这些验收要求修改项目，并运行列出的检查命令？'))return;await request('/sessions/'+sid+'/task',{draft,request_id:crypto.randomUUID()});await poll();});
$('resume').onclick=guard(async()=>{if(sid)await request('/sessions/'+sid+'/resume',{request_id:crypto.randomUUID()});});
async function exportFile(format){if(!sid)throw Error('请先选择会话');const text=await request('/sessions/'+sid+'/export?format='+format,undefined,true);const url=URL.createObjectURL(new Blob([text],{type:'text/plain;charset=utf-8'}));const a=node('a');a.href=url;a.download='LongCode-'+sid+'.'+format;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('export-md').onclick=guard(()=>exportFile('md'));$('export-jsonl').onclick=guard(()=>exportFile('jsonl'));
$('settings-open').onclick=guard(async()=>{settings=await request('/settings');for(const k of ['backend','provider','model','reasoning'])$(k).value=settings[k];$('base-url').value=settings.base_url;$('roles').value=JSON.stringify(settings.roles,null,2);$('skills').value=settings.skills.join('\n');$('mcp').value=JSON.stringify(settings.mcp,null,2);$('settings-dialog').showModal();});
$('save-settings').onclick=guard(async()=>{settings={...settings,backend:$('backend').value,provider:$('provider').value,model:$('model').value,reasoning:$('reasoning').value,base_url:$('base-url').value,roles:JSON.parse($('roles').value),skills:lines($('skills').value),mcp:JSON.parse($('mcp').value)};await request('/settings',settings);$('settings-notice').textContent='已保存。下一次工作会使用这些设置。';});
$('models-refresh').onclick=guard(async()=>{const result=await request('/auth/status?provider='+encodeURIComponent($('provider').value));$('model-list').replaceChildren();for(const m of result.models){const option=node('option');option.value=m.id;option.label=m.name;$('model-list').append(option);}$('auth-status').textContent=result.credentials.map(x=>x.providerId+'：已保存'+(x.type==='oauth'?'订阅凭据':'密钥')).join('；')||'尚未保存登录凭据。';});
$('save-key').onclick=guard(async()=>{await request('/auth/key',{provider:$('provider').value,key:$('api-key').value});$('api-key').value='';$('settings-notice').textContent='密钥已保存。';});
$('logout').onclick=guard(async()=>{await request('/auth/logout',{provider:$('provider').value});$('settings-notice').textContent='已退出所选服务。';});
async function login(method){const result=await request('/auth/start',{provider:'openai-codex',method});authId=result.id;authSeen=0;$('auth-events').replaceChildren();}
$('login-browser').onclick=guard(()=>login('browser'));$('login-device').onclick=guard(()=>login('device_code'));
$('cancel-login').onclick=guard(async()=>{if(authId)await request('/auth/'+authId+'/cancel',{});});
async function pollAuth(){if(!authId)return;const j=await request('/auth/'+authId);for(const item of j.events.slice(authSeen)){const e=item.data.event||{};if(e.type==='auth_url'){const a=node('a','打开登录页面');a.href=e.url;a.target='_blank';a.rel='noopener noreferrer';$('auth-events').append(a);}else if(e.type==='device_code'){$('auth-events').append(node('p','设备码：'+e.userCode));const a=node('a','打开设备验证页面');a.href=e.verificationUri;a.target='_blank';a.rel='noopener noreferrer';$('auth-events').append(a);}else if(e.message)$('auth-events').append(node('p',e.message));}authSeen=j.events.length;showQuestions($('auth-questions'),j.pending,'/auth/'+authId);if(j.status!=='running'){$('auth-status').textContent=j.status==='completed'?'登录完成。请选择模型并保存设置。':j.error;authId=null;}}
let polling=false;setInterval(async()=>{if(polling)return;polling=true;try{await poll();await pollAuth();}catch(e){$('notice').textContent=e.message;}finally{polling=false;}},800);
guard(async()=>{if(!token){token=prompt('请粘贴 longcode web 链接中的本地访问令牌');if(token)sessionStorage.setItem('longcode-token',token);}await refreshSessions();if(sid)await select(sid);})();

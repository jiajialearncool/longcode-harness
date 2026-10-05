# LongCode 自主编程助手使用说明

## 现在能做什么

LongCode 自己执行“调用模型 → 执行工具 → 交回结果 → 继续工作”的循环。
Python 负责文件、命令、权限和长程任务。Node.js 进程只连接模型服务和处理登录，不负责执行工具。
模型连接组件固定为 `@earendil-works/pi-ai@0.87.1`；依赖版本锁和 MIT 许可保留在 `src/longcode/provider/`。

三种方式可以在网页设置中选择，也可以按任务管理器、执行器和审计器分别设置。

| 方式 | 模型与工具由谁运行 | 凭据在哪里 |
| --- | --- | --- |
| `native` 自主执行 | LongCode | LongCode 的独立本机目录 |
| `codex` | Codex CLI | Codex 自己的登录目录 |
| `claude` | Claude CLI | Claude 自己的登录目录 |

自主执行不会启动 Codex CLI、Claude CLI 或 Pi Agent。程序中复用旧后端的报告格式，不等于运行旧后端。

## 安装与第一次使用

当前完整支持 macOS。需要 Python 3.11+、Node.js 22.19+ 和 npm。

```bash
# 在本项目目录执行；无需先安装 Python 包
./longcode setup
./longcode doctor --backend native
./longcode web
```

`setup` 根据锁文件安装模型组件，不运行依赖安装脚本。它需要联网。
也可以在独立 Python 虚拟环境中运行 `pip install .`，然后使用 `longcode` 命令；安装后仍需运行 `longcode setup`。

网页依次操作：

1. 打开“登录与设置”，选择自主执行。
2. 选择模型服务，完成登录或填写 API key。
3. 获取模型列表，选择模型，保存设置。
4. 新建会话，填写具体项目的绝对路径。
5. 在“对话”中说明希望修改什么。运行命令前会要求你批准。

网页链接包含本机访问令牌，不要分享。服务只监听 `127.0.0.1`；同一个用户的终端和网页复用该服务。
刷新网页不会重新提交任务。同一项目不能同时启动两个写入任务。

### ChatGPT 订阅

```bash
./longcode login --provider openai-codex
# 或设备码方式（账户和上游组件必须支持）
./longcode login --provider openai-codex --method device_code
./longcode config --backend native --provider openai-codex --model '填写模型列表里的名称'
```

凭据由 Pi 的认证组件获取和刷新。网页也支持粘贴登录流程要求的回调信息；终端优先使用浏览器自动回调，回调不可用时可改用设备码或网页。
登录失败不会自动切换为付费 API。此连接是需要实际联调的兼容能力，不是 OpenAI 对第三方应用的长期接口保证。

### API key

```bash
./longcode login --provider openai
./longcode config --backend native --provider openai --model '填写所选模型名称'

# Anthropic API
./longcode login --provider anthropic
./longcode config --backend native --provider anthropic --model '填写所选模型名称'

# 兼容 OpenAI Chat Completions 的服务
./longcode login --provider compatible
./longcode config --backend native --provider compatible \
  --base-url 'https://你的服务地址/v1' --model '你的模型名称' --reasoning ''
```

命令行交互输入密钥，不把密钥写入命令参数。兼容服务当前不统一映射推理强度，留空即可。
OpenAI 和 Anthropic 使用组件模型目录中的名称。网页显示的是组件目录，不是账户实际可用模型的实时探测结果。
不支持自行登录 Claude 订阅；Claude 订阅请使用 Claude CLI。

### CLI 接入

```bash
./longcode login --backend codex
./longcode config --backend codex --model '' --reasoning ''

./longcode login --backend claude
./longcode config --backend claude --model '' --reasoning ''
```

先安装对应 CLI。产品接入不使用评测专用的模型、缓存或权限绕过参数。
Codex 使用自身文件沙箱；Claude 使用默认权限模式。非交互 CLI 无法完成的授权会返回错误，需要按 CLI 自身机制配置后重试，不会自动跳过安全检查。
Claude 的推理强度暂不映射，填写时会明确报错。

## 普通对话与长程任务

```bash
./longcode chat --workspace /path/to/project
./longcode chat --session 已有会话编号
```

普通对话保留必要历史，直接修改选定项目。历史过长时整理摘要，并保存原始运行记录。
停止操作不会撤销已经写入的文件。建议在独立开发分支工作，修改前保留备份。

终端输入 `/task 目标`，或进入网页“长程任务”，会先整理验收建议：

- 从用户目标、此前对话和项目文件提出验收要求。
- 查找项目已有的测试、类型检查、构建命令。
- 把建议展示给用户，用户可以修改并确认。

没有必要检查时，不会启动长程验收。可以先在普通对话中补充测试，再审查检查方法。
拒绝空检查和常见恒定成功命令；这不能证明任意检查都可信，仍需用户确认它确实覆盖需求。

长程任务中的四个模块保持原有分工：

- **任务管理器**：根据已验证进度安排下一项工作。普通安排按现有规则执行，必要时调用模型调整方案。
- **执行器**：在候选工作区处理一个子任务，独立提供本次资料，不直接续接上一轮完整对话。
- **自动验证器**：运行确认过的检查，汇总结果。检查在临时副本中进行，生成文件不进入正式项目。
- **审计器**：独立读取产物和证据，返回复核意见。自主模式不向它开放写文件、任意命令和 MCP 工具。

失败的修改不能成为已验证进度。通过检查还需要满足原有接纳条件，才能合入项目。
当任务要求用户补充信息时，网页或终端会显示问题；回答会保存到任务记录。停止后可用 `/resume` 或网页“继续已有长程任务”。
异常恢复会先核对现有文件和已保存的接纳记录；遇到不一致会暂停，不直接重放上次命令。

## 设置岗位、Skills 和 MCP

网页“岗位、Skills 和 MCP 设置”支持各岗位覆盖：

```json
{
  "executor": {"backend": "native", "provider": "openai-codex", "model": "填写实际模型名称"},
  "auditor": {"backend": "claude", "model": "", "reasoning": ""}
}
```

未覆盖的字段继承通用设置。`E1`、`E2`、`E3` 可分别配置执行档位。
选择 Claude 时应清空继承的推理强度。没有相应账户、CLI 或接入能力，填写名称本身不会使它可用。

自主执行读取项目根目录和相关子目录的 `AGENTS.md`。Skills 由用户明确列出 `SKILL.md` 的绝对路径。
当前支持加载这些文件的指令，不包含技能市场，也不会自动安装 Skill 的依赖。

MCP 示例：

```json
[
  {"name": "local-tool", "command": ["/usr/local/bin/node", "/path/to/project/tools/server.mjs"], "network": false},
  {"name": "remote-tool", "url": "https://你的服务地址/mcp"}
]
```

支持 stdio 和 Streamable HTTP 的基本工具调用。连接与调用分别要求批准。
本地进程受与命令工具相同的 macOS 文件隔离；默认不能联网。需要联网时，配置 `network: true` 并明确批准。
本地 MCP 脚本必须位于允许读取的项目或系统工具目录。可用 `env_names` 明确列出要传入的环境变量名，不默认继承全部环境变量。
远程服务只会收到批准后的工具参数；其服务器内部权限由该服务管理。当前不实现 MCP OAuth、市场、资源订阅或并行子 Agent。

## 权限、凭据与记录

- 默认状态目录 `~/.longcode`，可通过 `LONGCODE_HOME` 指定独立目录。目录权限 700，凭据和状态文件 600。
- LongCode 不读取 Codex、Claude 或 Pi 的凭据文件，也不会自动采用环境里的模型 API key。
- 文件工具限制在项目内，拒绝越界、符号链接、常见凭据文件和内部记录。
- 命令工具使用 macOS `sandbox-exec`，默认不联网。模型调用由独立连接进程完成，模型凭据不传给命令子进程。
- 非 macOS 或隔离不可用时拒绝执行命令，不降级为无限权限。
- 取消会停止模型连接和子进程组，服务退出会等待清理。外部工具已经完成的远程副作用不能靠取消撤销。

网页按时间显示模型请求与返回、工具调用、用量、文件工具差异和长程任务检查记录。
不声称能看到模型未公开的完整思考过程。CLI 模式只能保存 CLI 实际返回的信息。
目前“文件修改”页展示 LongCode 文件工具记录的差异；命令或 CLI 内部修改需结合输出和项目差异核查。

```bash
./longcode export-session 会话编号 --format md > session.md
./longcode export-session 会话编号 --format jsonl > session.jsonl
./longcode logout --provider openai-codex
```

凭据不写入对话或导出文件；记录包含项目代码和工具输出，导出前仍应检查项目中自行硬编码的其他敏感内容。
退出模型登录不会删除会话，也不影响外部 CLI 的登录。

## 验证与边界

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
# 在允许本机监听、允许启动 macOS 沙箱的终端中运行完整集成测试：
LONGCODE_TEST_SANDBOX=1 PYTHONPATH=src python3 -m unittest discover -s tests -q
cd src/longcode/provider && npm test
```

已用真实 Pi 组件连接本机模拟模型接口，隐藏外部 CLI 后执行文件修改，并运行真实 Python 项目测试。
已测试错误修改即使自报完成也不能通过验收，随后可以修复并接纳；也测试路径限制、只读岗位、取消、并发写入、HTTP 来源检查、重连与中断记录。

**尚未完成真实账户验收**：ChatGPT 浏览器/设备码登录、订阅刷新后的真实模型请求，以及 OpenAI/Anthropic/第三方 API 的真实服务调用。
CLI 参数和模拟协议已经测试，未消耗用户账户额度进行 Codex/Claude 真实任务复测。
这些需要用户提供实际可用账户并在本机完成登录，不能由模拟测试代替。

此次不修改历史评测数据，也不宣称完成率或 token 成本已经改善。性能比较需要另行进行同条件评测。

## 参考与许可

- [Pi 模型与认证组件](https://github.com/earendil-works/pi/tree/main/packages/ai)
- [Codex 登录说明](https://developers.openai.com/codex/auth)
- [Pi 许可副本](../src/longcode/provider/THIRD_PARTY_NOTICES.md)

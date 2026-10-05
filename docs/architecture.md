# LongCode Harness V1.2 实现说明

## 数据与执行边界

`TaskContract` 保存不可变目标、验收、范围和确定性检查；`TaskState` 保存每项验收的状态、
控制层计数、重试、升级原因和可信证据。Executor 只在事务候选工作区内修改内容，通过验证后
才提升到正式工作区。每个已验证里程碑写入检查点，崩溃恢复会先重验已提升事务，不会凭模型
自述恢复完成状态。

## 自适应控制

`EvidenceDrivenManager` 先使用确定性依赖与风险规则。普通轮次不调用模型 Manager；只有
`SafetyBelt` 在最强执行层检测到重复故障并增加 `replan_count` 后，才消费一次新的 replan
触发并调用 Manager。已消费的计数持久化，恢复运行不会重复付费。

`VerifierMesh` 始终先运行范围、命令和配置的环境插件：

- 明确失败时直接返回 L1 证据，不调用 Auditor；
- 默认配置且证据一致时跳过 Auditor；
- `semantic`、`design`、`ux`、`security`、`irreversible` 等配置，或必要验证为
  `uncertain`、执行完成声明与确定性证据冲突时，升级到 L2；
- 需要 Auditor 但未配置时按 `uncertain` 失败关闭。

控制层以 `control_level_changed` 事件记录。`audit_completed` 只代表真实模型调用；跳过时记录
`audit_skipped`，不会制造一条“虚拟通过”的审计证据。Manager、Auditor、Executor 共用同一
Codex 后端时，所有调用的 JSONL Token 都累计到全系统用量。

## 故障与恢复

故障按 provider、agent runtime、tool、environment、execution、verification、policy、
control plane 和 goal 分类。重试证据包含失败命令、退出码、超时状态以及截断后的 stdout 或
stderr。瞬时故障只重启对应工具或轮次；重复故障按执行层升级；最高层重复故障进入 L3
重规划。范围违规和不可安全恢复的边界会阻止提交。

## 诊断—修复—复验闭环

V1.2 增加 `SemanticContractVerifier`，从公开目标和验收条件识别受支持的行为接口，并在候选
工作区外的临时目录中运行黑盒探针。当前内置的第一类编译器覆盖 JSON migration CLI：验证
真实语义转换、dry-run 无修改、确定性输出、幂等重跑、未知字段保留和冲突报告。探针不读取
隐藏验收，也不信任 Executor 自报完成。

若探针失败，Verifier 将每项证明义务的 expected/observed、复现入口、回归命令和必须保留的
约束写入结构化 repair packet。失败候选保持未信任且不提升到正式工作区；下一轮从该候选复制
新的修复工作区，但冲突检测仍锚定最后一个可信正式快照。修复通过相同语义探针和完整回归后
才允许提升。事件流以 `repair_packet_created`、`repair_attempt_started`、`repair_verified` 记录，
同时统计修复尝试、修复成功和转化率。

## Pilot 公平性

评测 runner 为三组复制同一冻结 fixture，随机化组别顺序，用 macOS `sandbox-exec` 阻止运行
过程读取隐藏验收，并在运行结束后由外部 runner 执行隐藏检查。Manifest 固定 Codex 可执行
文件及 SHA-256、模型、推理强度、认证方式、单次 900 秒上限和最多 6 轮。

认证模式为 ChatGPT 订阅登录。runner 会从子进程环境移除 OpenAI、Anthropic 及其他 API Key
变量，三组均通过本机 Codex CLI。Token 是成功完成的 Codex JSONL turn 的统一实际观测；
此模式没有供应商 API 代理，因此只执行墙钟和轮次上限，不声明 API 级 Token 硬封顶。

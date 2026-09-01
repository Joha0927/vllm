# Extend-prefill 暂存方案

## 状态

本方案暂不进入当前 profiling 工作流。当前主流程恢复为已经在 H20 上验证通过的
Prefill+decode；本文只保存 Extend-prefill 的设计，后续需要时再重新实现并进行 H20
qualification。

本地单测曾覆盖配置推导、streaming output evidence 和 profiler barrier，但该 workload
尚未在 H20 上完成 production qualification，因此不能标记为已验证结果。

## 目标 shape

```text
global batch          = 8
history/request       = 14336
target/request        = 2048
final context/request = 16384
DCP                    = 1
EP                     = 8

策略 A                = TP1 / DP8 / EP8
策略 B                = TP2 / DP4 / EP8
```

TP1/DP8 时每个 DP rank 处理 1 个 session；TP2/DP4 时每个 DP rank 处理 2 个
session。

## Production 执行设计

使用 `AsyncLLM` 的 production streaming-input session，不构造独立 block wrapper，也不
手工初始化 KV cache：

```text
向同一个 session 提交 14336-token history
  -> 等待本 rank 的 history 全部完成
  -> 8-rank barrier
  -> 启动所有 worker 的 Torch Profiler
  -> 8-rank profiler-started barrier
  -> 向同一个 session 追加 2048-token target
  -> 等待所有 target 完成
  -> 8-rank target-complete barrier
  -> 停止 Torch Profiler
```

trace 只采集 target 阶段，history 用于建立真实 KDA recurrent state 和 MLA KV cache。

每个 streaming chunk 使用：

```text
max_tokens = 1
ignore_eos = true
output_kind = DELTA
```

history 结束时采样但尚未执行的末尾 token，应由 vLLM production streaming update 在
追加 target 时丢弃。

## 已识别的实现约束

### Output evidence 必须即时快照

`RequestOutput.prompt_token_ids` 引用 session 内部的可变列表，第二次 streaming update
会原地扩展它。不能保存第一次 `RequestOutput` 后在末尾重新读取；必须在每个 output
到达时立即保存：

```text
history prompt-token count = 14336
target prompt-token count  = 16384
output-token count/chunk    = 1
```

### 必须同步 profiler 的开始和结束边界

所有 rank 完成 history 后才能启动 profiler；所有 rank 完成 target 后才能停止
profiler。否则不同 rank 的 trace 可能包含不同 capture 窗口。

跨 rank barrier 需要：

- 使用 `TORCHELASTIC_RUN_ID` 和 restart count 隔离不同运行；
- rank 失败时写 failure marker，让其他 rank fail fast；
- 仅适用于当前单机 8×H20 流程。

### 不强制 target 只有一次 forward

Kimi-K3 的 KDA/Mamba production cache 使用 768-token alignment，而：

```text
14336 % 768 = 512
```

因此 scheduler 可能将 2048-token target 拆成多次真实 forward，例如先执行到下一个
cache boundary。不能为了得到单次 Q=2048 而绕过 production scheduler。

正式验收应记录每次 decoder-layer execution 的 packed-token shape，并检查：

```text
TP1/DP8：每份 worker trace 的 target token 总和 = 2048
TP2/DP4：每份 worker trace 的 target token 总和 = 4096
```

## 重新启用时需要完成的工作

1. 增加 TP1/DP8 和 TP2/DP4 两份 Extend-prefill YAML。
2. 在 production runner 中增加 AsyncLLM streaming session 路径。
3. 实现 history-ready、profiler-started、target-complete 三个跨 rank barrier。
4. 对 output evidence 做不可变 token-count 快照。
5. 在 Torch Profiler layer scope 中记录可解析的输入 shape。
6. 增加 trace validator，校验 rank 0..7、layers 0..11 和 target token 总量。
7. 先在 H20 上运行 `profile=none` qualification。
8. qualification PASS 后再运行 `warmup=3, profile=1` 正式采集。

## Qualification 验收

```text
exit code                         = 0
stage=complete, status=PASS       = 8 ranks
history prompt snapshot/request   = 14336
target prompt snapshot/request    = 16384
TP/DP/EP/DCP                      = 与配置一致
request-to-DP mapping             = 完整且无重复
OOM/CUDA/NCCL/KV-cache failure    = none
运行后 GPU memory                 = 0 MiB × 8
```

只有 qualification 和正式 trace 验收都通过后，才能把 Extend-prefill 加回主 README。

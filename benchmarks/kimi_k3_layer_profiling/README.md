# Kimi-K3 首个 Block Production Profiling

## 1. 目标

在单机 8×NVIDIA H20 上，通过真实 vLLM production execution stack 测量 Kimi-K3
首个 AttnRes block，即 decoder `layers.0..11`。

执行路径为：

```text
LLM / AsyncLLM
  -> EngineCore
  -> production executor
  -> 8 production workers
  -> V2 GPU model runner
  -> KimiK3ForConditionalGeneration
  -> KimiLinearModel.layers.0..11
```

模型只通过 Hugging Face config override 将 `text_config.num_hidden_layers` 从 93
截断为 12。测试不使用自定义模型类、block wrapper、手工 attention metadata 或手工
KV-cache 初始化。

### 与直接修改层数并启动 `vllm serve` 的区别

两种方式可以使用相同的 production `EngineCore`、模型、KV cache、attention/MoE
backend 和通信实现；当前 benchmark 不是简化版 block forward。区别主要在实验控制：

| 当前 benchmark | 直接 `vllm serve` |
| --- | --- |
| 直接输入固定随机种子的 token IDs | 通常输入文本并经过 tokenizer |
| 固定 global/local batch 和 request-to-DP 映射 | continuous batching 和负载均衡可能改变请求组合 |
| 根据 workload 固定 `max_num_seqs`、token budget 和 `max_model_len` | 默认 scheduler 参数可能拆分或重组 prefill |
| 固定 warmup，并用 barrier 对齐所有 worker 的 profiler 起点 | 需要客户端自行协调 warmup、请求完成和 profile API |
| 验证每个请求实际生成 2 个 token，证明 decode 已执行 | 仅设置 `max_tokens=2` 不能代替结果验收 |
| 输出拓扑、请求映射、shape、backend 和 rank evidence | 需要额外客户端和日志处理才能获得同等证据 |

因此，`vllm serve` 更适合端到端在线服务延迟、排队、tokenizer 和 continuous batching
测试；当前 benchmark 更适合固定 shape、固定并行策略、可重复比较的 12-layer block
profiling。如果为 `vllm serve` 补齐固定 token IDs、请求屏障、scheduler 参数、warmup、
profile start/stop 和结果验收，两者的 worker-side trace 可以接近，但这相当于重新实现一套
profiling client。

## 2. 固定条件

以下内容属于模型或正式实验基线：

```text
GPU                     = 8 × NVIDIA H20
layers                  = 0..11
hidden_size             = 7168
num_experts             = 896
weight format           = dummy MXFP4
language_model_only     = true
prefix caching          = false
execution mode          = eager
profiler                = Torch Profiler
MoE routing simulation  = uniform_random
```

`hidden_size`、expert 数、attention 类型、量化格式和层内结构均从
`model_config/config.json` 读取，不允许通过 workload 覆盖。执行模式、profiler 和 routing
strategy 虽然在正式矩阵中固定，仍必须显式写入配置和 metadata。固定随机种子生成 token
IDs，相同配置可复现输入。

## 3. 正式测试矩阵

使用全局 batch 8、EP8、DCP1，对比两种并行策略：

```text
策略 A: TP1 / DP8 / EP8 / DCP1
策略 B: TP2 / DP4 / EP8 / DCP1
```

启用 expert parallel 后，当前配置没有 PCP，因此：

```text
EP = TP * DP
```

正式采集共两组：

| Workload | TP | DP | EP | 全局 BS | Prefill/request | Decode/request |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prefill + decode | 1 | 8 | 8 | 8 | 16384 | 1 |
| Prefill + decode | 2 | 4 | 8 | 8 | 16384 | 1 |

TP1/DP8 时每个 DP rank 处理 1 个请求；TP2/DP4 时每个 DP rank 处理 2 个请求。
runner 必须显式分配并记录 request-to-DP 映射，不能只根据全局 batch 推测本地 batch。

## 4. Workload 定义

### 4.1 Prefill + decode

每个请求输入 16384 tokens，并生成 2 个 output tokens：

```text
prompt/request        = 16384
prefill tokens/global = 8 * 16384 = 131072
generated tokens/global = 8 * 2 = 16
decode input tokens/global = 8 * 1 = 8
max_tokens            = 2
```

Profiler 在请求开始前启动，同一份 trace 记录：

```text
完整 16K prefill
  -> 一次单-token decode
```

`max_tokens` 表示每个请求最多生成的 output token 数，不是 forward 次数。本用例的
生产语义为：

```text
model execution #1: 输入 16384 个 prompt tokens
                    -> full-prefill forward
                    -> 根据末位 logits 采样 output token #1

model execution #2: 输入上一步的 output token #1
                    -> single-token decode forward
                    -> 采样 output token #2，达到 max_tokens=2 后停止
```

因此，`max_tokens=1` 只会有 prefill forward，不能证明 decode 已执行；
`max_tokens=2` 才会在 prefill 之后再触发一次真实 decode forward。运行时还必须设置
`ignore_eos=true`，不配置其他 stop token，并使 `max_model_len >= 16384 + 2`，否则请求
可能在 decode 之前提前停止。

每份 worker trace 中，每个 `layers.0..11` 应出现两次：第一次是 full
prefill，第二次是 decode。这一判定还依赖 full prefill 没有被 scheduler 切成多个
chunk；所以 qualification 必须核对每个 DP engine 的 token budget 足以一次调度其本地
prompt tokens。

不再单独采集 Full prefill trace。Full-prefill 性能直接取自 Prefill + decode trace 的
第一次 layer execution。独立 Full prefill 只用于不开 profiler 的 shape qualification。

## 5. 输入配置

输入分为 workload、并行、backend 和采集四类。所有会改变实际 kernel、通信或模型执行
路径的选项都必须进入 YAML/CLI、manifest 和 `run_meta.txt`，不能依赖未记录的环境变量。

### 5.1 Workload

```text
workload            full_prefill 或 prefill_decode
batch_size          全局请求数
history_len         目标阶段前每请求已经处理的 token 数
query_len           目标阶段每请求执行的 token 数
random_seed         token IDs 随机种子
```

正式配置为：

```text
Prefill+decode: workload=prefill_decode, history_len=16384, query_len=1
```

`prefill_decode` 的 `max_tokens` 由 workload 固定推导为 2，不作为用户可任意调整的
YAML 参数。实际设置位于 `production_profile.py` 的 `SamplingParams`，代码旁注明了
“第二个 output token 强制触发一次 decode execution”的原因。

`full_prefill` 只用于 qualification：

```text
Full prefill: workload=full_prefill, history_len=0, query_len=16384
```

### 5.2 并行和通信

```text
tensor_parallel_size
data_parallel_size
decode_context_parallel_size
enable_expert_parallel
all2all_backend
expert_placement_strategy
enable_dbo
```

必须满足：

```text
TP * DP = gpu_count
batch_size % DP = 0
DCP 能整除 TP
EP enabled 时，EP = TP * DP
```

`all2all_backend` 决定 MoE token dispatch/combine 的通信实现，必须作为输入配置。当前已
验证的基线是 `allgather_reducescatter`。其他 backend 必须分别 qualification，不能因为
名称存在就认为 H20 环境可用。

`expert_placement_strategy` 决定 global experts 到 EP rank 的映射，基线为 `linear`。
`enable_dbo` 控制 dual-batch overlap，基线为 `false`；开启后还必须记录 ubatch 和
prefill/decode threshold，不能只记录一个布尔值。

### 5.3 Kernel/backend

以下会改变 profiling 结果的选项已经加入 YAML、CLI、manifest 和 EngineArgs 传递链：

| 配置 | 作用 | 基线 |
| --- | --- | --- |
| `moe_backend` | 选择 MoE expert kernel；控制是否使用 MegaMoE | `auto` |
| `linear_backend` | 选择非 MoE quantized linear kernel | `auto` |
| `attention_backend` | 选择通用 attention/MLA decode backend | `auto` |
| `kda_prefill_backend` | KDA prefill 使用 Triton 或 FlashKDA | `auto` |
| `mla_prefill_backend` | MLA prefill backend | `auto` |
| `kv_cache_dtype` | 选择 production KV-cache 表示 | `auto` |
| `shard_sp_shared_expert` | TP2 sequence-parallel 下是否切分 shared expert | `false` |
| `kv_cache_memory_bytes` | 每 GPU 为 production cache 预留的字节数 | 明确记录 |

`auto` 不是一个可忽略的值。它表示由当前 commit、硬件和模型配置自动选择；每次运行仍须
记录最终解析出的 backend 和实际构造的类。

#### MegaMoE

Kimi-K3 是否使用 MegaMoE 由 `moe_backend` 控制。production CLI 对应：

```text
--moe-backend deep_gemm_mega_moe
```

源码中的判断是：

```text
use_mega_moe = kernel_config.moe_backend == "deep_gemm_mega_moe"
```

因此 MegaMoE 不是根据 batch、token 数或 EP size 自动启用。选择
`deep_gemm_mega_moe` 还要求启用 EP，并满足 Kimi-K3 对 SITU、latent MoE、grouped top-k
和 expert divisibility 的约束。配置不兼容时应在初始化阶段失败，不允许静默回退。

使用 `moe_backend=auto` 时走通用 `FusedMoEFactory` 路径，再由 MXFP4 backend selector
选择可用 expert kernel。正式结果必须区分“请求的 backend”和“最终解析的 backend”。

#### KDA 和 MLA

KDA prefill 可配置：

```text
auto
triton
flashkda
```

H20 满足 FlashKDA 的 SM90 条件，但最终是否可用还取决于 BF16、head dimension 和 KDA
gate。显式请求不可用 backend 时必须失败。

MLA prefill 可配置 backend 包括 FlashAttention、FlashInfer 和 TRTLLM ragged 等实现。
`attention_backend` 还会影响 MLA decode 使用的通用 attention implementation。配置层使用
`auto` 表示不强制选择，由 vLLM production backend registry 决定。

`kv_cache_dtype` 会改变 cache 占用、读写带宽和可用 attention kernel，必须与 backend
一起记录。正式矩阵可以先固定为 `auto`，但不能在不同运行中遗漏该字段。

#### Kimi-K3 shared expert

`VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT` 会改变 TP2/DP4 下 shared-expert 的权重布局和通信：
关闭时 shared expert 在 rank 间复制；打开时增加 all-gather/reduce-scatter，减少权重带宽和
驻留显存。它可能更适合低 token 场景，因此必须包装成显式输入
`shard_sp_shared_expert` 并写入 metadata。

以下 Kimi-K3 开关不进入本次 H20 输入：

- `VLLM_KIMI_K3_GEMM_AR` 和 `VLLM_KIMI_K3_GEMM_RS` 只支持 SM100 family，H20 为
  SM90；
- `VLLM_KIMI_K3_AUX_ATTN_RES_STREAM` 服务于 DFlash speculative drafter，本计划未启用
  speculative decoding。

### 5.4 Profiler

```text
warmup_iters        = 3
profile_iters       = 1
profile             = torch
profile_output_dir  = 每次运行唯一目录
```

结果目录不按 Git commit 分层。每次使用实验名和时间戳创建目录，commit 写入
`run_meta.txt`。

## 6. Production Engine 约束

所有 workload 均使用：

```text
hf_overrides = {"text_config": {"num_hidden_layers": 12}}
language_model_only = true
load_format = dummy
enable_prefix_caching = false
enforce_eager = true
max_num_seqs = local workload 所需容量
max_num_batched_tokens = 能容纳目标调度的容量
max_model_len = 能容纳 history、query 和 output tokens
```

TP1/DP8 和 TP2/DP4 使用 production external-launcher，必须通过单机 8-rank
`torchrun` 启动。每个 DP rank 只处理自己的 local batch，同一 TP group 内的 rank
处理相同请求。`max_num_seqs` 和 `max_num_batched_tokens` 按每个 DP engine 的本地
workload 推导。

Prefill + decode 必须使用 `max_tokens=2`。这些调度参数由 workload 推导，不作为用户任意
覆盖项。

Prefill + decode 还必须满足：

```text
ignore_eos = true
stop / stop_token_ids = unset
max_model_len >= prompt_len + max_tokens
long_prefill_token_threshold = 0（或足以容纳本地完整 prompt）
max_num_batched_tokens >= 每个 DP engine 在该 step 的本地 prompt token 数
```

上述数值必须在 evidence 中记录实际解析结果。不能只根据
`max_tokens=2` 就推断 trace 一定是“一次 prefill + 一次 decode”。

## 7. 实现状态和下一步

当前 production EngineCore、12-layer model loading、Torch Profiler worker traces、
`layers.0..11` scope、requested backend 输入和 Prefill+decode workload 已经实现并通过
TP1/DP8/EP8 qualification。TP2/DP4 使用同一条执行路径和独立配置。

实现顺序：

1. 对目标并行配置先执行 qualification；
2. 使用完全相同的配置执行正式 Torch Profiler 采集；
3. 汇总 layer、kernel、All-to-All/NCCL、rank-max latency 和峰值显存。

## 8. 验收标准

### 8.1 通用标准

- production EngineCore 路径；
- 8 个 worker 正常启动和退出；
- TP、DP、EP、DCP 与 request-to-DP 映射符合配置；
- requested/resolved backend 和实际模型类均有 evidence；
- `stage=complete, status=PASS`，进程退出码为 0；
- 无 OOM、CUDA、NCCL、worker 或 KV-cache failure；
- 8 份非空 `.pt.trace.json.gz`，覆盖所有 worker rank；
- trace 包含 CPU operators、CUDA kernels 和 `layers.0..11`；
- trace 不包含 `layers.12` 或更高 decoder layer；
- 运行后 GPU 显存释放。

### 8.2 Prefill + decode

- prompt 每请求恰好 16384 tokens；
- 每个请求生成 2 个 output tokens；
- 每个 complete record 的 `profiled_output_token_counts` 全部为 2；
- `decode_executions_per_request=1`；
- 每个 layer scope 在每份 trace 中恰好出现两次；
- 第一次是 full prefill，第二次是单-token decode；
- 两次执行可通过顺序、token 数和 recorded shapes 区分。

## 9. 结果解释

不能将 8 个 rank 的累计 CUDA 时间相加作为 block wall-clock latency。分布式 block
完成时间由最慢 rank 决定，应按 iteration 计算 rank-max，再汇总多次运行的中位数或
分位数。

Prefill + decode trace 必须分别统计第一次和第二次 layer execution。整份 trace 的总计
会被 16K prefill 主导，不能直接代表 decode latency。

结果使用 dummy MXFP4 weights 和 uniform-random routing，只代表指定 shape、并行策略、
backend、production execution stack 和 H20 环境下的性能，不代表真实 checkpoint 的生成
质量或自然 expert 分布。

## 10. 输出

每次运行至少保存：

```text
run_meta.txt
manifest.json
torch_profile.log
traces/*.pt.trace.json.gz
```

metadata 至少包含：

```text
Git commit
workload / BS / history / query
TP / DP / EP / DCP / local batch
request-to-DP mapping
requested and resolved backends
Kimi-K3 performance flags
warmup/profile iterations
random seed
exit code
```

## 11. 目录

```text
benchmarks/kimi_k3_layer_profiling/
├── README.md
├── EXTEND_PREFILL_PLAN.md
├── __init__.py
├── benchmark.py
├── config.py
├── production_profile.py
├── model_config/
│   └── config.json
└── shapes/
    ├── smoke.yaml
    ├── prefill_decode_bs8_p16384.yaml
    └── prefill_decode_bs8_p16384_tp2_dp4.yaml
```

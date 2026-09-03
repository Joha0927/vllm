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
profilers               = Torch Profiler、Nsight Compute
MoE routing simulation  = uniform_random
```

`hidden_size`、expert 数、attention 类型、量化格式和层内结构均从
`model_config/config.json` 读取，不允许通过 workload 覆盖。执行模式、profiler 和 routing
strategy 虽然在正式矩阵中固定，仍必须显式写入配置和 metadata。固定随机种子生成 token
IDs，相同配置可复现输入。

## 3. 正式测试矩阵

使用全局 batch 8、EP8、DCP1，对比三种执行组合：

```text
TP1 / DP8 / EP8 / DCP1 / AG+RS
TP2 / DP4 / EP8 / DCP1 / AG+RS
TP2 / DP4 / EP8 / DCP1 / FlashInfer NVLink one-sided A2A
```

启用 expert parallel 后，当前配置没有 PCP，因此：

```text
EP = TP * DP
```

三种执行组合均采集 with-stack trace：

| 并行与通信策略 | TP | DP | EP | with-stack | 用途 |
| --- | ---: | ---: | ---: | --- | --- |
| TP1/DP8/EP8 + AG+RS | 1 | 8 | 8 | true | TP1 基线与源码归因 |
| TP2/DP4/EP8 + AG+RS | 2 | 4 | 8 | true | TP2 基线与源码归因 |
| TP2/DP4/EP8 + FlashInfer one-sided A2A | 2 | 4 | 8 | true | TP2 A2A 对照与源码归因 |

三组 workload 均为每请求 16384-token prefill 加一次 single-token decode，全局 BS 为 8。
所有组使用相同的 profiler 设置，因此只在这三组 trace 之间进行相对性能比较；with-stack
开销意味着 trace latency 不能直接视为无 profiler 时的线上绝对 latency。

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
验证的基线是 `allgather_reducescatter`。新增对照为
`flashinfer_nvlink_one_sided`，它使用 FlashInfer NVLink one-sided All-to-All，而不是
DeepEP。Kimi-K3 在 H20 上使用的 MXFP4 Marlin experts 明确支持 one-sided，但不支持
`flashinfer_nvlink_two_sided`。正式采集前必须独立 qualification，确认 FlashInfer comm
模块实际可用，且 runtime manager 已解析为目标实现。`naive` 和已删除的 `pplx` 不可作为
真正 All-to-All 对照，因为当前 vLLM 会将它们回退为
`allgather_reducescatter`。

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
profiler_with_stack = true（当前正式 Torch trace）
enable_layerwise_nvtx_tracing = 是否独立注册模块级 NVTX 范围
```

结果目录不按 Git commit 分层。每次使用实验名和时间戳创建目录，commit 写入
`run_meta.txt`。

`profiler_with_stack=true` 会记录 Python 调用栈，但会增加 CPU profiler 开销和 trace
体积。当前三组正式 Torch trace 全部使用相同设置，只做组间相对比较。启用后应验证目标
CPU op 含非空 Python stack frame，并能通过 correlation/flow 关联到 CUDA kernel；不能
只检查是否存在名为 `python_function` 的事件。

### 5.5 Nsight Compute

NCU 不与 Torch Profiler 同时开启。NCU 配置必须使用：

```text
profile                        = none
profiler_with_stack            = false
enable_layerwise_nvtx_tracing  = true
```

独立 NVTX 开关让 production worker 为所有 PyTorch 子模块注册 NVTX，同时不创建 Torch
Profiler。NCU 仅采 GPU 0，即单机 rank 0，并用模块范围过滤目标 kernel。

固定 TP2/DP4/EP8 时，KDA、MLA 和 Dense 不受 MoE `all2all_backend` 直接影响，各采
prefill/decode 一次；MoE 分别采 AG+RS 和 FlashInfer one-sided A2A。正式 NCU 矩阵为：

| 并行与通信策略 | 阶段 | 模块 | 代表范围 |
| --- | --- | --- | --- |
| TP2/DP4/EP8 | prefill | KDA | `layers.1.self_attn` |
| TP2/DP4/EP8 | decode | KDA | `layers.1.self_attn` |
| TP2/DP4/EP8 | prefill | MLA | `layers.3.self_attn` |
| TP2/DP4/EP8 | decode | MLA | `layers.3.self_attn` |
| TP2/DP4/EP8 | prefill | Dense | `layers.0.mlp` |
| TP2/DP4/EP8 | decode | Dense | `layers.0.mlp` |
| TP2/DP4/EP8 + AG+RS | prefill | MoE | `layers.1.block_sparse_moe` |
| TP2/DP4/EP8 + AG+RS | decode | MoE | `layers.1.block_sparse_moe` |
| TP2/DP4/EP8 + FlashInfer one-sided A2A | prefill | MoE | `layers.1.block_sparse_moe` |
| TP2/DP4/EP8 + FlashInfer one-sided A2A | decode | MoE | `layers.1.block_sparse_moe` |

同一 `prefill_decode` 请求会进入目标模块两次。正式详细采集前必须先做一次轻量 NCU
NVTX inventory，记录 production trace 中准确的模块范围文本和两次调用的 input shape；
后续分别按 shape 过滤 prefill 与 decode。不能把同名 kernel 的两次调用聚合成一个结果。
分布式通信 kernel 使用 application replay，避免单独 kernel replay 破坏 collective 同步。

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
`layers.0..11` scope、requested backend 输入和 Prefill+decode workload 已经实现。
正式矩阵包含 AG+RS 的 TP1/DP8 与 TP2/DP4，以及 FlashInfer one-sided A2A 的
TP2/DP4；每组在正式采集前单独 qualification。

实现顺序：

1. 对目标并行和 All-to-All 配置先执行 qualification；
2. 为每种组合采集 `profiler_with_stack=true` 的 trace；
3. 仅在使用相同 profiler 设置的三组之间做相对比较；
4. 汇总 layer、kernel、All-to-All/NCCL、rank-max latency 和峰值显存。

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
├── __init__.py
├── benchmark.py
├── config.py
├── production_profile.py
├── model_config/
│   └── config.json
└── shapes/
    ├── smoke.yaml
    ├── ncu_tp2_dp4_ep8_ag_rs.yaml
    ├── ncu_tp2_dp4_ep8_flashinfer_one_sided.yaml
    ├── prefill_decode_bs8_p16384_with_stack.yaml
    ├── prefill_decode_bs8_p16384_tp2_dp4_with_stack.yaml
    └── prefill_decode_bs8_p16384_tp2_dp4_flashinfer_one_sided_with_stack.yaml
```

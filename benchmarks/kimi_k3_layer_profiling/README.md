# Kimi-K3 首个 Block Production Profiling

## 1. 目标

在单机 8×NVIDIA H20 上，通过真实 vLLM production execution stack 测量 Kimi-K3
首个 AttnRes block，即 decoder `layers.0..11`。

正式执行路径为：

```text
LLM
  -> EngineCore
  -> production executor
  -> 8 production workers
  -> V2 GPU model runner
  -> KimiK3ForConditionalGeneration
  -> KimiLinearModel.layers.0..11
```

模型只通过 Hugging Face config override 将 `text_config.num_hidden_layers` 从 93
截断为 12。没有自定义模型类、block wrapper、prompt-embedding shortcut、手工 attention
metadata 或手工 KV-cache 初始化。

当前只支持：

- prefill；
- 逻辑层 0 至 11；
- eager execution；
- dummy MXFP4 weights；
- TP8、DP1、EP8、DCP1；
- `uniform_random` MoE routing simulation；
- PyTorch Profiler。

## 2. 输入定义

用户配置 `batch_size` 和每请求 `query_len`：

```text
total_tokens = batch_size * query_len
block input shape = [total_tokens, 7168]
```

`hidden_size=7168` 从本地 Kimi-K3 config 读取，不可覆盖。输入 token IDs 使用固定
`random_seed` 生成，因此相同配置可以复现。token IDs 经过 production embedding 后才
进入 layer 0。

正式 profiling 点为：

```text
batch_size = 8
query_len = context_len = 4096
total_tokens = 32768
layers = 0..11
```

对应配置：

```text
benchmarks/kimi_k3_layer_profiling/shapes/prefill_bs8_q4096.yaml
```

## 3. 目录

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
    └── prefill_bs8_q4096.yaml
```

- `config.py`：读取 YAML 和模型 config，推导 token shape、层类型及并行配置；
- `benchmark.py`：提供 dry-run、层类型预览和 production profile 入口；
- `production_profile.py`：构造 production `LLM/EngineCore`，执行 warmup 和 PyTorch
  Profiler capture；
- `model_config/config.json`：离线模型结构快照，不包含权重；
- `shapes/`：smoke 和正式 workload。

## 4. 配置和 dry-run

本地检查正式 shape：

```bash
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/prefill_bs8_q4096.yaml \
  --dry-run
```

关键结果必须为：

```text
num_scheduled_tokens = 32768
packed_shape = [32768, 7168]
logical_layer_range = [0, 11]
expert_parallel_size = 8
```

正式配置只接受 12 层。hidden size、专家数、起始层和其他模型结构参数不能从 YAML
覆盖。

## 5. Production Engine 参数

`production_profile.py` 将 workload 映射为：

```text
hf_overrides = {"text_config": {"num_hidden_layers": 12}}
language_model_only = true
load_format = dummy
enable_expert_parallel = true
enable_prefix_caching = false
enforce_eager = true
kv_cache_memory_bytes = 4 GiB/GPU
max_num_seqs = batch_size
max_num_batched_tokens = max(total_tokens, context_len + 1)
max_model_len = context_len + 1
```

正式 shape 对应：

```text
max_num_seqs = 8
max_num_batched_tokens = 32768
max_model_len = 4097
```

`max_tokens=1` 使请求只完成一次 prefill model execution，不再执行额外 decode
forward。

## 6. Shape qualification

正式 profiling 前先运行一次不带 profiler 的 qualification：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/prefill_bs8_q4096.yaml \
  --production-profile \
  --profile none
```

通过标准：

- `stage=complete, status=PASS`；
- TP8、EP8、DCP1；
- 8 个 worker 正常退出；
- 无 OOM、CUDA、NCCL、worker 或 KV-cache failure；
- 退出后 GPU 显存释放。

已在 8×H20 上完成 BS8×4096 qualification。production hybrid-cache alignment、
`LBNHC` layout resolution 和固定 4 GiB/GPU KV cache 均成功。

## 7. PyTorch Profiler 工作流

结果目录不按 Git commit 分层。每次运行使用实验名和时间戳，commit 写入
`run_meta.txt`。

```bash
cd /home/l00948931/vllm

RUN_COMMIT=$(git rev-parse HEAD)
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="profile_outputs/prefill_bs8_q4096_torch/${RUN_ID}"
TRACE_DIR="${RUN_DIR}/traces"

mkdir -p "${TRACE_DIR}"

printf \
  'git_commit=%s\nrun_id=%s\nbatch_size=8\nquery_len=4096\ntotal_tokens=32768\nwarmup_iters=3\nprofile_iters=1\nprofiler=torch\n' \
  "${RUN_COMMIT}" \
  "${RUN_ID}" \
  | tee "${RUN_DIR}/run_meta.txt"

set -o pipefail

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/prefill_bs8_q4096.yaml \
  --production-profile \
  --warmup-iters 3 \
  --profile-iters 1 \
  --profile torch \
  --profile-output-dir "${TRACE_DIR}" \
  2>&1 | tee "${RUN_DIR}/torch_profile.log"

PROFILE_RC=${PIPESTATUS[0]}
printf 'profile_exit_code=%s\n' "${PROFILE_RC}" \
  | tee -a "${RUN_DIR}/run_meta.txt"
```

执行顺序为：

```text
Engine/KV-cache/JIT initialization
  -> warmup forward 1
  -> warmup forward 2
  -> warmup forward 3
  -> llm.start_profile()
  -> one profiled prefill forward
  -> llm.stop_profile()
```

每个 production worker 写出独立的 `.pt.trace.json.gz` 和 profiler summary。Torch
Profiler 记录 CPU/CUDA activity 和 operator shapes；为控制文件大小，默认不记录 Python
stack 和 memory timeline。

V2 runner 在 kernel warmup 后，为 decoder `layers.N` 注册 `record_function` scope。
因此 trace 中应当能按 `layers.0` 至 `layers.11` 过滤真实 production layer，并在每个
layer scope 内查看 PyTorch operators 与 CUDA kernels。

## 8. Torch trace 验收

正式采集必须满足：

- `profile_exit_code=0`；
- `stage=complete, status=PASS`；
- 8 个 worker 正常退出；
- trace 目录至少包含 8 份非空 `.pt.trace.json.gz`；
- 每个 TP rank 都有独立 trace；
- trace 中包含 `layers.0` 至 `layers.11`；
- 不包含 `layers.12` 或更高 decoder layer；
- trace 中存在 CPU operator 和 CUDA kernel events；
- 无 OOM、CUDA、NCCL、worker 或 KV-cache failure。

tokenizer/chat-template warmup 在 `skip_tokenizer_init=True` 下可能打印被捕获的 warning；
只要 EngineCore forward、worker exit 和最终退出码正常，该 warning 不判为失败。

## 9. 结果解释

PyTorch Profiler 适合分析：

- layer 0 至 11 的 CPU/CUDA 时间；
- KDA、MLA、MoE 和 AttnRes 对应的 operators/kernels；
- operator shape；
- 各 rank 的执行差异；
- NCCL 调用与计算的相对位置。

不能把 8 个 rank 的累计 CUDA 时间相加后当作 block wall-clock latency。分布式 block
完成时间由最慢 rank 决定；最终汇总应按每次 iteration 的 rank-max，再对多次重复计算
中位数或分位数。

当前结果使用 dummy MXFP4 weights 和 uniform-random routing，只代表真实 shape、
production execution stack、H20 backend 与指定 routing simulation 下的性能，不代表
真实 checkpoint 的生成质量或自然 expert 分布。

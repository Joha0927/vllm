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

## 10. 后续复用流程

这套工具的日常使用原则是：模型结构和执行路径不变时，只调整 workload 参数，不修改
Python 代码。

### 10.1 选择测试规模

测试规模由以下参数决定：

```text
batch_size          并发请求数
query_len           每个请求的 prefill token 数
context_len         当前仅支持 prefill，必须等于 query_len
total_tokens        batch_size * query_len
warmup_iters        正式采集前执行但不记录的次数
profile_iters       正式记录的次数
```

模型的 `hidden_size`、专家数、attention 类型等结构参数从
`model_config/config.json` 读取，不应随测试 workload 修改。正式 block profiling 固定加载
并执行 `layers.0..11`。

临时测试一个新 shape 时，可以直接使用命令行覆盖现有配置，无需新增 YAML。例如测试
`BS=4, Q=2048`：

```bash
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml \
  --batch-size 4 \
  --query-len 2048 \
  --context-len 2048 \
  --dry-run
```

需要长期保留或重复比较的 shape，应复制一份 YAML 并使用表达 workload 的文件名，例如：

```text
shapes/prefill_bs4_q2048.yaml
```

### 10.2 第一步：dry-run

每个新 shape 首先执行 dry-run。它不需要 GPU，也不会加载权重：

```bash
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml \
  --batch-size 4 \
  --query-len 2048 \
  --context-len 2048 \
  --dry-run
```

至少确认：

```text
num_scheduled_tokens = batch_size * query_len
packed_shape = [num_scheduled_tokens, 7168]
logical_layer_range = [0, 11]
TP = 8, EP = 8, DCP = 1
```

### 10.3 第二步：production qualification

在 H20 上先关闭 profiler，验证新 shape 能在真实 production 路径稳定运行：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml \
  --batch-size 4 \
  --query-len 2048 \
  --context-len 2048 \
  --warmup-iters 1 \
  --profile-iters 1 \
  --production-profile \
  --profile none
```

通过标准：

- 最终输出 `stage=complete, status=PASS`；
- 退出码为 0；
- 调度参数能够容纳指定 batch 和总 token 数；
- 无 OOM、CUDA、NCCL、worker 或 KV-cache failure；
- 8 个 worker 正常退出，运行后 GPU 显存释放。

qualification 失败时不要立即运行 profiler。先减小 `batch_size` 或 `query_len`，或者根据
日志处理显存和调度问题。

### 10.4 第三步：Torch Profiler 正式采集

qualification 通过后，保持同一个 shape，增加 warmup 并开启 Torch Profiler。下面的
`BS=4, Q=2048` 只是复用模板中的示例：

```bash
cd /home/l00948931/vllm

RUN_COMMIT=$(git rev-parse HEAD)
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR="profile_outputs/prefill_bs4_q2048_torch/${RUN_ID}"
TRACE_DIR="${RUN_DIR}/traces"

mkdir -p "${TRACE_DIR}"

printf \
  'git_commit=%s\nrun_id=%s\nbatch_size=4\nquery_len=2048\ntotal_tokens=8192\nwarmup_iters=3\nprofile_iters=1\nprofiler=torch\n' \
  "${RUN_COMMIT}" \
  "${RUN_ID}" \
  | tee "${RUN_DIR}/run_meta.txt"

set -o pipefail

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml \
  --batch-size 4 \
  --query-len 2048 \
  --context-len 2048 \
  --warmup-iters 3 \
  --profile-iters 1 \
  --production-profile \
  --profile torch \
  --profile-output-dir "${TRACE_DIR}" \
  2>&1 | tee "${RUN_DIR}/torch_profile.log"

PROFILE_RC=${PIPESTATUS[0]}
printf 'profile_exit_code=%s\n' "${PROFILE_RC}" \
  | tee -a "${RUN_DIR}/run_meta.txt"
```

每次测试必须使用新的 `RUN_ID`，避免覆盖旧 trace。结果目录不按 commit 分层；
`run_meta.txt` 必须记录 Git commit、shape、warmup 次数和 profile 次数。

### 10.5 第四步：验收和保存结果

每轮正式采集检查：

```bash
find "${TRACE_DIR}" -type f -name '*.pt.trace.json.gz' -size +0c | sort
find "${TRACE_DIR}" -type f -name '*.pt.trace.json.gz' -size +0c | wc -l

grep -E '"stage": "complete".*"status": "PASS"' \
  "${RUN_DIR}/torch_profile.log"

grep -E -i -n \
  'out of memory|CUDA error|NCCL error|worker.*failed|EngineCore failed|KV-cache failure' \
  "${RUN_DIR}/torch_profile.log" \
  || echo "no fatal error pattern"

cat "${RUN_DIR}/run_meta.txt"
```

正式结果必须满足第 8 节的 trace 验收标准。至少保存：

```text
run_meta.txt
torch_profile.log
traces/*.pt.trace.json.gz
```

比较不同 shape 时，应保持模型 config、并行配置、routing strategy、warmup 次数、profile
次数和软件 commit 一致，只改变计划比较的 workload 参数。

### 10.6 什么时候需要修改代码

以下情况不需要修改代码，只需改 YAML 或命令行参数：

- 改 `batch_size`；
- 改每请求 `query_len/context_len`；
- 改 warmup 或 profile 次数；
- 改输出目录或随机种子。

只有以下需求才需要扩展实现：

- 不再测试固定的 `layers.0..11`；
- 改为 decode 或混合 prefill/decode workload；
- 加载真实 checkpoint，而不是 dummy weights；
- 改模型内部结构或支持其他模型；
- 自动汇总 layer、kernel、通信时间或生成跨 rank 报表。

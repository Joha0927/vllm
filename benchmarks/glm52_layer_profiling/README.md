# GLM-5.2 前七层 Production Profiling

本目录使用真实 `LLM -> EngineCore -> production worker` 路径构造
`zai-org/GLM-5.2-FP8` 的前七层。运行时保留官方 FP8、DSA、MLA、Indexer、MoE
和 shared-expert 配置，只通过 `hf_overrides` 将 `num_hidden_layers` 从 78 截断为 7。
不使用自定义模型 wrapper，不手工构造 attention metadata 或 KV cache，也不启用 MTP
speculative decoding。

`model_config/config.json` 来自官方
[`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8/blob/main/config.json)
配置，并随 benchmark 由 Git 管理。权重使用 vLLM dummy loader，因此 runtime 参数形状、
dtype、FP8 quantization 和 TP/EP shard 与该七层配置一致，但权重数值和输出质量没有意义，
也不包含真实 checkpoint 下载与加载的临时内存。

当前配置文件 SHA-256 为
`22e49334abf8562fecf70ca3292ba3f5b33f5602fb2bf10b52dd64a66cfe65ff`。

## 覆盖范围

前七层覆盖 GLM-5.2 的三种 production layer 组合：

| 层 | Indexer | FFN |
| --- | --- | --- |
| 0–2 | full | dense |
| 3–5 | shared | MoE |
| 6 | full | MoE |

这里的 `full/shared` 描述是否由该层执行 lightning-indexer top-k；所有七层仍走 GLM-5.2
的 DSA/MLA attention。`shared` 层复用前面 full-indexer 层产生的 top-k 结果，不等于普通
dense attention。

CUDA `DeepseekV32IndexerBackend` 的 kernel block size 固定为 64，因此两套 YAML 都显式
设置 `block_size=64`。程序会在 EngineCore 初始化后读取最终值；若
`resolved_kv_manager_block_size` 不是 64，会在任何 warmup 或 profile 之前失败。

## 固定实验条件

```text
model              = GLM-5.2-FP8 production implementation
layers             = 0..6
global batch       = 32
prompt/request     = 4096 tokens
generated/request  = 2 tokens
global prefill     = 131072 tokens
KV cache           = 4 GiB/GPU
weights            = dummy FP8
MoE routing        = uniform_random
execution          = eager
prefix caching     = disabled
MTP/speculation    = disabled
```

`max_tokens=2` 使一次 `generate` 包含完整 prefill、首 token sampling 和一次 single-token
decode。TP1/DP8 时每个 DP replica 处理 4 个请求；TP2/DP4 时每个 replica 处理 8 个请求。
这里的 BS32 是 global batch，不是每卡 BS32。

实验清单：

| ID | 并行与通信 | 模式 |
| --- | --- | --- |
| G-Q1 | TP1/DP8/EP8 + AG/RS | qualification，`profile=none` |
| G-Q2 | TP2/DP4/EP8 + AG/RS | qualification，`profile=none` |
| G-P1 | TP1/DP8/EP8 + AG/RS | Torch Profiler，with-stack |
| G-P2 | TP2/DP4/EP8 + AG/RS | Torch Profiler，with-stack |

## 执行顺序

服务器必须使用与本地相同的 Git commit、8 张同一 NVLink domain 的 H20，以及能离线解析
`GlmMoeDsaConfig` 的 Transformers。建议 Transformers `>=5.9.0`；正式运行前先打印实际
版本。不要在服务器工作区直接改代码。

### 1. 离线配置预览

```bash
cd /home/l00948931/vllm

.venv/bin/python - <<'PY'
import transformers
from vllm.transformers_utils.config import get_config

path = "benchmarks/glm52_layer_profiling/model_config"
config = get_config(path, False)
print("transformers=", transformers.__version__)
print("config_class=", type(config).__name__)
print("architectures=", config.architectures)
print("model_type=", config.model_type)
print("num_hidden_layers=", config.num_hidden_layers)
print("quant_method=", config.quantization_config["quant_method"])
PY

.venv/bin/python -m benchmarks.glm52_layer_profiling.benchmark \
  --config benchmarks/glm52_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml \
  --dry-run
```

预览必须显示 `GlmMoeDsaConfig`、`GlmMoeDsaForCausalLM`、`glm_moe_dsa`、官方 78 层、
`fp8`，以及 benchmark 的 `expected_layer_range=[0,6]` 和三种 layer 组合。

### 2. Qualification（G-Q1、G-Q2）

先运行 G-Q1。通过后将 `CONFIG`、`EXPERIMENT` 改成 G-Q2，重复同一命令。

```bash
cd /home/l00948931/vllm

CONFIG=benchmarks/glm52_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml
EXPERIMENT=glm52_g_q1_tp1_dp8
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=profile_outputs/${EXPERIMENT}/${RUN_ID}
mkdir -p "${RUN_DIR}"

set -o pipefail
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.glm52_layer_profiling.benchmark \
  --config "${CONFIG}" \
  --production-profile --profile none \
  2>&1 | tee "${RUN_DIR}/qualification.log"

QUALIFICATION_RC=${PIPESTATUS[0]}
printf 'git_commit=%s\nqualification_exit_code=%s\n' \
  "$(git rev-parse HEAD)" "${QUALIFICATION_RC}" \
  | tee "${RUN_DIR}/run_meta.txt"
```

G-Q2 使用：

```text
CONFIG=benchmarks/glm52_layer_profiling/shapes/prefill_decode_bs32_p4096_tp2_dp4_ag_rs.yaml
EXPERIMENT=glm52_g_q2_tp2_dp4
```

### 3. Production Torch trace（G-P1、G-P2）

只有对应 qualification PASS 后才能采集。每组使用独立输出目录。

```bash
cd /home/l00948931/vllm

CONFIG=benchmarks/glm52_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml
EXPERIMENT=glm52_g_p1_tp1_dp8
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=profile_outputs/${EXPERIMENT}/${RUN_ID}
mkdir -p "${RUN_DIR}/traces"

set -o pipefail
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.glm52_layer_profiling.benchmark \
  --config "${CONFIG}" \
  --production-profile --profile torch \
  --profile-output-dir "${RUN_DIR}/traces" \
  2>&1 | tee "${RUN_DIR}/torch_profile.log"

PROFILE_RC=${PIPESTATUS[0]}
printf 'git_commit=%s\nprofile_exit_code=%s\n' \
  "$(git rev-parse HEAD)" "${PROFILE_RC}" \
  | tee "${RUN_DIR}/run_meta.txt"
```

G-P2 使用 TP2 YAML，并将 `EXPERIMENT` 改为 `glm52_g_p2_tp2_dp4`。

## 验收标准

G-Q1 和 G-Q2 都必须满足：

- 退出码为 0；8 条 `stage=ready` 和 8 条 `stage=complete,status=PASS`。
- `resolved_kv_manager_block_size=64`，不出现 `No common block size`。
- worker 实际模型类为 `DeepseekV32ForCausalLM`。这是 CUDA 下
  `GlmMoeDsaForCausalLM` 注册架构对应的 production 实现，不是错误加载 DeepSeek checkpoint。
- 每个 worker 的 `parameter_count`、`parameter_bytes` 大于 0，并记录 dtype breakdown、
  persistent buffer 和 `persistent_tensor_bytes`。这些字段描述模型持久 tensor，不包含 KV
  cache、临时 activation、通信 workspace 或 CUDA allocator 开销，不能替代峰值显存。
- `quantization=fp8`、`weight_source=dummy`、`routing_strategy=uniform_random`。
- `expected_layer_range=[0,6]`，且三种 layer 组合与上表一致。
- G-Q1 rank mapping 为 TP1/DP8/EP8，每个 DP rank 处理 4 个请求。
- G-Q2 rank mapping 为 TP2/DP4/EP8，每个 DP rank 处理 8 个请求；同一 TP group
  的 request indices 必须相同。
- 每个请求输出 2 个 token，`decode_executions_per_request=1`。
- 无 OOM、CUDA、NCCL、worker、FP8 backend、DSA/indexer 或 KV-cache failure；退出后显存释放。

G-P1 和 G-P2 还要求各自产生 8 份非空 trace；每份 trace 包含 CPU operators、CUDA
kernels、recorded shapes、非空 Python stack，以及 `layers.0..6` 的 prefill/decode 执行，
不包含 `layers.7+`。

with-stack trace 只用于算子、kernel、调用栈和通信归因，不能作为正式端到端时延。若后续
测 latency，应保持相同 shape 和拓扑、关闭 profiler，并按每次运行的最慢 rank 取值后再报告
多次运行的中位数。

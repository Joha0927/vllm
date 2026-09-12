# MiniMax M3 前四层 Production Profiling

本目录使用真实 `LLM -> EngineCore -> production worker` 路径，以
`MiniMaxAI/MiniMax-M3` 的仓库内默认结构和 dummy BF16 权重构造
`MiniMaxM3SparseForCausalLM`，只将 `num_hidden_layers` 截断为 4。该配置不是
`MiniMax-M3-MXFP8`；若要测 MXFP8 checkpoint backend，必须另建配置和 qualification，
不能把本结果解释为 MXFP8 性能。
不使用自定义模型 wrapper，也不手工构造 attention metadata 或 KV cache。

前四层覆盖两类结构：

```text
layers.0..2 = full attention + dense MLP
layer.3     = sparse attention + routed MoE/shared expert
```

正式 workload 是 global BS32、每请求 4096-token prefill，并生成 2 个 token，
因此同一次运行包含完整 prefill 和一次 single-token decode。正式 trace 使用
Torch Profiler、eager execution、`with_stack=true`。

## 实验清单

MiniMax M3 当前只执行一组并行策略，不与 Kimi-K3 的 TP/A2A 消融混合：

| ID | 并行与通信 | 模式 | 目的 |
| --- | --- | --- | --- |
| M-Q | TP1/DP8/EP8 + AG+RS | `profile=none` | 验证四层模型、shape、拓扑和 prefill+decode |
| M-P | TP1/DP8/EP8 + AG+RS | Torch、with-stack | 采集前四层 production trace |

固定配置文件为：

```text
shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml
```

## 执行顺序

### 1. Qualification（M-Q）

先创建独立结果目录，再运行不开 profiler 的 production workload：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=profile_outputs/minimax_m3_m_q/${RUN_ID}
mkdir -p "${RUN_DIR}"

set -o pipefail
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.minimax_m3_layer_profiling.benchmark \
  --config benchmarks/minimax_m3_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml \
  --production-profile --profile none \
  2>&1 | tee "${RUN_DIR}/qualification.log"

QUALIFICATION_RC=${PIPESTATUS[0]}
printf 'git_commit=%s\nqualification_exit_code=%s\n' \
  "$(git rev-parse HEAD)" "${QUALIFICATION_RC}" \
  | tee "${RUN_DIR}/run_meta.txt"
```

只有 M-Q 达到后面的验收标准后才能进入 M-P。

### 2. Production Torch trace（M-P）

使用新目录采集正式 trace，不得复用 qualification 目录：

```bash
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=profile_outputs/minimax_m3_m_p/${RUN_ID}
mkdir -p "${RUN_DIR}/traces"

set -o pipefail
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.minimax_m3_layer_profiling.benchmark \
  --config benchmarks/minimax_m3_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml \
  --production-profile --profile torch \
  --profile-output-dir "${RUN_DIR}/traces" \
  2>&1 | tee "${RUN_DIR}/torch_profile.log"

PROFILE_RC=${PIPESTATUS[0]}
printf 'git_commit=%s\nprofile_exit_code=%s\n' \
  "$(git rev-parse HEAD)" "${PROFILE_RC}" \
  | tee "${RUN_DIR}/run_meta.txt"
```

YAML 已固定 `profiler_with_stack=true`、`warmup_iters=3` 和 `profile_iters=1`。若要对
MiniMax M3 时延作正式统计，应使用相同配置独立运行至少五次，逐 iteration 先取 8 个 EP
ranks 的最大值，再报告中位数；不能把各 rank 的 CUDA 时间直接相加。

## 验收标准

M-Q 要求退出码为 0，8 个 rank 均 `status=PASS`，每请求都生成 2 个 token，且
`decode_executions_per_request=1`；拓扑必须为 TP1/DP8/EP8，请求必须按 DP rank 正确分片。

M-P 除满足 M-Q 的运行条件外，还要求 8 份 trace
非空，均包含 `layers.0..3` 的 prefill 和 decode execution，不包含 `layers.4+`；
trace 必须包含 CPU operators、CUDA kernels、recorded shapes 和非空 Python stack。

两阶段均不得出现 OOM、CUDA、NCCL、worker 或 KV-cache failure，退出后 GPU 显存必须
释放。MiniMax M3 结果只代表 dummy BF16 权重、当前 commit、固定 shape 和当前 backend，
不能解释成真实 checkpoint 的输出质量或 MXFP8 性能。

# MiniMax M3 前四层 Production Profiling

本目录使用真实 `LLM -> EngineCore -> production worker` 路径，以
`MiniMaxAI/MiniMax-M3` 的仓库内默认结构和 dummy BF16 权重构造
`MiniMaxM3SparseForCausalLM`，只将 `num_hidden_layers` 截断为 4。该配置不是
`MiniMax-M3-MXFP8`；若要测 MXFP8 checkpoint backend，必须另建配置和 qualification，
不能把本结果解释为 MXFP8 性能。
不使用自定义模型 wrapper，也不手工构造 attention metadata 或 KV cache。

固定模型资产 `model_config/config.json` 来自官方
[`MiniMaxAI/MiniMax-M3`](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json)
配置（2026-09-12 核对），并随 benchmark 一起由 Git 管理。运行时只通过
`hf_overrides` 将 text layer 数截断为 4；配置文件中的其余 text/vision 架构字段保持
官方值。

前四层覆盖两类结构：

```text
layers.0..2 = full attention + dense MLP
layer.3     = sparse attention + routed MoE/shared expert
```

正式 workload 是 global BS32、每请求 4096-token prefill，并生成 2 个 token，
因此同一次运行包含完整 prefill 和一次 single-token decode。正式 trace 使用
Torch Profiler、eager execution、`with_stack=true`。

## 实验清单

MiniMax M3 对比两组 TP/DP 策略；两组都启用 EP8，并固定使用 AG+RS：

| ID | 并行与通信 | 模式 | 目的 |
| --- | --- | --- | --- |
| M-Q1 | TP1/DP8/EP8 + AG+RS | `profile=none` | 验证 TP1 production workload |
| M-Q2 | TP2/DP4/EP8 + AG+RS | `profile=none` | 验证 TP2 production workload |
| M-P1 | TP1/DP8/EP8 + AG+RS | Torch、with-stack | 采集 TP1 前四层 trace |
| M-P2 | TP2/DP4/EP8 + AG+RS | Torch、with-stack | 采集 TP2 前四层 trace |

配置映射为：

```text
M-Q1 / M-P1 -> shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml
M-Q2 / M-P2 -> shapes/prefill_decode_bs32_p4096_tp2_dp4_ag_rs.yaml
```

## 执行顺序

### 1. Qualification（M-Q1、M-Q2）

先运行 M-Q1；通过后将 `CONFIG` 和 `EXPERIMENT` 改为 M-Q2，再执行相同命令。两个
qualification 必须使用独立目录。

```bash
CONFIG=benchmarks/minimax_m3_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml
EXPERIMENT=minimax_m3_m_q1_tp1_dp8
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=profile_outputs/${EXPERIMENT}/${RUN_ID}
mkdir -p "${RUN_DIR}"

set -o pipefail
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.minimax_m3_layer_profiling.benchmark \
  --config "${CONFIG}" \
  --production-profile --profile none \
  2>&1 | tee "${RUN_DIR}/qualification.log"

QUALIFICATION_RC=${PIPESTATUS[0]}
printf 'git_commit=%s\nqualification_exit_code=%s\n' \
  "$(git rev-parse HEAD)" "${QUALIFICATION_RC}" \
  | tee "${RUN_DIR}/run_meta.txt"
```

M-Q1 和 M-Q2 都达到后面的验收标准后才能进入正式采集。

### 2. Production Torch trace（M-P1、M-P2）

按 M-P1、M-P2 顺序运行。分别选择与 M-Q1、M-Q2 相同的配置，且不得复用
qualification 或另一组 trace 的目录。

```bash
CONFIG=benchmarks/minimax_m3_layer_profiling/shapes/prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml
EXPERIMENT=minimax_m3_m_p1_tp1_dp8
RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_DIR=profile_outputs/${EXPERIMENT}/${RUN_ID}
mkdir -p "${RUN_DIR}/traces"

set -o pipefail
.venv/bin/torchrun --standalone --nproc-per-node=8 \
  -m benchmarks.minimax_m3_layer_profiling.benchmark \
  --config "${CONFIG}" \
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

M-Q1 和 M-Q2 均要求退出码为 0，8 个 rank 均 `status=PASS`，每请求都生成 2 个 token，
且 `decode_executions_per_request=1`。M-Q1 拓扑必须为 TP1/DP8/EP8，每个 DP rank
处理 4 个请求；M-Q2 必须为 TP2/DP4/EP8，每个 DP rank 处理 8 个请求。同一 TP group
处理相同请求，request-to-DP 映射不得重复或遗漏。

M-P1 和 M-P2 除分别满足对应 qualification 的运行条件外，还要求各自产生 8 份 trace
非空，均包含 `layers.0..3` 的 prefill 和 decode execution，不包含 `layers.4+`；
trace 必须包含 CPU operators、CUDA kernels、recorded shapes 和非空 Python stack。

两阶段均不得出现 OOM、CUDA、NCCL、worker 或 KV-cache failure，退出后 GPU 显存必须
释放。MiniMax M3 结果只代表 dummy BF16 权重、当前 commit、固定 shape 和当前 backend，
不能解释成真实 checkpoint 的输出质量或 MXFP8 性能。

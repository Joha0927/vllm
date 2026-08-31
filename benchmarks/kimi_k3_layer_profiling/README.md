# Kimi-K3 首个 AttnRes Block Profiling 工作文档

## 1. 背景与目标

目标硬件为单机 8 张 NVIDIA H20，每张显存 96 GiB。GPU 之间通过完整的
NVLink 域互联。服务器运行 Ubuntu 22.04，当前 CUDA Toolkit 为 13.0。

完整 Kimi-K3 无法放入该服务器，因此本项目不尝试启动完整模型。主 benchmark
运行首个真实尺寸的完整 AttnRes block；当前 Kimi-K3 配置中一个 block 为逻辑层
0 至 11，共 12 个连续 decoder layers。当前阶段只把这个 block 的 profiling 流程
完整走通；单层模式、中间/末端 block 和完整模型外推均延后决策。在给定输入形状
和并行策略后，该工具应当能够：

- 优先在 GPU 上构造逻辑层 0 至 11 的首个完整 AttnRes block；
- 当 12 层受初始化峰值或 cache 限制无法运行时停止并重新评审，不把部分 block
  静默当作完整 block；
- 保留真实 hidden size、attention、MoE、量化和通信行为；
- 分别测量 prefill、decode、chunked prefill 和 mixed batch；
- 支持 TP、DP、EP 和 DCP 的组合；
- 输出 block 延迟、各 rank 延迟、显存和 profiler trace；
- 按 KDA、MLA、MoE 和 AttnRes 对 block 内开销进行 profiler 归因；
- 形成可复现的首个 block 性能报告，并明确它不能代表完整模型。

本项目首先服务于 H20 上的性能研究，不以生成正确文本或部署线上服务为目标。

核心原则是：**缩小模型规模，但不缩短 vLLM execution stack**。主 benchmark
应尽量保留真实的 model runner、输入准备、forward context、cache 注册、分布式
通信和模型 forward 路径。当前阶段不实现直接调用 `KimiDecoderLayer.forward`
的单层 benchmark。

## 2. 非目标

- 不把单个任意层的时间直接乘以总层数；
- 不下载或加载完整 Kimi-K3 权重；
- 不在第一阶段修改 C++/CUDA kernel；
- 不把 PyTorch Profiler 或 Nsys trace 中的耗时直接作为最终 benchmark 数字；
- 不把原始 trace、模型权重、token 或服务器敏感信息提交到 Git；
- 不在服务器上修改 tracked 文件。代码和配置从 GitHub 单向发布到服务器。
- 不在主 benchmark 中手工伪造一套简化 metadata 后直接调用 decoder layer；
- 不把 cache eviction buffer 等同于完整模型中的真实权重流式访问。
- 不实现 token-smoke，不要求输出 token 或文字；
- 当前工具不加载 embedding、LM head、vision tower、tokenizer 或 sampler；
- 不把 embedding、LM head、scheduler 和 sampling 时间混入 block 延迟。
- 当前阶段不实现任意逻辑层或任意逻辑 block；
- 当前阶段不根据首个 block 外推 93 层 decoder 或完整服务延迟；
- 当前阶段不承诺实现单层 profiling，是否需要由首个 block 报告决定。

## 3. 已知硬件与实现约束

H20 属于 Hopper，CUDA compute capability 为 9.0。当前 Kimi-K3 NVIDIA
实现中，不同路径在 H20 上的行为不同：

- KDA fused decode 支持 SM90；
- FlashKDA 支持 Hopper；
- 原生 Kimi-K3 AttnRes CUDA kernel 仅支持 SM100，H20 使用 Triton fallback；
- latent-MoE tail fusion 仅支持 SM100，H20 使用 portable fallback；
- Kimi-K3 专用 GEMM-RS/GEMM-AR CuTeDSL 路径仅支持 SM100；
- routed experts 使用 MXFP4，必须记录 H20 实际选择的 MXFP4 MoE backend。

因此，本项目测量的是 Kimi-K3 在 H20 实际可运行 backend 上的性能，不能直接与
B200、GB200 或 GB300 的结果比较。

## 4. 为什么先只做首个 Block

Kimi-K3 decoder layer 不是完全同构的。至少需要区分：

- KDA attention layer；
- MLA attention layer；
- MoE layer；
- 可能存在的 dense layer；
- AttnRes block-write layer；
- 不同 AttnRes 历史 block 深度的 layer。

首个 12 层 block 能自然保留连续层、AttnRes 累积和 block-write，而且可直接使用
逻辑层 0 至 11，避免第一版就引入逻辑 offset 和合成历史 AttnRes state。先完成
一个 block 的全流程，更容易判断显存、execution stack、backend 和 profiler 是否
可行。首个 block 不能代表模型中后部更深的 AttnRes 历史，因此当前报告只描述
这个 block，不做完整模型外推。

首个 block 的物理层号和逻辑层号暂时一一对应：

```text
physical_layer_idx = 0..11
logical_layer_idx  = 0..11
```

只有未来扩展到中间/末端 block 时，才需要把物理层号和逻辑层号解耦，并注入
历史 AttnRes state。该扩展不属于当前阶段。

benchmark 的 I/O 固定为 packed synthetic hidden states，不经过 token 接口：

```text
[M, 7168] BF16 hidden states
    -> Kimi 逻辑层 0..11
    -> (pending_hidden_states, prefix_sum, block_residual_bank)
```

这里不能调用截断 `KimiLinearModel` 的模型末端 `output_attn_res` 聚合。完整模型中
第 11 层之后仍要把上述三类状态传给第 12 层，而不是提前生成最终 hidden states。
输出验证 shape、dtype、device、有限值、cache/state 更新和 rank 一致性，不验证
生成文本质量。

本项目使用 dummy weights，因此“实测 Kimi-K3 block”特指真实 shape、MXFP4
物理表示、H20 backend、TP8/EP8 通信和 vLLM execution stack；不代表真实
checkpoint 的 router、激活值或生成质量。报告统一标记为
`shape-faithful/backend-faithful block profiling`。

## 5. 总体实施路线

项目按以下四级验收推进，内部仍拆成七个实施阶段：

| 验收级别 | 目标 | 对应阶段 |
| --- | --- | --- |
| A：能运行 | 首个 12 层 block 可初始化并 forward | 0-1 |
| B：结构正确 | block 边界、层类型、AttnRes 状态和 backend 正确 | 2-4 |
| C：负载稳定 | shape、routing、rank 和计时可复现 | 5 |
| D：形成报告 | 得到首个 block 的可复现性能结论和限制 | 6 |

### 验证执行策略：本地优先、严格串行

服务器只能从 GitHub 拉取代码，因此尽量在本地消除错误，减少无效的
`commit -> push -> server pull` 循环。每个阶段严格按以下顺序执行：

```text
实现一个最小步骤
    -> 本地可执行验证
    -> 本地验证全部通过
    -> 提交并推送该步骤
    -> 仅在确有 GPU 依赖时由服务器拉取
    -> H20 验证并保存证据
    -> 当前步骤验收通过
    -> 才进入下一步骤
```

任何验证失败时都停留在当前步骤。修复后从该步骤的本地验证重新开始，不跳过
失败项，也不把多个未经服务器验证的 GPU 相关改动堆入同一次发布。

验证位置按依赖划分：

| 验证内容 | 默认位置 | 原因 |
| --- | --- | --- |
| 配置解析、CLI、dry-run、manifest schema | 本地 | 不依赖 GPU |
| 逻辑层/block 分类、范围和 AttnRes 深度计算 | 本地 | 可由 config 和纯 Python 验证 |
| tiny config parity、CPU 单元测试 | 本地 | 成本低，反馈快 |
| import、lint、类型和文档检查 | 本地 | 不应占用服务器 |
| CUDA 单元测试 | 本地有兼容 NVIDIA GPU时优先本地，否则服务器 | 依赖 CUDA/backend |
| H20 MXFP4/KDA/MLA backend | 服务器 | 依赖 SM90 和服务器安装环境 |
| TP8/EP8/NCCL、逐 rank experts | 服务器 | 依赖 8 GPU 拓扑 |
| 1/4/8/12 层峰值显存 | 服务器 | 依赖 8×H20 96 GiB |
| Nsys/NCU 和最终性能数据 | 服务器 | 必须测目标硬件 |

每一步的验证记录至少包含：Git commit、验证位置、命令、退出码、关键输出、
通过/失败结论。服务器结果写入对应 manifest；本地测试结果记录在提交说明或阶段
验收记录中。

### 阶段 0：环境和硬件基线

- [ ] 确认 8 张 H20 均可见；
- [ ] 确认 compute capability 为 9.0；
- [ ] 保存 `nvidia-smi topo -m`；
- [ ] 使用 Python 3.12 和 `uv` 创建 `.venv`；
- [ ] 使用 CUDA 13.0 对应的预编译 wheel 安装 editable vLLM；
- [ ] 确认当前导入的 `vllm` 来自本仓库；
- [ ] 确认 PyTorch NCCL 可用；
- [ ] 在本目录提交经过核对的 Kimi-K3 config-only 快照；
- [ ] 服务器从本地 Git 工作区读取 config，不访问 Hugging Face；
- [ ] 保存 `vllm collect-env` 输出；
- [ ] 确认 Nsys 是否可用。

建议的环境验证命令：

```bash
git branch --show-current
git rev-parse HEAD
nvidia-smi
nvidia-smi topo -m
nvcc --version

.venv/bin/python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.nccl.version())'
.venv/bin/python -c 'import vllm; print(vllm.__version__); print(vllm.__file__)'
.venv/bin/python -c 'from vllm.transformers_utils.config import get_config; c=get_config("benchmarks/kimi_k3_layer_profiling/model_config", False); t=c.text_config; print(t.num_hidden_layers, t.attn_res_block_size, t.hidden_size, t.num_experts)'
vllm collect-env
nsys --version
```

#### 阶段 0 验证

验证位置：服务器（硬件与运行环境基线无法由本地替代）。

1. 将以上命令的完整输出保存到 `profile_outputs/env/`，并在环境记录中写入
   `git rev-parse HEAD` 的结果；
2. 确认 `torch.cuda.device_count() == 8`，每张卡名称包含 `H20`，capability 为
   `(9, 0)`；
3. 确认当前分支和 commit 与 GitHub 发布版本一致；
4. 运行一个 8 rank NCCL all-reduce smoke，确认所有 rank 正常退出；
5. 确认本地 config 输出为 93 层、AttnRes block size 12、hidden size 7168、
   896 experts；
6. 确认没有启用 CPU offload。

通过标准：六项全部满足。失败时停止模型实验，不带着环境问题进入阶段 1。

### 阶段 1：12 层 Block 显存可行性

先使用 vLLM 已有能力验证最小方案：

- `--language-model-only` 排除视觉模型；
- `--load-format dummy` 避免下载完整权重；
- `--skip-tokenizer-init` 避免初始化和访问 tokenizer；
- `--hf-overrides` 先将 text model 缩短为 1 层，再测试 12 层；
- `--enforce-eager` 简化首次调试；
- 降低 `max_model_len`、`max_num_seqs` 和 KV cache 占用。

按 `1 -> 4 -> 8 -> 12` 层逐步增加，不直接从 1 跳到 12。1/4/8 层只用于定位
初始化失败和控制显存风险，不进入正式 profiling 数据集。每次固定：

```text
TP=8, DP=1, EP=8, DCP=1, PP=1
language-model-only, dummy weights, eager
```

阶段 1 使用现有 `vllm bench latency` 只做加载可行性检查。以下是 12 层命令；
1/4/8 层仅修改 `num_hidden_layers`：

```bash
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random

vllm bench latency \
  --model benchmarks/kimi_k3_layer_profiling/model_config \
  --skip-tokenizer-init \
  --language-model-only \
  --load-format dummy \
  --hf-overrides '{"text_config":{"num_hidden_layers":12}}' \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --decode-context-parallel-size 1 \
  --enforce-eager \
  --max-model-len 2048 \
  --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 4294967296 \
  --batch-size 1 \
  --input-len 128 \
  --output-len 1 \
  --num-iters-warmup 1 \
  --num-iters 1
```

该命令仍会构造/执行 CausalLM 外壳，因此其 latency 不是 block latency，只用于
回答 12 层能否初始化和运行。正式 profiling 必须使用阶段 2 的专用 wrapper。

这一阶段回答以下问题：

- 真实维度的 Kimi-K3 层能否在 8 张 H20 上逐步扩展到完整 block；
- MXFP4、KDA、MLA 和 MoE 分别选择了什么 backend；
- 各层数下参数、KV/KDA state 和通信 buffer 占用多少显存；
- TP8 和 EP8 是否可以正常初始化。
- 首个完整 AttnRes block 是否能在每卡 96 GiB 内完成初始化和 forward；
- 初始化峰值、稳态权重、workspace 和 cache 各占多少显存。

`--load-format dummy` 只解决“不下载完整 checkpoint”，不能自动证明执行形态
等价于真实 MXFP4 checkpoint。smoke test 还必须逐 rank 验证：

- 权重加载后最终的 storage、shape 和 dtype；
- 实际选择的 MXFP4 MoE backend 和 fallback 原因；
- 每个 rank 的权重显存；
- 首次稳定迭代中实际出现的关键 kernel 名称。

这一阶段的时间不能用于完整模型外推，因为尚未完成稳定计时和校准。

#### 阶段 1 验证

验证位置：服务器。本地只先检查命令、配置和 override 解析，不在本地猜测 H20
显存或 backend 结果。

每个层数都执行以下验证：

1. 在第二个终端用 `nvidia-smi dmon -s m -d 1` 覆盖整个进程生命周期；
2. 完成一次现有 latency smoke，保存 stdout/stderr 和 dmon 输出；
3. 检查所有 rank 正常退出；
4. 检查实际层数、逻辑层号、KDA/MLA 类型、local expert ID 和 MXFP4 backend；
5. 记录外部观测到的每卡最大显存。更细的模型构造、dummy quantization、KV cache
   和首次 forward 分阶段 allocated/reserved 在阶段 2 wrapper 中实现。

通过标准：

- 1 层必须通过，否则停止并修复环境/backend，但不据此启动单层 profiling 开发；
- 在固定 4 GiB KV cache 时，12 层若每卡峰值不超过 80 GiB，进入主 benchmark；
- 峰值在 80 至 88 GiB 时标记为高风险，先评估 profiler 和目标 context 余量；
- 超过 88 GiB、OOM 或初始化不稳定时停止在阶段 1，记录最大成功层数并重新评审；
- 不允许通过 CPU offload 获得“通过”。

#### 阶段 1 实际验收记录（2026-08-28）

验收 commit：`5e37eba8b47d545e681d7d284ba63eb57ef3a8f5`。

12 层 direct smoke 在单机 8×H20 上完成初始化、一次 warmup 和一次正式
generation，`benchmark_exit_code=0`，8 个 worker 正常退出，无 OOM、NCCL error、
timeout 或 worker crash。固定 4 GiB KV cache 时，各卡进程生命周期峰值均为
34829 MiB（约 34.0 GiB），低于 80 GiB 安全线。运行配置确认如下：

```text
num_hidden_layers=12
load_format=dummy
TP=8, DP=1, EP enabled, DCP=1, PP=1
execution_mode=eager
quantization=mxfp4
```

普通运行日志已明确证明以下 runtime backend，而不是仅从 config 推断：

```text
KDA decode: fused KDA decode kernel enabled
KDA prefill: FlashKDA
MLA attention: FLASH_ATTN_MLA
MLA prefill: FLASH_ATTN
MXFP4 MoE backend: MARLIN
MoE expert implementation: MarlinExperts
EP rank 0 local/global experts: 112/896
EP rank 0 local-to-global expert map: 0..111
NCCL world size: 8
KV cache: 4.0 GiB per worker
```

验收结论：`PASS WITH MINOR EVIDENCE GAPS`。阶段 1 的核心问题——12 层能否运行、
是否低于显存安全线，以及 H20 上实际 KDA、MLA 和 MXFP4 MoE backend——已经回答，
允许进入阶段 2。以下缺口不能从现有 `vllm bench latency` INFO 日志充分证明，转为
阶段 2 首批 instrumentation 验收项，不视为已完成或静默跳过：

- 8 个 rank 各自的 local expert 数量和完整 expert ID；
- 每个 rank 的最终 dummy MXFP4 参数 storage、shape 和 dtype；
- MoE dispatch/combine 实际使用 `allgather_reducescatter` 的运行时确认；
- 逻辑层 0 至 11 的逐层 KDA/MLA、dense/MoE 和 block-write 分类；
- H20 上 AttnRes、latent-MoE tail 和专用 GEMM 路径的完整 fallback 原因。

低层 CUDA kernel 名称不再要求由普通 INFO 日志完整提供，留到阶段 5 使用 Nsys
采集和分类。阶段 2 的最小 instrumentation 必须先补齐上述结构与 backend 证据，
再继续实现正式计时路径。

### 阶段 2：实现 profiling benchmark 框架

#### 当前实现与 8-rank distributed-group 实测记录

截至 2026-08-29，已实现配置和 dry-run 骨架：

- `config.py` 从固定的 `model_config/config.json` 读取模型结构；YAML 只能配置
  workload、层数和运行策略，不能覆盖 hidden size、专家数、注意力头数或起始层；
- `benchmark.py` 已提供 `--dry-run`、`--list-layer-types`、manifest 预览和
  `--distributed-smoke`；
- `distributed.py` 将 benchmark 配置转换为 vLLM `ParallelConfig`，先在 NCCL
  world group 上检查 8 个 rank 的配置摘要一致，再初始化 TP/DCP/DP/EP group；
- 本地配置测试共 25 项通过，且 dry-run 不导入 Torch 或 vLLM。

服务器上的 8×H20 distributed-group smoke 已通过：

```text
world_size = 8
TP = 8
EP = 8
DCP = 1
rank/local_rank/tp_rank/ep_rank = 0..7 exactly once
dcp_rank = 0 on every rank
device = NVIDIA H20 on every rank
requested_all2all_backend = allgather_reducescatter
smoke_scope = distributed_groups
PASS records = 8
torchrun_exit_code = 0
```

日志中未出现 traceback、NCCL error、timeout 或 worker 异常退出。该结果只证明
基础 NCCL world 和 vLLM distributed groups 能够建立，不能证明 worker、model
runner、模型加载、MoE dispatch/combine 或 block forward 已执行。backend 字段是
请求值；实际 MoE backend 必须在 block forward 中另行确认。

本次 `run_meta.txt` 的 `git_commit` 为空。这不影响 distributed smoke 判定，但属于
追溯信息缺口；正式 block 实验必须在启动前记录 `git rev-parse HEAD`，不得沿用该
空值。

计划在本目录新增：

```text
benchmarks/kimi_k3_layer_profiling/
├── __init__.py
├── README.md
├── benchmark.py
├── block_model.py
├── config.py
├── workload.py
├── layer_factory.py
├── runner.py
├── timing.py
├── profiling.py
├── result.py
├── model_config/
│   └── config.json
└── shapes/
    ├── smoke.yaml
    ├── prefill.yaml
    ├── decode.yaml
    └── parallel_sweep.yaml
```

各文件职责：

- `benchmark.py`：命令行入口、分布式启动和实验循环；
- `block_model.py`：benchmark 专用模型 wrapper，只运行层 0 至 11 并返回 block
  边界状态，不构造 embedding/LM head，不执行模型末端 `output_attn_res`；复用或
  委托 Kimi 的 hybrid-cache state shape/copy 接口；若调用 `compute_logits` 则
  fail fast，防止 benchmark 意外进入 sampling 路径；
- `config.py`：读取 YAML、校验 shape 和并行策略；
- `workload.py`：把实验 shape 转换为 vLLM 可消费的请求和调度输入；
- `layer_factory.py`：复用真实 `KimiDecoderLayer` 构造逻辑层 0 至 11 的
  `ModuleList` 和 block state；不能直接实例化会附带 embedding/output 聚合的
  `KimiLinearModel`；
- `runner.py`：复用真实 vLLM model runner 完成输入准备、forward context、
  cache metadata 和模型 forward，不自行实现简化执行栈；
- `timing.py`：warmup、CUDA event、同步和各 rank latency 汇总；
- `profiling.py`：控制 PyTorch Profiler、Nsys capture range 和 Proton；
- `result.py`：汇总各 rank 数据并写出 JSON/Markdown；
- `shapes/*.yaml`：可复现实验配置。
- `model_config/config.json`：从 GitHub 分发的 Kimi-K3 config-only 快照；
- `tools/profiler/nsys_profile_tools/vllm_engine_model.json`：在现有 `vllm` 映射内
  新增 `kimi-k3` 的 KDA、MLA、MXFP4 MoE、AttnRes 和 NCCL kernel 分类。

命令行接口至少应包含：

```text
--model
--num-layers
--phase
--batch-size
--query-len
--context-len
--tensor-parallel-size
--data-parallel-size
--decode-context-parallel-size
--enable-expert-parallel
--all2all-backend
--execution-mode
--routing-strategy
--cache-mode
--warmup-iters
--repeat-iters
--output-dir
--profile
```

还应支持：

- `--dry-run`：只解析配置、打印 layer 类型和显存估算；
- `--list-layer-types`：列出完整 config 中每个逻辑层的分类；
- `--config FILE`：从 YAML 批量执行；
- `--profile none|torch|cuda|proton`：选择 profiler。

这里的 `--profile` 是本 benchmark 新增的适配层，不是可以直接假设存在的 vLLM
参数。实现必须映射到真实 `ProfilerConfig` 和 start/stop：

- `torch`：配置绝对 `torch_profiler_dir`，输出逐 rank `.pt.trace.json.gz`；
- `cuda`：调用 `torch.cuda.profiler.start/stop`，供外部 Nsys capture range 使用；
- `proton`：配置 Proton 输出目录和格式；
- `none`：不创建 profiler，专门用于正式 latency。

第一版固定逻辑起始层为 0，正式实验要求 `--num-layers 12`。该参数保留仅用于
1/4/8/12 层显存阶梯和失败排查，不提供单层类型选择或逻辑 offset。若 12 层最终
无法安全运行，必须先重新评审目标，不能静默把较少层数称为完整 block。

#### 阶段 2 验证

验证顺序：先完成配置、schema 和结果聚合的本地验证；全部通过并提交后，服务器
只执行真实 8 rank 输出、execution stack 和最小 P1/D1 smoke。

1. `--dry-run` 对非法 TP/DP/EP/DCP、负长度和当前阶段不允许的起始层/层数必须
   fail closed；正式模式只接受逻辑起始层 0、层数 12；
2. 同一 YAML 连续运行两次，除时间和温度等动态字段外，manifest 必须一致；
3. 本地使用 8 份合成 rank record 验证 summary JSON/Markdown 和 manifest 聚合；
4. 本地验证 group/local-expert 计算，服务器再与真实 8 rank 输出交叉检查；
5. 服务器 P1 和 D1 均生成 8 个真实 rank JSON 和完整 summary/manifest；
6. 用 trace 或 instrumentation 证明主路径经过 model runner 输入准备、真实
   forward context 和模型 forward，而不是直接调用 decoder layer。
7. 检查模型参数名中不存在 `embed_tokens`、`lm_head`、`vision_tower`，并检查
   `model.layers.12` 不存在；
8. 检查 block forward 没有调用 `output_attn_res_norm/output_attn_res_proj`。
9. 在 worker 内部分别记录模型构造后、dummy quantization 后、KV cache 分配后、
   首次 forward 后和 CUDA Graph capture 后的 allocated/reserved。
10. 主 benchmark 调用 `compute_logits` 或 sampler 时测试必须失败，而不是产生一份
    混入 LM head 的结果。

通过标准：以上十项全部通过，并且失败实验也能写出失败阶段和异常类型。

### 阶段 3：首个 Block Execution Stack 接入

本阶段只接入逻辑层 0 至 11。wrapper 复用真实 `KimiDecoderLayer`、KDA/MLA、
MoE 和 cache state 接口，但不能直接实例化会附带 embedding、模型末端 AttnRes
和 norm 的 `KimiLinearModel`。只有在必要时才把可复用的 layer-loop/cache helper
从生产模型中做行为不变的窄范围抽取。禁止为了未来任意逻辑 block 提前加入
logical offset 或合成历史 AttnRes state。

优先通过 `--model-class-overrides` 将 `KimiLinearForCausalLM` 指向
`block_model.py` 中的 benchmark wrapper，避免把研究接口加入正式 serving CLI：

```text
--model-class-overrides \
  '{"KimiLinearForCausalLM":"benchmarks.kimi_k3_layer_profiling.block_model:KimiK3BlockProfiler"}'
```

如果 worker 子进程无法导入 `benchmarks` 命名空间，必须先修复包路径/模块注册，
不能回退到标准 CausalLM 并把额外开销从结果中“估算扣除”。

首个 block 必须保留：

1. `model.layers.0...11` 的真实模块和 prefix；
2. vLLM model runner 的输入准备与 forward context；
3. 每层真实 KDA/MLA 和 MoE 选择；
4. block 内 AttnRes prefix、bank、第 0 层 block-write 和 block 输出状态；
5. 真实 KV/KDA cache metadata 和 TP8/EP8 collective；
6. synthetic hidden states 输入，但不构造 embedding 和 LM head。

#### 阶段 3 验证

验证顺序：模块树、层分类和 tiny block parity 先在本地完成；真实尺寸 forward 和
backend 检查放到服务器。

1. 确认只创建 `model.layers.0...11`，不存在第 12 层、embedding、LM head 或
   vision tower 参数；
2. 比较 0 至 11 层分类与原始 K3 config，必须逐层一致；
3. 用 tiny config 比较 isolated 12-layer block 返回的
   `(pending_hidden_states, prefix_sum, block_residual_bank)` 与小型完整模型在
   第 11 层后的内部状态；
4. 证明主路径经过 model runner 输入准备、forward context、cache 注册和 model
   forward；
5. H20 上完成 P1/D1 forward，并确认实际 backend、collective 和 AttnRes kernel。

通过标准：模块范围准确；tiny block 三类边界状态均满足测试容差；P1/D1 状态的
shape、dtype 和有限值正确；8 rank 无 hang、cache 冲突或 rank divergence。

### 阶段 4：正确性与分布式验证

测试设计问题：

1. 模块用途：在不加载完整模型的情况下复现首个真实 K3 连续 block；
2. I/O 合约：输入为 packed hidden states、position 和真实 cache metadata，输出为
   pending hidden states、prefix sum 和 block residual bank；
3. 防止的故障：逻辑层分类错误、AttnRes 深度错误、cache 名冲突、collective
   shape 不一致；
4. 最低成本测试：tiny config block parity，其次 2 GPU distributed smoke。

计划新增：

```text
tests/models/kimi_k3/test_block_profiling.py
```

测试至少覆盖：

- [ ] 默认配置下现有 K3 构造行为不变；
- [ ] 0 至 11 层的 KDA/MLA 和 MoE 分类与完整 config 一致；
- [ ] block 内 AttnRes bank 和 block-write 位置正确；
- [ ] block 模式没有创建第 12 层、embedding、LM head 或视觉参数；
- [ ] block forward 没有执行模型末端 `output_attn_res`；
- [ ] 意外进入 logits/sampling 路径时 fail fast；
- [ ] tiny config 下，isolated block 与小型完整模型对应连续层输出一致；
- [ ] 主 benchmark 确实经过 model runner 输入准备和 forward context；
- [ ] TP/EP 下所有 rank tensor shape 一致。

测试命令必须通过 `.venv/bin/python` 运行：

```bash
.venv/bin/python -m pytest tests/models/kimi_k3/test_block_profiling.py -v
.venv/bin/python -m pytest tests/models/kimi_k3/test_kda.py -v
.venv/bin/python -m pytest tests/models/kimi_k3/test_attn_res.py -v
.venv/bin/python -m pytest tests/models/kimi_k3/test_sequence_parallel.py -v
```

#### 阶段 4 验证

验证顺序：

1. 本地先运行新增 CPU/tiny 单元测试；
2. 本地完成 import、配置测试和 `pre-commit`，保存命令和结果；
3. 本地有兼容 CUDA GPU 时运行 1 GPU KDA/MLA/AttnRes 测试；否则放到服务器；
4. 服务器运行 2 GPU tiny distributed smoke，检查 collective shape 和退出状态；
5. 最后在 H20 上运行 8 GPU、P1/D1 首个 block smoke。

通过标准：相关测试与 lint 全部通过；tiny block parity 达到测试中声明的
容差；8 GPU 不发生 hang、rank divergence、NaN/Inf 或 cache 冲突。

### 阶段 5：性能测量和 profiler

最终 latency 使用 CUDA event 或独立 wall-clock benchmark 测量。Profiler 只用于
解释时间花在哪里。

#### 普通计时

- 先完成 kernel JIT、autotune、communicator 初始化和 CUDA Graph capture；
- warmup 至少 5 次，正式重复 30 至 100 次；
- 输出 min、P10、P50、P90 和 max；
- 分布式 latency 使用最慢 rank，而不是 rank 平均值；
- 同时记录每张 GPU 的峰值 allocated/reserved memory。

#### PyTorch Profiler

用途：查看 operator、shape、Python/CUDA 映射和 vLLM iteration 标记。Trace 使用
Perfetto 查看。正式计时时关闭 stack、shape 和 memory recording。

采集命令由 benchmark 适配层实现，示例：

```bash
.venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
  --config benchmarks/kimi_k3_layer_profiling/shapes/prefill.yaml \
  --profile torch \
  --output-dir profile_outputs/torch_p1
```

验收时必须看到至少 8 个带 worker/rank 标识的 `.pt.trace.json.gz`，并能由
Perfetto 打开。

#### Nsight Systems

主要用于：

- GPU idle gap；
- CPU launch gap；
- NCCL 与计算重叠；
- 各 rank 到达 collective 的偏斜；
- CUDA Graph capture/replay；
- MoE dispatch、expert compute 和 combine 的时间线。

使用 Nsys 时设置：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

正式 Nsys 命令：

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p profile_outputs

nsys profile \
  --trace=cuda,nvtx,osrt \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat \
  --force-overwrite=true \
  --output profile_outputs/nsys_p1 \
  .venv/bin/python -m benchmarks.kimi_k3_layer_profiling.benchmark \
    --config benchmarks/kimi_k3_layer_profiling/shapes/prefill.yaml \
    --profile cuda \
    --output-dir profile_outputs/nsys_p1_results
```

`--profile cuda` 必须在 warmup 和 cache priming 完成后调用 CUDA profiler start，
采集少量稳定 iteration 后 stop。最终验收文件为
`profile_outputs/nsys_p1.nsys-rep`；若没有该文件则 profiling 未完成。

可使用仓库内的 `tools/profiler/nsys_profile_tools/gputrc2graph.py` 将 Nsys GPU
trace 转换为便于分析的图结构；原始 `.nsys-rep` 和导出的数据库仍不提交 Git。
使用前必须修改 `tools/profiler/nsys_profile_tools/vllm_engine_model.json`，在现有
`vllm` 节点内增加 `kimi-k3`。不要另建一个同样以 `vllm` 为顶层键的 JSON；当前
loader 使用浅层 `dict.update`，会覆盖已有 `vllm` 模型映射。未添加 Kimi-K3
映射时大量 kernel 会落入 `misc`。

分析命令必须使用项目 Python 环境：

```bash
.venv/bin/python tools/profiler/nsys_profile_tools/gputrc2graph.py \
  --in_file profile_outputs/nsys_p1.nsys-rep,vllm,kimi-k3,0 \
  --out_dir profile_outputs/nsys_p1_analysis \
  --title "Kimi-K3 block 0-11 P1"
```

#### Triton Proton

用于查看 Triton 聚合调用树。Proton 要求 eager execution，不能用它得出 CUDA
Graph 模式的最终性能结论。

#### Nsight Compute

先通过 Nsys 找到最重的 1 至 3 个 kernel，再用 NCU 分析 occupancy、HBM/L2
带宽、Tensor Core 利用率、warp stall 和 roofline。不要直接用 NCU profile 整个
服务进程。

分析报告按组件归因，而不是只罗列 kernel：

- KDA/MLA：projection、prefill/decode attention、cache/state 读写；
- MoE：routing、dispatch/all-to-all、expert GEMM、combine；
- 公共路径：norm、AttnRes read/write、collective、CPU launch 和 GPU idle。

组件耗时来自 profiler，端到端延迟来自独立计时；二者允许因并发重叠而不相加。

#### 阶段 5 验证

验证位置：最终计时和 profiler 在服务器；配置展开、结果聚合、统计计算和报告生成
先用本地合成数据验证。

1. 对 P1、P2、D1、D2 各重复执行至少 3 个独立进程 run；
2. 每个 run warmup 后采集 30 至 100 次，使用最慢 rank 作为 distributed latency；
3. 若同配置跨 run 的 P50 极差超过 5%，继续增加 warmup 并排查时钟、温度、JIT、
   autotune 和后台进程，不发布该数据；
4. 比较 profiler 开关前后的独立计时，确认正式数字来自 profiler 关闭状态；
5. Nsys trace 中确认 attention/MoE 计算、NCCL、idle gap 和 rank skew 可定位；
6. 对至少一个 shape 比较 block 总时间与逐层 profiler 归因，明确并发重叠导致的
   不可加性。
7. Decode trace 的 capture range 中不包含 cache priming，且 cache/state 在计时前
   已由真实 prefill 写入；
8. 对实际启用的 profiler 检查预期扩展名、逐 rank 命名和文件可读性；Nsys 和
   PyTorch trace 为当前阶段必需，Proton 为可选补充。

通过标准：核心 shape 跨 run P50 极差不超过 5%，没有失败 iteration，所有数字都
能追溯到 manifest、逐 rank 数据和对应 trace。

### 阶段 6：首个 Block 报告与后续决策

本阶段不做 93 层外推，只汇总逻辑层 0 至 11 的实测结论。报告至少包含：

- 适用的 Git commit、H20 环境和固定并行策略；
- 每个 canonical shape 的 P10/P50/P90、最慢 rank 和峰值显存；
- eager/CUDA Graph、routing 和 cache 条件；
- KDA/MLA、MoE、NCCL、AttnRes、CPU launch 和 GPU idle 的 profiler 归因；
- block 连续重复造成的缓存偏差；
- dummy MXFP4 权重与真实 checkpoint 的差异；
- 明确声明结果只适用于首个 block，不代表中间 block、末端 block 或完整服务。

#### 阶段 6 验证

验证位置：报告代码先在本地用合成 fixture 验证并随 benchmark 推送；真实报告在
服务器直接读取 H20 输出并生成，不要求服务器向 GitHub 或其他外部位置上传数据。
如安全策略允许人工下载小型脱敏 summary，可在本地复核；原始 trace 默认留在
服务器。

1. 从 manifest 自动生成所有表格，随机抽查至少 3 个 shape 与逐 rank 原始数据一致；
2. 确认 distributed latency 始终取最慢 rank，而不是 rank 平均值；
3. 确认正式 latency 来自 profiler 关闭的独立计时；
4. 确认每个性能结论都能追溯到 commit、shape、backend、routing 和 trace；
5. 确认报告中没有 93 层点估计、token 输出或完整服务延迟表述；
6. 根据 trace 和实验缺口，单独形成后续决策记录：是否需要单层归因、任意逻辑
   block、中间/末端 block 或更多并行策略。

通过标准：数据抽查一致、限制说明完整、首个 block 的 profiling 流程可由同一
commit 和配置复现。只有完成本阶段后，才评审是否启动单层 profiling。

## 6. 输入形状定义

vLLM 使用 packed tokens，实验配置不能只记录 `[batch, seq]`。

### Prefill

```text
B = 请求数量
Q = 每个请求本轮新增 token 数
K = 本轮结束后的总 context 长度
M = sum(Q)，本轮总 scheduled tokens
```

需要区分：

- 单请求长 prompt；
- 多请求短 prompt但总 token 数相同；
- 不等长请求；
- chunked prefill；
- prefix cache 命中与不命中。

### Decode

```text
B = 活跃序列数
Q = 通常为 1，speculative decoding 时可以大于 1
K = 每条序列已有 context 长度
M ≈ B × Q
```

MLA cache 访问随 context length 变化，因此 decode 实验必须记录和扫描 `K`。

Decode 不能只伪造 `K` 或 block table。每个 decode shape 必须先在 timed/profiled
range 外完成 cache priming：

```text
cache_setup: prefill
cache_setup_tokens_per_request: K - Q
timed_query_tokens_per_request: Q
```

priming 必须真实经过同一个 block 的 KV/KDA state 写入路径。完成 priming 后同步，
清零正式计时统计，但不清空 cache，再采集 decode iteration。D4/D5 若 priming 时间
过长，可以把预热结果作为独立 setup 阶段复用，但 manifest 必须记录创建方式；
不得用未初始化 cache 冒充 32K/128K context。

第一版固定使用以下 canonical shapes，先建立可复现基线，再扩展不等长、mixed 和
speculative workload：

| ID | Phase | B | 每请求 Q | 每请求 K | M |
| --- | --- | ---: | ---: | ---: | ---: |
| P1 | prefill | 1 | 128 | 128 | 128 |
| P2 | prefill | 1 | 2048 | 2048 | 2048 |
| P3 | prefill | 8 | 256 | 256 | 2048 |
| P4 | prefill | 32 | 64 | 64 | 2048 |
| D1 | decode | 1 | 1 | 2048 | 1 |
| D2 | decode | 32 | 1 | 2048 | 32 |
| D3 | decode | 128 | 1 | 2048 | 128 |
| D4 | decode | 32 | 1 | 32768 | 32 |
| D5 | decode | 32 | 1 | 131072 | 32 |

若某个 context 超过当前 cache 配额，结果应标记为 unsupported/OOM，而不是静默
缩短 `K`。

### 切换边界

K3 部分实现以 token 数 256 为路径切换阈值。目标 shape 附近至少补测：

```text
M = 255, 256, 257
```

还应覆盖 CUDA Graph capture size 和 padding 边界。

## 7. 初始并行策略矩阵

固定使用 8 张 GPU 时，优先测试：

| TP | DP | EP | Attention | Experts |
| ---: | ---: | ---: | --- | --- |
| 8 | 1 | 8 | TP8 | EP8 |
| 4 | 2 | 8 | 每个 DP group 内 TP4 | EP8 |
| 2 | 4 | 8 | 每个 DP group 内 TP2 | EP8 |
| 1 | 8 | 8 | attention 按 DP 复制 | EP8 |

第一轮只运行 `TP=8, DP=1, EP=8, DCP=1, PP=1`，先稳定 execution stack、
shape 和采集流程；通过验收后再展开矩阵。

第一轮固定 `PP=1`。首个 block 实验不能测量真实 pipeline bubble。

DCP 子矩阵根据 TP 约束选择，例如：

```text
TP8: DCP 1, 2, 4, 8
TP4: DCP 1, 2, 4
TP2: DCP 1, 2
```

第一轮 EP 使用默认 `allgather_reducescatter` backend。基础功能和数据可信后，
再安装和比较 DeepEP/DeepGEMM 等可选组件。

EP 不是一个可以脱离 TP/DP 单独相乘的抽象数字。vLLM 的 expert-parallel group
由实际并行布局共同决定，因此 manifest 必须保存每个 rank 的 TP、DP、EP、DCP
group 成员，以及该 rank 的 local expert 数量和 expert ID；不能只记录四个 size。

当前阶段固定逻辑层 0 至 11，不扫描其他层。报告仍应逐层记录 attention、FFN 和
block-write 类型，方便判断完成首个 block 后是否需要单层或其他逻辑 block。

## 8. MoE routing 策略

dummy weights 的 router 结果不代表真实请求分布。初始性能实验使用可控 routing：

```bash
export VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random
export VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1
```

后续至少比较：

- uniform random；
- 单热点或少数热点 expert；
- Zipf/skewed 分布；
- 若能从真实系统合规获取，则使用真实 expert histogram 回放。

实验 manifest 必须记录 routing 策略和随机种子。

## 9. 结果目录和 Git 规则

代码通过以下单向链路发布：

```text
本地开发机 -> GitHub fork -> H20 服务器只读 pull
```

服务器不得修改 tracked 文件。实验配置应当在本地提交后由服务器拉取。临时的
服务器覆盖配置必须是 untracked 文件，且 benchmark 将合并后的最终配置写入
manifest。

服务器结果目录不得按 Git commit 分层。所有阶段统一直接使用实验名称组织目录，
Git commit 写入 manifest、`run_meta.txt` 和最终 PASS/FAIL 记录，用于追溯运行代码。
建议目录结构：

```text
profile_outputs/
├── env/
├── distributed_smoke/
├── model_construction/
├── forward_smoke/
└── <experiment-name>/
    ├── manifest.json
    ├── run_meta.txt
    ├── rank_0.json
    ├── rank_1.json
    ├── ...
    ├── summary.json
    ├── summary.md
    └── traces/
```

后续所有阶段均遵守该规则。例如 forward smoke 固定写入
`profile_outputs/forward_smoke/`，Nsys 实验写入对应的
`profile_outputs/<experiment-name>/`，不得重新引入
`profile_outputs/<git-commit>/...`。如果同一实验需要保留多次运行，可在实验目录下
增加不含 commit 的运行编号或时间戳子目录；每次运行仍必须在 manifest 中记录完整
40 位 Git commit。

以下内容不得提交：

```gitignore
profile_outputs/
*.nsys-rep
*.sqlite
*.pt.trace.json
*.pt.trace.json.gz
*.hatchet
```

可以提交不含敏感信息的小型汇总表，但原始 trace 和权重不进入 Git 历史。

## 10. 实验 Manifest

每次运行至少记录：

```yaml
model: moonshotai/Kimi-K3
model_config_source: benchmarks/kimi_k3_layer_profiling/model_config/config.json
git_commit: null
gpu_name: null
gpu_count: 8
cuda_version: null
torch_version: null
vllm_version: null
nccl_version: null
dtype: bfloat16
weight_format: mxfp4
weight_source: dummy
measurement_fidelity: shape-faithful/backend-faithful
selected_backends: {}
weight_storage_by_rank: []
kernel_names: []
physical_layer_index: 0
profiling_unit: block
physical_start_layer: 0
logical_start_layer: 0
num_profiled_layers: 12
logical_layer_range: [0, 11]
block_output_contract:
  - pending_hidden_states
  - prefix_sum
  - block_residual_bank
applies_model_output_attn_res: false
layer_type: null
attention_type: null
ffn_type: null
attn_res_depth: null
attn_res_block_write: null
phase: null
batch_size: null
query_lengths: []
context_lengths: []
num_scheduled_tokens: null
tensor_parallel_size: null
data_parallel_size: null
expert_parallel_size: null
decode_context_parallel_size: null
rank_groups: {}
local_expert_ids_by_rank: {}
all2all_backend: null
execution_mode: eager
routing_strategy: null
cache_mode: null
cache_priming:
  method: null
  tokens_per_request: null
warmup_iters: null
repeat_iters: null
random_seed: 0
latency_ms_by_rank: []
distributed_latency_ms: null
memory_by_phase: {}
profile_artifacts: []
```

## 11. 每次实验的检查清单

运行前：

- [ ] 工作区干净；
- [ ] 记录 Git commit；
- [ ] 没有其他进程占用 GPU；
- [ ] GPU clocks、power state 和温度稳定；
- [ ] 8 个 rank 使用预期 GPU；
- [ ] 各 rank 的 TP/DP/EP/DCP group 和 local experts 符合预期；
- [ ] backend 日志与 manifest 一致；
- [ ] dummy 权重的最终 storage/dtype 与选定 backend 已验证；
- [ ] profiling unit、物理/逻辑层范围和 AttnRes 入站深度已核对；
- [ ] block 输出契约为三类内部状态，且末端 output AttnRes 未执行；
- [ ] kernel JIT、autotune 和 graph capture 不计入正式时间。

运行后：

- [ ] 八个 rank 均正常结束；
- [ ] 使用最慢 rank 计算 distributed latency；
- [ ] 记录峰值显存；
- [ ] 记录异常值和失败次数；
- [ ] trace 只包含少量稳定 iteration；
- [ ] decode cache priming 不在正式计时/trace range 内；
- [ ] profiling 文件存在、非空、可由对应工具打开；
- [ ] summary 中明确连续重复/cache-perturbed block、routing 和 eager/graph；
- [ ] 结果目录包含完整 manifest。

## 12. 当前里程碑

当前状态：

- [x] 创建 GitHub fork；
- [x] 创建 `codex/kimi-k3-layer-profiling` 分支；
- [x] H20 服务器成功拉取工作分支；
- [x] 确认 8 张 H20 位于完整 NVLink 域；
- [x] 确认服务器 CUDA Toolkit 为 13.0；
- [x] 完成 Python 3.12、uv 和 editable vLLM 环境安装；
- [x] 提交并验证 config-only Kimi-K3 模型目录；
- [x] 保存环境基线；
- [ ] 完成不修改模型代码的 1/4/8/12 层显存阶梯测试；
- [x] 确定 12 层 block 是否低于每卡 80 GiB 安全线；
- [x] 确认 H20 上 K3/MXFP4 实际 backend；
- [x] 实现 benchmark 配置、dry-run 和 deterministic manifest 预览；
- [x] 完成 8×H20 的 TP8/EP8/DCP1 distributed-group smoke；
- [x] 使用 8 份合成 rank record 验证结果聚合和 summary 输出；
- [ ] 实现 benchmark 专用 `KimiK3BlockProfiler` 模型 wrapper；
- [ ] 验证 block 边界三元状态且不执行模型末端 AttnRes；
- [ ] 实现首个 12 层 AttnRes block profiling；
- [ ] 完成正确性测试；
- [ ] 完成 TP8/EP8 首轮数据；
- [ ] 完成首个 block 的 canonical shape 和 profiler 数据；
- [ ] 增加并验证 Kimi-K3 Nsys kernel 分类；
- [ ] 生成并打开至少一份 `.nsys-rep` 和一组逐 rank PyTorch trace；
- [ ] 形成首个 block profiling 报告；
- [ ] 评审是否需要单层或其他逻辑 block（当前不实施）。

## 13. 工作原则

1. 每次只推进一个可验证的小阶段；
2. 先正确性和可解释性，再追求性能；
3. benchmark 数字和 profiler trace 分开采集；
4. 每个结论都能追溯到 commit、配置、硬件和原始结果；
5. 任何 backend fallback 都必须显式记录；
6. 首个 block 报告完成前，不启动单层、其他逻辑 block 或 93 层外推；
7. 当前报告只覆盖 decoder，不得暗示包含 embedding、LM head 或完整服务开销；
8. 每个阶段必须保存验证命令、结果、通过标准和失败原因后才能进入下一阶段。

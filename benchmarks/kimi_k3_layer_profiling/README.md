# Kimi-K3 单层 Profiling 工作文档

## 1. 背景与目标

目标硬件为单机 8 张 NVIDIA H20，每张显存 96 GiB。GPU 之间通过完整的
NVLink 域互联。服务器运行 Ubuntu 22.04，当前 CUDA Toolkit 为 13.0。

完整 Kimi-K3 无法放入该服务器，因此本项目不尝试启动完整模型，而是构建一个
Kimi-K3 单层 profiling 工具。在给定输入形状和并行策略后，该工具应当能够：

- 只在 GPU 上构造一个有代表性的 Kimi-K3 decoder layer；
- 保留真实 hidden size、attention、MoE、量化和通信行为；
- 分别测量 prefill、decode、chunked prefill 和 mixed batch；
- 支持 TP、DP、EP 和 DCP 的组合；
- 输出单层延迟、各 rank 延迟、显存和 profiler trace；
- 按 KDA、MLA、MoE/dense 和 AttnRes 深度对层分类；
- 使用多类代表层估算完整模型的执行时间，并明确误差来源。

本项目首先服务于 H20 上的性能研究，不以生成正确文本或部署线上服务为目标。

核心原则是：**缩小模型规模，但不缩短 vLLM execution stack**。主 benchmark
应尽量保留真实的 model runner、输入准备、forward context、cache 注册、分布式
通信和模型 forward 路径。直接调用 `KimiDecoderLayer.forward` 的实验只能作为
组件级诊断，不能作为单层端到端延迟或完整模型外推的主要数据。

## 2. 非目标

- 不把单个任意层的时间直接乘以总层数；
- 不下载或加载完整 Kimi-K3 权重；
- 不在第一阶段修改 C++/CUDA kernel；
- 不把 PyTorch Profiler 或 Nsys trace 中的耗时直接作为最终 benchmark 数字；
- 不把原始 trace、模型权重、token 或服务器敏感信息提交到 Git；
- 不在服务器上修改 tracked 文件。代码和配置从 GitHub 单向发布到服务器。
- 不在主 benchmark 中手工伪造一套简化 metadata 后直接调用 decoder layer；
- 不把 cache eviction buffer 等同于完整模型中的真实权重流式访问。

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

## 4. 为什么不能只保留普通的第 0 层

Kimi-K3 decoder layer 不是完全同构的。至少需要区分：

- KDA attention layer；
- MLA attention layer；
- MoE layer；
- 可能存在的 dense layer；
- AttnRes block-write layer；
- 不同 AttnRes 历史 block 深度的 layer。

另外，物理上只构造一层时，该层在 vLLM 中通常会成为 `model.layers.0`。但是我们
可能希望它表现为完整模型中的第 30 层或第 60 层。因此必须区分：

```text
physical_layer_idx = 0
logical_layer_idx  = 完整 Kimi-K3 中的真实层号
```

物理层号用于模块名和 cache 注册；逻辑层号用于确定 KDA/MLA、MoE/dense、
AttnRes block-write 状态和已有 block 数量。

## 5. 总体实施路线

项目按以下四级验收推进，内部仍拆成七个实施阶段：

| 验收级别 | 目标 | 对应阶段 |
|---|---|---|
| A：能运行 | 单层真实维度可初始化并完成 prefill/decode | 0-1 |
| B：层正确 | 逻辑层类型、AttnRes 状态和 backend 正确 | 2-4 |
| C：负载稳定 | shape、routing、rank 和计时可复现 | 5 |
| D：可解释外推 | 多层校准后才能估算完整模型 | 6 |

### 阶段 0：环境和硬件基线

- [ ] 确认 8 张 H20 均可见；
- [ ] 确认 compute capability 为 9.0；
- [ ] 保存 `nvidia-smi topo -m`；
- [ ] 使用 Python 3.12 和 `uv` 创建 `.venv`；
- [ ] 使用 CUDA 13.0 对应的预编译 wheel 安装 editable vLLM；
- [ ] 确认当前导入的 `vllm` 来自本仓库；
- [ ] 确认 PyTorch NCCL 可用；
- [ ] 确认能够读取 `moonshotai/Kimi-K3` 配置；
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
vllm collect-env
nsys --version
```

验收标准：所有基础检查成功，且没有安装或启用不必要的 CPU offload。

### 阶段 1：不修改模型代码的单层启动实验

先使用 vLLM 已有能力验证最小方案：

- `--language-model-only` 排除视觉模型；
- `--load-format dummy` 避免下载完整权重；
- `--hf-overrides` 将 text model 缩短为一层；
- `--enforce-eager` 简化首次调试；
- 降低 `max_model_len`、`max_num_seqs` 和 KV cache 占用。

这一阶段只回答以下问题：

- 一层真实维度的 Kimi-K3 是否能在 8 张 H20 上构造；
- MXFP4、KDA、MLA 和 MoE 分别选择了什么 backend；
- 单层参数、KV/KDA state 和通信 buffer 占用多少显存；
- TP8 和 EP8 是否可以正常初始化。

`--load-format dummy` 只解决“不下载完整 checkpoint”，不能自动证明执行形态
等价于真实 MXFP4 checkpoint。smoke test 还必须逐 rank 验证：

- 权重加载后最终的 storage、shape 和 dtype；
- 实际选择的 MXFP4 MoE backend 和 fallback 原因；
- 每个 rank 的权重显存；
- 首次稳定迭代中实际出现的关键 kernel 名称。

这一阶段的时间不能用于完整模型外推，因为它只代表物理第 0 层。

验收标准：dummy 单层可以完成至少一次 prefill 和一次 decode，所有 rank 正常退出。

### 阶段 2：实现 profiling benchmark 框架

计划在本目录新增：

```text
benchmarks/kimi_k3_layer_profiling/
├── README.md
├── benchmark.py
├── config.py
├── workload.py
├── layer_factory.py
├── runner.py
├── timing.py
├── profiling.py
├── result.py
└── shapes/
    ├── smoke.yaml
    ├── prefill.yaml
    ├── decode.yaml
    └── parallel_sweep.yaml
```

各文件职责：

- `benchmark.py`：命令行入口、分布式启动和实验循环；
- `config.py`：读取 YAML、校验 shape 和并行策略；
- `workload.py`：把实验 shape 转换为 vLLM 可消费的请求和调度输入；
- `layer_factory.py`：构造只含目标层的 `KimiLinearModel`，并注入 benchmark
  专用逻辑层描述；
- `runner.py`：复用真实 vLLM model runner 完成输入准备、forward context、
  cache metadata 和模型 forward，不自行实现简化执行栈；
- `timing.py`：warmup、CUDA event、同步和各 rank latency 汇总；
- `profiling.py`：控制 PyTorch Profiler、Nsys capture range 和 Proton；
- `result.py`：汇总各 rank 数据并写出 JSON/Markdown；
- `shapes/*.yaml`：可复现实验配置。

命令行接口至少应包含：

```text
--model
--logical-layer-index
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

### 阶段 3：支持任意逻辑层

优先保持生产模型的默认行为不变，只增加默认关闭的 profiling 能力。预计需要对
以下文件做窄范围修改：

```text
vllm/models/kimi_k3/nvidia/model.py
```

拟议修改：

1. benchmark 专用 factory 向单层模型传入可选的 `logical_layer_idx`；
2. 未指定时继续从 `prefix` 解析层号，保证正常模型行为不变；
3. profiling 模式下只构造一个物理层，但使用指定逻辑层号判断层类型；
4. 将“目标层身份”和“进入该层前的 AttnRes 状态”作为两个独立输入；
5. cache layer name 保持物理单层命名，避免为不存在的层分配 cache；
6. 不增加正式 serving CLI，生产路径在未传 profiling 配置时完全不变。

benchmark 内部配置示例：

```python
KimiLayerProfilingConfig(
    logical_layer_idx=30,
    incoming_attn_res_blocks=synthetic_blocks,
    incoming_attn_res_prefix_sum=synthetic_prefix_sum,
)
```

该对象属于 benchmark/factory，不写入 Hugging Face model config。逻辑层号负责
选择 KDA/MLA、MoE/dense、block-write 等结构；合成的入站状态负责复现深层
AttnRes 的读取成本。二者不能混为“按层号自动构造一些零张量”。物理 prefix 仍为
`model.layers.0`，以保持 cache 和静态 forward context 的注册一致。

验收标准：能够分别构造 KDA 和 MLA 代表层，且层分类与完整 K3 config 一致。

### 阶段 4：正确性与分布式验证

测试设计问题：

1. 模块用途：在不加载完整模型的情况下复现一个真实 K3 decoder layer；
2. I/O 合约：输入为 packed hidden states、position 和真实 cache metadata，输出为
   hidden states 及相应 cache/state 更新；
3. 防止的故障：逻辑层分类错误、AttnRes 深度错误、cache 名冲突、collective
   shape 不一致；
4. 最低成本测试：tiny config 单层 parity，其次 2 GPU distributed smoke。

计划新增：

```text
tests/models/kimi_k3/test_layer_profiling.py
```

测试至少覆盖：

- [ ] 默认配置下现有 K3 构造行为不变；
- [ ] 指定 KDA 逻辑层时构造正确 attention；
- [ ] 指定 MLA 逻辑层时构造正确 attention；
- [ ] MoE/dense 分类与完整 config 一致；
- [ ] AttnRes block 数与逻辑层位置一致；
- [ ] 合成 AttnRes 入站状态的 shape、dtype、device 和真实小模型一致；
- [ ] 单层模式没有创建其他 decoder layer 参数；
- [ ] tiny config 下，isolated layer 与小型完整模型对应层输出一致；
- [ ] 主 benchmark 确实经过 model runner 输入准备和 forward context；
- [ ] TP/EP 下所有 rank tensor shape 一致。

测试命令必须通过 `.venv/bin/python` 运行：

```bash
.venv/bin/python -m pytest tests/models/kimi_k3/test_layer_profiling.py -v
.venv/bin/python -m pytest tests/models/kimi_k3/test_kda.py -v
.venv/bin/python -m pytest tests/models/kimi_k3/test_attn_res.py -v
.venv/bin/python -m pytest tests/models/kimi_k3/test_sequence_parallel.py -v
```

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

可使用仓库内的 `tools/profiler/nsys_profile_tools/gputrc2graph.py` 将 Nsys GPU
trace 转换为便于分析的图结构；原始 `.nsys-rep` 和导出的数据库仍不提交 Git。

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

### 阶段 6：完整模型估算和校准

按代表层分桶：

```text
T_model_step ≈
    Σ N_kda_bucket  × T_kda_bucket
  + Σ N_mla_bucket  × T_mla_bucket
  + Σ N_dense_bucket × T_dense_bucket
  + T_embedding
  + T_lm_head
  + T_scheduler
  + T_sampling
```

必须避免以下偏差：

- 单层重复运行导致权重长期驻留 L2；
- dummy router 总是选择少量 expert；
- 把所有 kernel duration 简单相加，忽略并发和重叠；
- 忽略 AttnRes 成本随逻辑层深度变化；
- 用 eager 结果预测 CUDA Graph；
- 用纯 prefill 与纯 decode 线性预测 mixed batch；
- 用单层实验预测 pipeline parallel bubble。

每个 shape 至少区分：

- `hot-layer`：同一层连续执行，权重和 metadata 具有较强复用；
- `cache-perturbed-layer`：使用 L2 eviction buffer 或轮换多组权重，作为缓存
  敏感性实验，不宣称它等价于完整模型流式执行；
- `multi-layer-calibrated`：以真实多层运行测得的校准系数修正单层外推；
- `uniform routing`；
- 至少一种 skewed routing。

校准分成两个目的不同的实验：

1. 使用显存允许的最多连续层（目标 4 至 8 层）、真实 hidden size 的 dummy K3，
   校准权重/cache locality、kernel 调度和通信重叠；若不足 4 层，必须记录实际
   层数并扩大外推误差范围；
2. 使用至少跨越一个 AttnRes block 边界的小模型，必要时缩小 hidden size，校准
   block bank 的创建、读取和 block-write 结构。

先由单层数据预测校准模型，再与其真实端到端结果比较；达到预先设定的误差目标
后，才外推完整模型。

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

第一版固定使用以下 canonical shapes，先建立可复现基线，再扩展不等长、mixed 和
speculative workload：

| ID | Phase | B | 每请求 Q | 每请求 K | M |
|---|---|---:|---:|---:|---:|
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
|---:|---:|---:|---|---|
| 8 | 1 | 8 | TP8 | EP8 |
| 4 | 2 | 8 | 每个 DP group 内 TP4 | EP8 |
| 2 | 4 | 8 | 每个 DP group 内 TP2 | EP8 |
| 1 | 8 | 8 | attention 按 DP 复制 | EP8 |

第一轮只运行 `TP=8, DP=1, EP=8, DCP=1, PP=1`，先稳定 execution stack、
shape 和采集流程；通过验收后再展开矩阵。

第一轮固定 `PP=1`。单层实验不能测量真实 pipeline bubble。

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

逻辑层采样至少覆盖：第一个有效层、AttnRes 周期 `R` 附近、`4R` 附近和最后
一层，并保证 KDA/MLA、MoE/dense 与 block-write 分类均被覆盖。最终分桶键至少为：

```text
(attention_type, ffn_type, attn_res_depth, block_write)
```

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

建议服务器结果目录：

```text
profile_outputs/
└── <git-commit>/
    └── <experiment-name>/
        ├── manifest.json
        ├── rank_0.json
        ├── rank_1.json
        ├── ...
        ├── summary.json
        ├── summary.md
        └── traces/
```

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
git_commit: null
gpu_name: null
gpu_count: 8
cuda_version: null
torch_version: null
vllm_version: null
nccl_version: null
dtype: bfloat16
weight_format: mxfp4
selected_backends: {}
weight_storage_by_rank: []
kernel_names: []
physical_layer_index: 0
logical_layer_index: null
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
warmup_iters: null
repeat_iters: null
random_seed: 0
latency_ms_by_rank: []
distributed_latency_ms: null
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
- [ ] kernel JIT、autotune 和 graph capture 不计入正式时间。

运行后：

- [ ] 八个 rank 均正常结束；
- [ ] 使用最慢 rank 计算 distributed latency；
- [ ] 记录峰值显存；
- [ ] 记录异常值和失败次数；
- [ ] trace 只包含少量稳定 iteration；
- [ ] summary 中明确 hot-layer/cache-perturbed/multi-layer-calibrated、routing
  和 eager/graph；
- [ ] 结果目录包含完整 manifest。

## 12. 当前里程碑

当前状态：

- [x] 创建 GitHub fork；
- [x] 创建 `codex/kimi-k3-layer-profiling` 分支；
- [x] H20 服务器成功拉取工作分支；
- [x] 确认 8 张 H20 位于完整 NVLink 域；
- [x] 确认服务器 CUDA Toolkit 为 13.0；
- [ ] 完成 Python 3.12、uv 和 editable vLLM 环境安装；
- [ ] 保存环境基线；
- [ ] 完成不修改模型代码的单层 smoke test；
- [ ] 确认 H20 上 K3/MXFP4 实际 backend；
- [ ] 实现 benchmark 配置与结果框架；
- [ ] 实现 logical layer profiling；
- [ ] 完成正确性测试；
- [ ] 完成 TP8/EP8 首轮数据；
- [ ] 完成显存允许的最大连续多层校准（目标 4 至 8 层）；
- [ ] 形成完整 profiling 报告。

## 13. 工作原则

1. 每次只推进一个可验证的小阶段；
2. 先正确性和可解释性，再追求性能；
3. benchmark 数字和 profiler trace 分开采集；
4. 每个结论都能追溯到 commit、配置、硬件和原始结果；
5. 任何 backend fallback 都必须显式记录；
6. 在外推完整模型前，必须先通过多层小模型校准。

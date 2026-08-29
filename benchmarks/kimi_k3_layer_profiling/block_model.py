# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiLinearForCausalLM,
    maybe_init_gemm_rs_ar,
)


BLOCK_LAYER_COUNT = 12


class KimiK3FirstBlock(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if config.attn_res_block_size != BLOCK_LAYER_COUNT:
            raise ValueError(
                "KimiK3BlockProfiler requires attn_res_block_size=12"
            )
        if config.num_hidden_layers < BLOCK_LAYER_COUNT:
            raise ValueError("Kimi-K3 config does not contain a complete first block")

        parallel_config = vllm_config.parallel_config
        use_mega_moe = (
            vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        )
        self.use_sequence_parallel = (
            parallel_config.pipeline_parallel_size == 1
            and parallel_config.enable_expert_parallel
            and parallel_config.tensor_parallel_size > 1
            and (use_mega_moe or parallel_config.data_parallel_size > 1)
        )
        self.config = config
        self.start_layer = 0
        self.end_layer = BLOCK_LAYER_COUNT
        self.run_gemm_rs_ar = maybe_init_gemm_rs_ar(
            vllm_config, self.use_sequence_parallel
        )

        aux_stream = torch.cuda.Stream()
        self.layers = nn.ModuleList(
            [
                KimiDecoderLayer(
                    config,
                    vllm_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                    aux_stream=aux_stream,
                    run_gemm_rs_ar=self.run_gemm_rs_ar,
                )
                for layer_idx in range(BLOCK_LAYER_COUNT)
            ]
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("Kimi-K3 block forward is not implemented yet")


class KimiK3BlockProfiler(KimiLinearForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        model_prefix = f"{prefix}.model" if prefix else "model"
        self.model = KimiK3FirstBlock(
            vllm_config=vllm_config,
            prefix=model_prefix,
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("Kimi-K3 block forward is not implemented yet")

    def compute_logits(self, *args, **kwargs):
        raise RuntimeError("Kimi-K3 block profiling must not compute logits")

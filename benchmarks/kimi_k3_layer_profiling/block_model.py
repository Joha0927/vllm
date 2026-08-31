# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from typing_extensions import NamedTuple

from vllm import envs
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.models.common.ops.sequence_parallel import sp_padding_mask, sp_shard
from vllm.models.kimi_k3.nvidia.model import (
    KimiDecoderLayer,
    KimiLinearForCausalLM,
    maybe_init_gemm_rs_ar,
)

BLOCK_LAYER_COUNT = 12


class BlockForwardOutput(NamedTuple):
    pending_hidden_states: torch.Tensor
    prefix_sum: torch.Tensor
    block_residual_bank: torch.Tensor


class KimiK3FirstBlock(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        if config.attn_res_block_size != BLOCK_LAYER_COUNT:
            raise ValueError("KimiK3BlockProfiler requires attn_res_block_size=12")
        if config.num_hidden_layers < BLOCK_LAYER_COUNT:
            raise ValueError("Kimi-K3 config does not contain a complete first block")

        parallel_config = vllm_config.parallel_config
        use_mega_moe = vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
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

        aux_stream = torch.Stream()
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

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> BlockForwardOutput:
        if inputs_embeds.ndim != 2:
            raise ValueError("inputs_embeds must have shape [tokens, hidden_size]")
        if inputs_embeds.shape[0] != positions.shape[0]:
            raise ValueError("positions and inputs_embeds must have equal token counts")
        if inputs_embeds.shape[1] != self.config.hidden_size:
            raise ValueError(
                f"Expected hidden_size={self.config.hidden_size}, got "
                f"{inputs_embeds.shape[1]}"
            )

        hidden_states = inputs_embeds
        if self.use_sequence_parallel:
            if envs.VLLM_MOE_SKIP_PADDING and is_forward_context_available():
                forward_context = get_forward_context()
                forward_context.is_padding = sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = sp_shard(hidden_states)

        prefix_sum = hidden_states
        pending_hidden_states = None
        block_residual_bank = hidden_states.new_empty(
            hidden_states.shape[0],
            1,
            hidden_states.shape[1],
        )
        for layer in self.layers:
            pending_hidden_states, prefix_sum, block_residual_bank = layer(
                positions=positions,
                hidden_states=pending_hidden_states,
                prefix_sum=prefix_sum,
                residual=block_residual_bank,
            )

        assert pending_hidden_states is not None
        assert prefix_sum is not None
        return BlockForwardOutput(
            pending_hidden_states=pending_hidden_states,
            prefix_sum=prefix_sum,
            block_residual_bank=block_residual_bank,
        )


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

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BlockForwardOutput:
        if input_ids is not None:
            raise ValueError("Block profiling requires inputs_embeds, not input_ids")
        if inputs_embeds is None:
            raise ValueError("Block profiling requires inputs_embeds")
        if intermediate_tensors is not None:
            raise ValueError("Pipeline-parallel intermediate tensors are unsupported")
        return self.model(inputs_embeds=inputs_embeds, positions=positions)

    def compute_logits(self, *args, **kwargs):
        raise RuntimeError("Kimi-K3 block profiling must not compute logits")

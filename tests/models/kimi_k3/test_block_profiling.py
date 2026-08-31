# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from benchmarks.kimi_k3_layer_profiling import block_model


class _FakeDecoderLayer(
    torch.nn.Module if torch is not None else object  # type: ignore[misc]
):
    def __init__(
        self,
        config,
        vllm_config,
        prefix,
        aux_stream,
        run_gemm_rs_ar,
    ) -> None:
        if torch is not None:
            super().__init__()
        self.prefix = prefix
        self.layer_idx = int(prefix.rsplit(".", 1)[1])

    def forward(
        self,
        *,
        positions,
        hidden_states,
        prefix_sum,
        residual,
    ):
        del positions
        if hidden_states is None:
            hidden_states = prefix_sum
        return hidden_states + 1, prefix_sum + 1, residual


def _vllm_config() -> SimpleNamespace:
    text_config = SimpleNamespace(
        attn_res_block_size=12,
        hidden_size=16,
        num_hidden_layers=93,
    )
    return SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="auto"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(),
            hf_text_config=text_config,
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            tensor_parallel_size=8,
        ),
        quant_config=None,
    )


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_first_block_constructs_only_layers_zero_through_eleven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(block_model, "KimiDecoderLayer", _FakeDecoderLayer)
    monkeypatch.setattr(block_model, "maybe_init_gemm_rs_ar", lambda *_: False)
    monkeypatch.setattr(block_model.torch, "Stream", lambda: object())

    model = block_model.KimiK3BlockProfiler(vllm_config=_vllm_config())

    assert len(model.model.layers) == 12
    assert [layer.prefix for layer in model.model.layers] == [
        f"model.layers.{layer_idx}" for layer_idx in range(12)
    ]
    parameter_names = [name for name, _ in model.named_parameters()]
    assert not any("embed_tokens" in name for name in parameter_names)
    assert not any("lm_head" in name for name in parameter_names)
    assert not any("vision_tower" in name for name in parameter_names)
    assert not hasattr(model.model, "norm")
    assert not hasattr(model.model, "output_attn_res_norm")
    assert not hasattr(model.model, "output_attn_res_proj")


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_forward_returns_the_real_block_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(block_model, "KimiDecoderLayer", _FakeDecoderLayer)
    monkeypatch.setattr(block_model, "maybe_init_gemm_rs_ar", lambda *_: False)
    monkeypatch.setattr(block_model.torch, "Stream", lambda: object())
    model = block_model.KimiK3BlockProfiler(vllm_config=_vllm_config())

    inputs_embeds = torch.zeros(8, 16)
    positions = torch.arange(8)
    output = model(
        input_ids=None,
        positions=positions,
        inputs_embeds=inputs_embeds,
    )

    assert output.pending_hidden_states.shape == (8, 16)
    assert output.prefix_sum.shape == (8, 16)
    assert output.block_residual_bank.shape == (8, 1, 16)
    assert torch.equal(output.pending_hidden_states, torch.full((8, 16), 12.0))
    assert torch.equal(output.prefix_sum, torch.full((8, 16), 12.0))


@pytest.mark.skipif(torch is None, reason="requires torch")
def test_unsupported_execution_paths_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(block_model, "KimiDecoderLayer", _FakeDecoderLayer)
    monkeypatch.setattr(block_model, "maybe_init_gemm_rs_ar", lambda *_: False)
    monkeypatch.setattr(block_model.torch, "Stream", lambda: object())
    model = block_model.KimiK3BlockProfiler(vllm_config=_vllm_config())
    positions = torch.arange(1)
    inputs_embeds = torch.zeros(1, 16)

    with pytest.raises(ValueError, match="requires inputs_embeds"):
        model(input_ids=None, positions=positions)
    with pytest.raises(ValueError, match="not input_ids"):
        model(input_ids=torch.zeros(1), positions=positions)
    with pytest.raises(ValueError, match="intermediate tensors"):
        model(
            input_ids=None,
            positions=positions,
            intermediate_tensors=object(),
            inputs_embeds=inputs_embeds,
        )
    with pytest.raises(RuntimeError, match="must not compute logits"):
        model.compute_logits(None)

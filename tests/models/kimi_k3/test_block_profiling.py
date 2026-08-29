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


class _FakeDecoderLayer(torch.nn.Module if torch is not None else object):
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


def _vllm_config() -> SimpleNamespace:
    text_config = SimpleNamespace(
        attn_res_block_size=12,
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
    monkeypatch.setattr(block_model.torch.cuda, "Stream", lambda: object())

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
def test_unimplemented_execution_paths_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(block_model, "KimiDecoderLayer", _FakeDecoderLayer)
    monkeypatch.setattr(block_model, "maybe_init_gemm_rs_ar", lambda *_: False)
    monkeypatch.setattr(block_model.torch.cuda, "Stream", lambda: object())
    model = block_model.KimiK3BlockProfiler(vllm_config=_vllm_config())

    with pytest.raises(RuntimeError, match="forward is not implemented"):
        model()
    with pytest.raises(RuntimeError, match="must not compute logits"):
        model.compute_logits(None)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.kernels.benchmark_kimi_k3_mla_attn_res_ncu import (
    NVTX_RANGES,
    TARGETS,
    parse_args,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def test_ncu_targets_have_unique_ranges() -> None:
    assert set(TARGETS) == {
        "mla-kv-insert",
        "attn-res-prefill",
        "attn-res-decode",
        "attn-res-block-write",
        "mla-fa-prefill",
    }
    assert set(NVTX_RANGES) == set(TARGETS)
    assert len(set(NVTX_RANGES.values())) == len(TARGETS)


@pytest.mark.parametrize("target", TARGETS)
def test_ncu_target_defaults_to_one_stateful_profile_call(target: str) -> None:
    args = parse_args(["--target", target])

    assert args.target == target
    assert args.warmup_iters == 3
    assert args.profile_iters == 1


def test_ncu_target_rejects_multiple_profile_calls() -> None:
    with pytest.raises(ValueError, match="cache state is mutable"):
        parse_args(["--target", "mla-kv-insert", "--profile-iters", "2"])

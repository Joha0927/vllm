# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from benchmarks.kernels.benchmark_kimi_k3_kda_ncu import NVTX_RANGES, parse_args


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize("target", sorted(NVTX_RANGES))
def test_ncu_target_has_one_profile_iteration(target: str) -> None:
    args = parse_args(["--target", target])

    assert args.target == target
    assert args.warmup_iters == 3
    assert args.profile_iters == 1


def test_ncu_target_rejects_multiple_stateful_iterations() -> None:
    with pytest.raises(ValueError, match="KDA state is mutable"):
        parse_args(["--target", "decode-fused", "--profile-iters", "2"])

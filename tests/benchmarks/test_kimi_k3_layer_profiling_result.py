# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from benchmarks.kimi_k3_layer_profiling.config import dry_run, load_yaml
from benchmarks.kimi_k3_layer_profiling.result import (
    RankRecord,
    aggregate_results,
    manifest_digest,
    write_results,
)


ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml"


def _manifest() -> dict:
    return dry_run(load_yaml(SMOKE_CONFIG)).manifest()


def _passing_records(manifest: dict) -> list[RankRecord]:
    digest = manifest_digest(manifest)
    return [
        RankRecord(
            rank=rank,
            world_size=8,
            manifest_digest=digest,
            status="PASS",
            latency_samples_ms=(1.0 + rank / 10, 2.0 + rank / 10, 3.0 + rank / 10),
        )
        for rank in range(8)
    ]


def test_distributed_latency_uses_per_iteration_slowest_rank() -> None:
    manifest = _manifest()
    summary = aggregate_results(manifest, _passing_records(manifest))

    assert summary["status"] == "PASS"
    assert summary["per_iteration_max_ms"] == pytest.approx([1.7, 2.7, 3.7])
    assert summary["distributed_latency_ms"] == pytest.approx(2.7)
    assert summary["latency_ms"] == pytest.approx(
        {"min": 1.7, "p10": 1.9, "p50": 2.7, "p90": 3.5, "max": 3.7}
    )


def test_write_results_is_deterministic(tmp_path: Path) -> None:
    manifest = _manifest()
    records = list(reversed(_passing_records(manifest)))
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_results(first, manifest, records)
    write_results(second, manifest, records)

    expected_names = {
        "manifest.json",
        "summary.json",
        "summary.md",
        *(f"rank_{rank}.json" for rank in range(8)),
    }
    assert {path.name for path in first.iterdir()} == expected_names
    for name in expected_names:
        assert (first / name).read_bytes() == (second / name).read_bytes()


@pytest.mark.parametrize("failure", ["missing", "duplicate", "manifest"])
def test_invalid_rank_sets_fail_closed(failure: str) -> None:
    manifest = _manifest()
    records = _passing_records(manifest)
    if failure == "missing":
        records.pop()
    elif failure == "duplicate":
        records[-1] = records[0]
    else:
        records[-1] = RankRecord(
            rank=7,
            world_size=8,
            manifest_digest="different",
            status="PASS",
            latency_samples_ms=(1.0, 2.0, 3.0),
        )

    with pytest.raises(ValueError):
        aggregate_results(manifest, records)


def test_failed_rank_preserves_failure_evidence(tmp_path: Path) -> None:
    manifest = _manifest()
    records = _passing_records(manifest)
    records[3] = RankRecord(
        rank=3,
        world_size=8,
        manifest_digest=manifest_digest(manifest),
        status="FAIL",
        failure_stage="model_forward",
        exception_type="RuntimeError",
        error_message="synthetic failure",
    )
    records[4] = RankRecord(
        rank=4,
        world_size=8,
        manifest_digest=manifest_digest(manifest),
        status="PASS",
        latency_samples_ms=(1.0,),
    )

    summary = write_results(tmp_path, manifest, records)

    assert summary["status"] == "FAIL"
    assert summary["failed_ranks"] == [3]
    assert summary["distributed_latency_ms"] is None
    assert summary["failures"][0]["failure_stage"] == "model_forward"
    assert "synthetic failure" in (tmp_path / "summary.md").read_text()


def test_mismatched_sample_counts_fail_closed() -> None:
    manifest = _manifest()
    records = _passing_records(manifest)
    records[0] = RankRecord(
        rank=0,
        world_size=8,
        manifest_digest=manifest_digest(manifest),
        status="PASS",
        latency_samples_ms=(1.0,),
    )

    with pytest.raises(ValueError, match="same latency sample count"):
        aggregate_results(manifest, records)

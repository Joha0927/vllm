# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class RankRecord:
    rank: int
    world_size: int
    manifest_digest: str
    status: str
    latency_samples_ms: tuple[float, ...] = ()
    failure_stage: str | None = None
    exception_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["latency_samples_ms"] = list(self.latency_samples_ms)
        return data


def manifest_digest(manifest: dict[str, Any]) -> str:
    payload = json.dumps(
        manifest, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _validate_records(
    manifest: dict[str, Any], records: Sequence[RankRecord]
) -> list[RankRecord]:
    expected_world_size = int(manifest["gpu_count"])
    if len(records) != expected_world_size:
        raise ValueError(
            f"Expected {expected_world_size} rank records, got {len(records)}"
        )

    ordered = sorted(records, key=lambda record: record.rank)
    ranks = [record.rank for record in ordered]
    expected_ranks = list(range(expected_world_size))
    if ranks != expected_ranks:
        raise ValueError(
            f"Rank records must cover {expected_ranks}, got {ranks}"
        )

    expected_digest = manifest_digest(manifest)
    for record in ordered:
        if record.world_size != expected_world_size:
            raise ValueError(
                f"Rank {record.rank} has world_size={record.world_size}, "
                f"expected {expected_world_size}"
            )
        if record.manifest_digest != expected_digest:
            raise ValueError(f"Rank {record.rank} used a different manifest")
        if record.status not in {"PASS", "FAIL"}:
            raise ValueError(f"Rank {record.rank} has invalid status {record.status!r}")
        if record.status == "FAIL":
            if not record.failure_stage or not record.exception_type:
                raise ValueError(
                    f"Failed rank {record.rank} must record failure_stage "
                    "and exception_type"
                )
            continue
        if not record.latency_samples_ms:
            raise ValueError(f"Passing rank {record.rank} has no latency samples")
        if any(
            not math.isfinite(value) or value < 0
            for value in record.latency_samples_ms
        ):
            raise ValueError(
                f"Rank {record.rank} latency samples must be finite and non-negative"
            )

    passing = [record for record in ordered if record.status == "PASS"]
    sample_counts = {len(record.latency_samples_ms) for record in passing}
    has_failures = any(record.status == "FAIL" for record in ordered)
    if not has_failures and len(sample_counts) > 1:
        raise ValueError("Passing ranks must have the same latency sample count")
    return ordered


def aggregate_results(
    manifest: dict[str, Any], records: Sequence[RankRecord]
) -> dict[str, Any]:
    ordered = _validate_records(manifest, records)
    failures = [
        {
            "error_message": record.error_message,
            "exception_type": record.exception_type,
            "failure_stage": record.failure_stage,
            "rank": record.rank,
        }
        for record in ordered
        if record.status == "FAIL"
    ]
    summary: dict[str, Any] = {
        "distributed_latency_ms": None,
        "failed_ranks": [failure["rank"] for failure in failures],
        "failures": failures,
        "git_commit": manifest.get("git_commit"),
        "latency_aggregation": "median_of_per_iteration_rank_max",
        "latency_ms": None,
        "per_iteration_max_ms": [],
        "rank_count": len(ordered),
        "sample_count": 0,
        "status": "FAIL" if failures else "PASS",
        "world_size": int(manifest["gpu_count"]),
    }
    if failures:
        return summary

    per_iteration_max = [
        max(record.latency_samples_ms[index] for record in ordered)
        for index in range(len(ordered[0].latency_samples_ms))
    ]
    summary.update(
        {
            "distributed_latency_ms": statistics.median(per_iteration_max),
            "latency_ms": {
                "max": max(per_iteration_max),
                "min": min(per_iteration_max),
                "p10": _percentile(per_iteration_max, 0.10),
                "p50": _percentile(per_iteration_max, 0.50),
                "p90": _percentile(per_iteration_max, 0.90),
            },
            "per_iteration_max_ms": per_iteration_max,
            "sample_count": len(per_iteration_max),
        }
    )
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Kimi-K3 Block Profiling Summary",
        "",
        f"- Status: {summary['status']}",
        f"- Git commit: {summary['git_commit'] or 'not recorded'}",
        f"- World size: {summary['world_size']}",
        f"- Rank records: {summary['rank_count']}",
        f"- Aggregation: {summary['latency_aggregation']}",
    ]
    if summary["status"] == "PASS":
        lines.extend(
            [
                f"- Samples: {summary['sample_count']}",
                "- Distributed latency (ms): "
                f"{summary['distributed_latency_ms']}",
            ]
        )
    else:
        lines.extend(["", "## Failures", ""])
        for failure in summary["failures"]:
            lines.append(
                f"- Rank {failure['rank']}: {failure['failure_stage']} / "
                f"{failure['exception_type']}: "
                f"{failure['error_message'] or 'no message'}"
            )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_results(
    output_dir: str | Path,
    manifest: dict[str, Any],
    records: Sequence[RankRecord],
) -> dict[str, Any]:
    ordered = _validate_records(manifest, records)
    summary = aggregate_results(manifest, ordered)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    _write_json(destination / "manifest.json", manifest)
    for record in ordered:
        _write_json(destination / f"rank_{record.rank}.json", record.to_dict())
    _write_json(destination / "summary.json", summary)
    (destination / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    return summary

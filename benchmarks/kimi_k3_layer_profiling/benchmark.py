# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import (
    apply_overrides,
    dry_run,
    load_yaml,
)

_OVERRIDE_FIELDS = (
    "phase",
    "batch_size",
    "query_len",
    "context_len",
    "num_layers",
    "tensor_parallel_size",
    "data_parallel_size",
    "decode_context_parallel_size",
    "enable_expert_parallel",
    "all2all_backend",
    "execution_mode",
    "routing_strategy",
    "cache_mode",
    "warmup_iters",
    "profile_iters",
    "repeat_iters",
    "profile",
    "gpu_count",
    "random_seed",
    "diagnostic_partial_block",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile the first Kimi-K3 block")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--distributed-smoke",
        action="store_true",
        help="Initialize and validate distributed groups without loading a model",
    )
    parser.add_argument(
        "--model-construction-smoke",
        action="store_true",
        help="Construct the real 12-layer block without running forward",
    )
    parser.add_argument(
        "--forward-smoke",
        action="store_true",
        help="Run one untimed eager forward of the real 12-layer block",
    )
    parser.add_argument(
        "--production-profile",
        action="store_true",
        help="Profile the production EngineCore path with 12 real Kimi layers",
    )
    parser.add_argument("--list-layer-types", action="store_true")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--phase", choices=("prefill", "decode"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--query-len", type=int)
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--decode-context-parallel-size", type=int)
    parser.add_argument(
        "--enable-expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--all2all-backend")
    parser.add_argument("--execution-mode", choices=("eager", "cudagraph"))
    parser.add_argument("--routing-strategy")
    parser.add_argument("--cache-mode")
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--profile-iters", type=int)
    parser.add_argument("--repeat-iters", type=int)
    parser.add_argument("--profile", choices=("none", "torch", "cuda", "proton"))
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument(
        "--diagnostic-partial-block",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {field: getattr(args, field) for field in _OVERRIDE_FIELDS}


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data: dict[str, Any] = load_yaml(args.config)
    result = dry_run(apply_overrides(data, _overrides_from_args(args)))
    selected_modes = sum(
        (
            args.dry_run,
            args.distributed_smoke,
            args.list_layer_types,
            args.model_construction_smoke,
            args.forward_smoke,
            args.production_profile,
        )
    )
    if selected_modes != 1:
        raise SystemExit("Select exactly one execution mode")
    if (
        args.distributed_smoke
        or args.model_construction_smoke
        or args.forward_smoke
        or args.production_profile
    ) and args.manifest_out is not None:
        raise SystemExit("--manifest-out is not supported with GPU smoke modes")
    if args.manifest_out is not None:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(
            json.dumps(result.manifest(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.list_layer_types:
        print(json.dumps([asdict(layer) for layer in result.layers], indent=2))
        return 0
    if args.distributed_smoke:
        from benchmarks.kimi_k3_layer_profiling.distributed import (
            run_distributed_smoke,
        )

        run_distributed_smoke(result.config)
        return 0
    if args.model_construction_smoke:
        from benchmarks.kimi_k3_layer_profiling.model_construction import (
            run_model_construction_smoke,
        )

        run_model_construction_smoke(result.config)
        return 0
    if args.forward_smoke:
        from benchmarks.kimi_k3_layer_profiling.forward_smoke import (
            run_forward_smoke,
        )

        run_forward_smoke(result.config)
        return 0
    if args.production_profile:
        from benchmarks.kimi_k3_layer_profiling.production_profile import (
            run_production_profile,
        )

        run_production_profile(result.config)
        return 0
    print(result.to_json())
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()

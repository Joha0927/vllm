# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any


class KimiK3ProfilingWorkerExtension:
    """Opt-in worker instrumentation for controlled profiling ablations."""

    def install_tp_all_gather_sync_ablation(self) -> dict[str, Any]:
        import torch

        from vllm.distributed import get_tp_group
        from vllm.models.kimi_k3.nvidia import model as kimi_model

        if getattr(kimi_model, "_profiling_original_sp_all_gather", None) is not None:
            return {"installed": True, "already_installed": True}

        original = kimi_model.sp_all_gather
        self._tp_all_gather_events: list[
            tuple[torch.Event, torch.Event, tuple[int, ...]]
        ] = []

        def synchronized_sp_all_gather(x: torch.Tensor) -> torch.Tensor:
            with torch.profiler.record_function("kimi_k3.tp_sync.cuda_drain"):
                torch.accelerator.synchronize()
            with torch.profiler.record_function("kimi_k3.tp_sync.host_barrier"):
                get_tp_group().barrier()
            with torch.profiler.record_function("kimi_k3.tp_collective.sp_all_gather"):
                start = torch.Event(enable_timing=True)
                end = torch.Event(enable_timing=True)
                start.record()
                output = original(x)
                end.record()
                self._tp_all_gather_events.append((start, end, tuple(x.shape)))
                return output

        kimi_model._profiling_original_sp_all_gather = original
        kimi_model.sp_all_gather = synchronized_sp_all_gather
        return {"installed": True, "already_installed": False}

    def reset_tp_all_gather_sync_measurements(self) -> None:
        import torch

        torch.accelerator.synchronize()
        self._tp_all_gather_events.clear()

    def collect_tp_all_gather_sync_measurements(self) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        from vllm.distributed import get_tp_group

        torch.accelerator.synchronize()
        calls = [
            {
                "call_index": index,
                "elapsed_ms": start.elapsed_time(end),
                "input_shape": list(shape),
            }
            for index, (start, end, shape) in enumerate(self._tp_all_gather_events)
        ]
        return {
            "global_rank": dist.get_rank() if dist.is_initialized() else None,
            "tensor_parallel_rank": get_tp_group().rank_in_group,
            "calls": calls,
        }

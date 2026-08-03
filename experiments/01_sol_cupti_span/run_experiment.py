#!/usr/bin/env python3
"""Compare official SOL CUPTI span timing with activity sums and CUDA Events.

This script imports NVIDIA SOL-ExecBench's timing implementation from a fixed
checkout supplied through SOL_EXECBENCH_SRC. It refuses to run unless physical
GPU 1 is selected through CUDA_VISIBLE_DEVICES=1.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


SOL_COMMIT = "a9fa0804c793d438e70850c33fe34426e66d53dd"


def require_gpu1() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise RuntimeError(
            "This experiment must run on physical GPU 1: "
            "set CUDA_VISIBLE_DEVICES=1 (got %r)" % visible
        )


def load_sol_modules():
    source_root = os.environ.get("SOL_EXECBENCH_SRC")
    if not source_root:
        raise RuntimeError(
            "SOL_EXECBENCH_SRC must point to the src/ directory of SOL-ExecBench "
            f"commit {SOL_COMMIT}"
        )
    source_path = Path(source_root).resolve()
    repo_path = source_path.parent
    if not (source_path / "sol_execbench/core/bench/timing.py").is_file():
        raise RuntimeError(f"Invalid SOL_EXECBENCH_SRC: {source_path}")
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot resolve SOL checkout commit: {repo_path}") from exc
    if head != SOL_COMMIT:
        raise RuntimeError(f"Expected SOL commit {SOL_COMMIT}, got {head}")
    sys.path.insert(0, str(source_path))

    import torch
    from sol_execbench.core.bench import cupti_utils
    from sol_execbench.core.bench import timing

    return torch, timing, cupti_utils, head


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty sample")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    mean = statistics.mean(ordered)
    stdev = statistics.pstdev(ordered)
    return {
        "min_us": ordered[0] * 1000,
        "p10_us": percentile(0.10) * 1000,
        "median_us": statistics.median(ordered) * 1000,
        "mean_us": mean * 1000,
        "p90_us": percentile(0.90) * 1000,
        "max_us": ordered[-1] * 1000,
        "cv": stdev / mean if mean else 0.0,
    }


def activity_record(activity) -> dict[str, Any]:
    return {
        "name": activity.name,
        "identity": activity.kernel_string(),
        "kind": str(activity.kind),
        "start_ns": activity.start,
        "end_ns": activity.end,
        "duration_us": (activity.end - activity.start) / 1000,
        "correlation_id": activity.correlation_id,
        "bytes": activity.bytes,
        "copy_kind": activity.copy_kind,
        "value": activity.value,
    }


def run_sol_with_trace(
    timing,
    cupti_utils,
    fn: Callable[[], Any],
    *,
    warmup: int,
    rep: int,
) -> dict[str, Any]:
    """Run official SOL timing while retaining its discovery/timing buffers."""

    original_collect = timing.collect_cupti_activities
    original_timestamp = timing.cupti.get_timestamp
    captures = []
    timestamps: list[float] = []

    @contextlib.contextmanager
    def capturing_collect(*args, **kwargs) -> Iterator[Any]:
        with original_collect(*args, **kwargs) as buffers:
            yield buffers
        captures.append(buffers)

    def capturing_timestamp() -> float:
        timestamp = original_timestamp()
        timestamps.append(timestamp)
        return timestamp

    timing.collect_cupti_activities = capturing_collect
    timing.cupti.get_timestamp = capturing_timestamp
    try:
        sol_times_ms = timing.bench_gpu_time_with_cupti(
            fn,
            warmup=warmup,
            rep=rep,
            cold_l2_cache=True,
            device="cuda:0",
        )
    finally:
        timing.collect_cupti_activities = original_collect
        timing.cupti.get_timestamp = original_timestamp

    if len(captures) != 2:
        raise RuntimeError(f"Expected discovery and timing captures, got {len(captures)}")
    if len(timestamps) != rep * 2:
        raise RuntimeError(f"Expected {rep * 2} timestamps, got {len(timestamps)}")

    discovery = sorted(
        captures[0].kernels,
        key=lambda item: (item.start, item.end, item.correlation_id),
    )
    timed = sorted(
        captures[1].kernels,
        key=lambda item: (item.start, item.end, item.correlation_id),
    )
    expected = cupti_utils.kernel_activity_sequence(discovery)
    starts = [item.start for item in timed]
    windows = []
    for iteration in range(rep):
        start_cpu = timestamps[2 * iteration]
        end_cpu = timestamps[2 * iteration + 1]
        left = bisect.bisect_left(starts, start_cpu)
        right = bisect.bisect_right(starts, end_cpu)
        all_activities = timed[left:right]
        selected = cupti_utils.select_activity_sequence(
            all_activities, expected, iteration=iteration
        )
        all_sum_us = sum(item.end - item.start for item in all_activities) / 1000
        all_span_us = (
            (max(item.end for item in all_activities) - min(item.start for item in all_activities))
            / 1000
            if all_activities
            else 0.0
        )
        selected_sum_us = sum(item.end - item.start for item in selected) / 1000
        selected_span_us = (
            max(item.end for item in selected) - min(item.start for item in selected)
        ) / 1000
        windows.append(
            {
                "iteration": iteration,
                "cpu_window_ns": [start_cpu, end_cpu],
                "all_activity_count": len(all_activities),
                "selected_activity_count": len(selected),
                "all_activity_sum_us": all_sum_us,
                "all_activity_span_us": all_span_us,
                "selected_activity_sum_us": selected_sum_us,
                "selected_activity_span_us": selected_span_us,
                "all_activities": [activity_record(item) for item in all_activities],
                "selected_activities": [activity_record(item) for item in selected],
            }
        )

    return {
        "sol_times_ms": sol_times_ms,
        "sol_summary": summarize(sol_times_ms),
        "discovery_activities": [activity_record(item) for item in discovery],
        "windows": windows,
        "selected_sum_summary": summarize(
            [window["selected_activity_sum_us"] / 1000 for window in windows]
        ),
        "all_sum_summary": summarize(
            [window["all_activity_sum_us"] / 1000 for window in windows]
        ),
        "all_span_summary": summarize(
            [window["all_activity_span_us"] / 1000 for window in windows]
        ),
    }


def add_factory(torch, elements: int) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        source = torch.randn(elements, device="cuda")
        output = torch.empty_like(source)

        def run() -> None:
            torch.add(source, 1.0, out=output)

        return run

    return factory


def sequential_factory(torch, elements: int) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        source_a = torch.randn(elements, device="cuda")
        source_b = torch.randn(elements, device="cuda")
        output_a = torch.empty_like(source_a)
        output_b = torch.empty_like(source_b)

        def run() -> None:
            torch.add(source_a, 1.0, out=output_a)
            torch.add(source_b, 1.0, out=output_b)

        return run

    return factory


def host_gap_factory(
    torch, elements: int, gap_seconds: float
) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        source_a = torch.randn(elements, device="cuda")
        source_b = torch.randn(elements, device="cuda")
        output_a = torch.empty_like(source_a)
        output_b = torch.empty_like(source_b)

        def run() -> None:
            torch.add(source_a, 1.0, out=output_a)
            time.sleep(gap_seconds)
            torch.add(source_b, 1.0, out=output_b)

        return run

    return factory


def concurrent_sleep_factory(torch, cycles: int) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        def run() -> None:
            with torch.cuda.stream(stream_a):
                torch.cuda._sleep(cycles)
            with torch.cuda.stream(stream_b):
                torch.cuda._sleep(cycles)

        return run

    return factory


def dynamic_extra_factory(
    torch, elements: int, discovery_after_calls: int
) -> Callable[[], Callable[[], None]]:
    """Discovery sees add only; timed calls additionally launch sin."""

    def factory() -> Callable[[], None]:
        source = torch.randn(elements, device="cuda")
        add_output = torch.empty_like(source)
        sin_output = torch.empty_like(source)
        calls = 0

        def run() -> None:
            nonlocal calls
            calls += 1
            torch.add(source, 1.0, out=add_output)
            if calls > discovery_after_calls:
                torch.sin(source, out=sin_output)

        return run

    return factory


def dynamic_full_event_factory(torch, elements: int) -> Callable[[], Callable[[], None]]:
    def factory() -> Callable[[], None]:
        source = torch.randn(elements, device="cuda")
        add_output = torch.empty_like(source)
        sin_output = torch.empty_like(source)

        def run() -> None:
            torch.add(source, 1.0, out=add_output)
            torch.sin(source, out=sin_output)

        return run

    return factory


def run_case(
    torch,
    timing,
    cupti_utils,
    *,
    name: str,
    sol_factory: Callable[[], Callable[[], None]],
    event_factory: Callable[[], Callable[[], None]],
    warmup: int,
    rep: int,
    purpose: str,
) -> dict[str, Any]:
    event_fn = event_factory()
    torch.cuda.synchronize()
    event_times_ms = timing.bench_time_with_cuda_events(
        event_fn, warmup=warmup, rep=rep, device="cuda:0"
    )
    del event_fn
    gc.collect()
    torch.cuda.empty_cache()

    sol_fn = sol_factory()
    torch.cuda.synchronize()
    sol_result = run_sol_with_trace(
        timing, cupti_utils, sol_fn, warmup=warmup, rep=rep
    )
    del sol_fn
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "purpose": purpose,
        "event_times_ms": event_times_ms,
        "event_summary": summarize(event_times_ms),
        **sol_result,
    }


def gpu_metadata(torch) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-gpu=index,name,driver_version,pstate,"
            "clocks.current.sm,clocks.current.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "physical_gpu": 1,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "logical_gpu": 0,
        "name": properties.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "l2_cache_bytes": properties.L2_cache_size,
        "nvidia_smi": query,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_gpu1()
    torch, timing, cupti_utils, sol_head = load_sol_modules()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    torch.cuda.set_device(0)

    elements = 8 * 1024 * 1024
    small_elements = 1024 * 1024
    cases = [
        {
            "name": "single_add",
            "sol_factory": add_factory(torch, elements),
            "event_factory": add_factory(torch, elements),
            "purpose": "Single activity: SOL span, activity sum, and Event should converge.",
        },
        {
            "name": "two_sequential_adds",
            "sol_factory": sequential_factory(torch, elements),
            "event_factory": sequential_factory(torch, elements),
            "purpose": "Two default-stream activities: selected sum and span should be close.",
        },
        {
            "name": "two_adds_with_host_gap",
            "sol_factory": host_gap_factory(torch, small_elements, 0.0002),
            "event_factory": host_gap_factory(torch, small_elements, 0.0002),
            "purpose": "A host launch gap becomes a device idle gap: span should exceed sum.",
        },
        {
            "name": "two_stream_concurrent_sleep",
            "sol_factory": concurrent_sleep_factory(torch, 500_000),
            "event_factory": concurrent_sleep_factory(torch, 500_000),
            "purpose": "Hidden-stream activities test concurrency and default-stream Event coverage.",
        },
        {
            "name": "dynamic_extra_sin_after_discovery",
            "sol_factory": dynamic_extra_factory(
                torch, elements, discovery_after_calls=args.warmup + 1
            ),
            "event_factory": dynamic_full_event_factory(torch, elements),
            "purpose": (
                "Discovery sees add only; timed calls add sin. Tests whether unexpected "
                "legitimate activity is rejected or silently excluded."
            ),
        },
    ]

    started = time.time()
    results = []
    for case in cases:
        print(f"running {case['name']} on physical GPU 1", flush=True)
        results.append(
            run_case(
                torch,
                timing,
                cupti_utils,
                warmup=args.warmup,
                rep=args.rep,
                **case,
            )
        )

    payload = {
        "experiment": "01_sol_cupti_span",
        "timestamp_unix": started,
        "duration_seconds": time.time() - started,
        "sol_commit": sol_head,
        "sol_source": str(Path(os.environ["SOL_EXECBENCH_SRC"]).resolve()),
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cupti_python_version": "12.8.0 compatibility binding",
        "official_sol_declared_cupti_requirement": ">=13.0.1",
        "warmup": args.warmup,
        "rep": args.rep,
        "gpu": gpu_metadata(torch),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

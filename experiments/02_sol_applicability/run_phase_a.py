#!/usr/bin/env python3
"""Probe SOL activity selection across controlled single/multi-kernel cases."""

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
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any


SOL_COMMIT = "a9fa0804c793d438e70850c33fe34426e66d53dd"


def require_gpu1() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise RuntimeError(
            "This experiment must run on physical GPU 1: "
            f"set CUDA_VISIBLE_DEVICES=1 (got {visible!r})"
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
    head = subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != SOL_COMMIT:
        raise RuntimeError(f"Expected SOL commit {SOL_COMMIT}, got {head}")
    sys.path.insert(0, str(source_path))

    import torch
    from sol_execbench.core.bench import cupti_utils, timing

    return torch, timing, cupti_utils, head


def summarize_ms(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

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


def activity_key(activity) -> tuple[Any, ...]:
    return (
        activity.kernel_string(),
        activity.start,
        activity.end,
        activity.correlation_id,
    )


def is_cache_management_activity(activity) -> bool:
    # Scope-limited allowlist: none of this experiment's business workloads
    # launches a FillFunctor kernel. Do not reuse for arbitrary operators.
    return "FillFunctor" in activity.name


def multiset_difference(left: list[Any], right: list[Any]) -> list[Any]:
    remaining = Counter(activity_key(item) for item in right)
    result = []
    for item in left:
        key = activity_key(item)
        if remaining[key]:
            remaining[key] -= 1
        else:
            result.append(item)
    return result


def span_us(activities: list[Any]) -> float | None:
    if not activities:
        return None
    return (
        max(item.end for item in activities)
        - min(item.start for item in activities)
    ) / 1000


def sum_us(activities: list[Any]) -> float:
    return sum(item.end - item.start for item in activities) / 1000


def run_sol_with_strict_trace(
    timing,
    cupti_utils,
    fn: Callable[[], Any],
    *,
    warmup: int,
    rep: int,
    expected_business_counts: list[int],
    cold_l2_cache: bool,
    forced_status: str | None = None,
) -> dict[str, Any]:
    """Run official SOL and independently validate raw-vs-selected activities."""

    original_collect = timing.collect_cupti_activities
    original_timestamp = timing.cupti.get_timestamp
    captures = []
    timestamps: list[float] = []

    @contextlib.contextmanager
    def capturing_collect(*args, **kwargs) -> Iterator[Any]:
        buffers = None
        try:
            with original_collect(*args, **kwargs) as buffers:
                yield buffers
        finally:
            if buffers is not None:
                captures.append(buffers)

    def capturing_timestamp() -> float:
        timestamp = original_timestamp()
        timestamps.append(timestamp)
        return timestamp

    timing.collect_cupti_activities = capturing_collect
    timing.cupti.get_timestamp = capturing_timestamp
    official_error = None
    sol_times_ms: list[float] = []
    try:
        sol_times_ms = timing.bench_gpu_time_with_cupti(
            fn,
            warmup=warmup,
            rep=rep,
            cold_l2_cache=cold_l2_cache,
            device="cuda:0",
        )
    except Exception as exc:  # Preserve official fail-closed behavior as data.
        official_error = f"{type(exc).__name__}: {exc}"
    finally:
        timing.collect_cupti_activities = original_collect
        timing.cupti.get_timestamp = original_timestamp

    if len(captures) != 2:
        raise RuntimeError(f"Expected discovery and timing captures, got {len(captures)}")
    if len(timestamps) != rep * 2:
        raise RuntimeError(f"Expected {rep * 2} timestamps, got {len(timestamps)}")

    discovery_all = sorted(
        captures[0].kernels,
        key=lambda item: (item.start, item.end, item.correlation_id),
    )
    discovery_business = [
        item for item in discovery_all if not is_cache_management_activity(item)
    ]
    expected_signature = cupti_utils.kernel_activity_sequence(discovery_business)

    timed = sorted(
        captures[1].kernels,
        key=lambda item: (item.start, item.end, item.correlation_id),
    )
    starts = [item.start for item in timed]
    windows = []
    for iteration, expected_count in enumerate(expected_business_counts):
        start_cpu = timestamps[2 * iteration]
        end_cpu = timestamps[2 * iteration + 1]
        left = bisect.bisect_left(starts, start_cpu)
        right = bisect.bisect_right(starts, end_cpu)
        all_activities = timed[left:right]
        cache_activities = [
            item for item in all_activities if is_cache_management_activity(item)
        ]
        business_activities = [
            item for item in all_activities if not is_cache_management_activity(item)
        ]

        selection_error = None
        selected = []
        try:
            selected = cupti_utils.select_activity_sequence(
                all_activities,
                expected_signature,
                iteration=iteration,
            )
        except Exception as exc:
            selection_error = f"{type(exc).__name__}: {exc}"

        missing = multiset_difference(business_activities, selected)
        foreign = multiset_difference(selected, business_activities)
        business_signature = cupti_utils.kernel_activity_sequence(business_activities)
        count_matches = len(business_activities) == expected_count
        selection_complete = not missing and not foreign
        sequence_matches = business_signature == expected_signature
        strict_pass = (
            selection_error is None
            and count_matches
            and selection_complete
            and sequence_matches
        )

        windows.append(
            {
                "iteration": iteration,
                "cpu_window_ns": [start_cpu, end_cpu],
                "expected_business_activity_count": expected_count,
                "all_activity_count": len(all_activities),
                "cache_management_activity_count": len(cache_activities),
                "business_activity_count": len(business_activities),
                "selected_activity_count": len(selected),
                "count_matches_ground_truth": count_matches,
                "selection_complete": selection_complete,
                "ordered_signature_matches_discovery": sequence_matches,
                "strict_pass": strict_pass,
                "selection_error": selection_error,
                "business_duration_sum_us": sum_us(business_activities),
                "business_device_span_us": span_us(business_activities),
                "selected_duration_sum_us": sum_us(selected),
                "selected_device_span_us": span_us(selected),
                "business_signature": business_signature,
                "all_activities": [activity_record(item) for item in all_activities],
                "cache_management_activities": [
                    activity_record(item) for item in cache_activities
                ],
                "business_activities": [
                    activity_record(item) for item in business_activities
                ],
                "selected_activities": [activity_record(item) for item in selected],
                "missing_business_activities": [
                    activity_record(item) for item in missing
                ],
                "foreign_selected_activities": [
                    activity_record(item) for item in foreign
                ],
            }
        )

    if forced_status is not None:
        status = forced_status
    elif any(not window["count_matches_ground_truth"] for window in windows):
        status = "FAIL_CAPTURE"
    elif any(window["selection_error"] for window in windows):
        status = "FAIL_DYNAMIC_DISCOVERY"
    elif any(
        not window["selection_complete"]
        or not window["ordered_signature_matches_discovery"]
        for window in windows
    ):
        status = "FAIL_DYNAMIC_DISCOVERY"
    else:
        status = "PASS_STATIC_SOL"

    variants = Counter(tuple(window["business_signature"]) for window in windows)
    return {
        "status": status,
        "official_sol_error": official_error,
        "official_sol_times_ms": sol_times_ms,
        "official_sol_summary": summarize_ms(sol_times_ms),
        "strict_pass_count": sum(window["strict_pass"] for window in windows),
        "strict_fail_count": sum(not window["strict_pass"] for window in windows),
        "discovery_activities": [activity_record(item) for item in discovery_all],
        "discovery_business_signature": expected_signature,
        "timed_business_signature_variants": [
            {"signature": list(signature), "iterations": count}
            for signature, count in variants.items()
        ],
        "windows": windows,
    }


def single_add_factory(torch, elements: int):
    def factory():
        source = torch.randn(elements, device="cuda")
        output = torch.empty_like(source)

        def run():
            torch.add(source, 1.0, out=output)

        return run

    return factory


def single_sleep_factory(torch, cycles: int):
    def factory():
        def run():
            torch.cuda._sleep(cycles)

        return run

    return factory


def add_sin_factory(torch, elements: int):
    def factory():
        source = torch.randn(elements, device="cuda")
        add_output = torch.empty_like(source)
        sin_output = torch.empty_like(source)

        def run():
            torch.add(source, 1.0, out=add_output)
            torch.sin(source, out=sin_output)

        return run

    return factory


def two_adds_factory(torch, elements: int, gap_seconds: float = 0.0):
    def factory():
        source_a = torch.randn(elements, device="cuda")
        source_b = torch.randn(elements, device="cuda")
        output_a = torch.empty_like(source_a)
        output_b = torch.empty_like(source_b)

        def run():
            torch.add(source_a, 1.0, out=output_a)
            if gap_seconds:
                time.sleep(gap_seconds)
            torch.add(source_b, 1.0, out=output_b)

        return run

    return factory


def two_stream_sleep_factory(torch, cycles: int, *, join: bool):
    def factory():
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()
        done_a = torch.cuda.Event()
        done_b = torch.cuda.Event()

        def run():
            caller = torch.cuda.current_stream()
            with torch.cuda.stream(stream_a):
                torch.cuda._sleep(cycles)
                done_a.record()
            with torch.cuda.stream(stream_b):
                torch.cuda._sleep(cycles)
                done_b.record()
            if join:
                caller.wait_event(done_a)
                caller.wait_event(done_b)

        return run

    return factory


def dynamic_extra_factory(
    torch,
    elements: int,
    *,
    discovery_call: int,
    same_name: bool,
):
    def factory():
        source_a = torch.randn(elements, device="cuda")
        source_b = torch.randn(elements, device="cuda")
        output_a = torch.empty_like(source_a)
        output_b = torch.empty_like(source_b)
        calls = 0

        def run():
            nonlocal calls
            calls += 1
            torch.add(source_a, 1.0, out=output_a)
            if calls > discovery_call:
                if same_name:
                    torch.add(source_b, 1.0, out=output_b)
                else:
                    torch.sin(source_b, out=output_b)

        return run

    return factory


def alternating_order_factory(torch, elements: int, *, discovery_call: int):
    def factory():
        source = torch.randn(elements, device="cuda")
        add_output = torch.empty_like(source)
        sin_output = torch.empty_like(source)
        calls = 0

        def add_then_sin():
            torch.add(source, 1.0, out=add_output)
            torch.sin(source, out=sin_output)

        def sin_then_add():
            torch.sin(source, out=sin_output)
            torch.add(source, 1.0, out=add_output)

        def run():
            nonlocal calls
            calls += 1
            if calls <= discovery_call or (calls - discovery_call) % 2 == 0:
                add_then_sin()
            else:
                sin_then_add()

        return run

    return factory


def run_case(
    torch,
    timing,
    cupti_utils,
    *,
    name: str,
    purpose: str,
    sol_factory: Callable[[], Callable[[], None]],
    event_factory: Callable[[], Callable[[], None]],
    expected_business_count: int,
    warmup: int,
    rep: int,
    cold_l2_cache: bool,
    forced_status: str | None = None,
) -> dict[str, Any]:
    event_fn = event_factory()
    torch.cuda.synchronize()
    event_times_ms = timing.bench_time_with_cuda_events(
        event_fn,
        warmup=warmup,
        rep=rep,
        device="cuda:0",
    )
    del event_fn
    gc.collect()
    torch.cuda.empty_cache()

    sol_fn = sol_factory()
    torch.cuda.synchronize()
    sol_result = run_sol_with_strict_trace(
        timing,
        cupti_utils,
        sol_fn,
        warmup=warmup,
        rep=rep,
        expected_business_counts=[expected_business_count] * rep,
        cold_l2_cache=cold_l2_cache,
        forced_status=forced_status,
    )
    del sol_fn
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "purpose": purpose,
        "event_times_ms": event_times_ms,
        "event_summary": summarize_ms(event_times_ms),
        **sol_result,
    }


def gpu_metadata(torch) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-gpu=index,name,driver_version,pstate,utilization.gpu,"
            "memory.used,memory.total,clocks.current.sm,"
            "clocks.current.memory,temperature.gpu",
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


def external_gpu_process_summary() -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            "1",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    external_memory_mib = 0
    external_process_count = 0
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            pid = int(fields[0])
            memory_mib = int(fields[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        external_process_count += 1
        external_memory_mib += memory_mib
    return {
        "external_compute_process_count": external_process_count,
        "external_compute_memory_mib": external_memory_mib,
    }


def markdown_summary(payload: dict[str, Any]) -> str:
    cases = {case["name"]: case for case in payload["cases"]}
    dynamic_different = cases["dynamic_extra_different_name"]
    dynamic_same = cases["dynamic_extra_same_name"]
    dynamic_order = cases["dynamic_order_add_sin"]
    unjoined = cases["two_stream_unjoined_sleep"]
    lines = [
        "# Experiment 02 results",
        "",
        "状态：Phase A controlled workload 结果；不能外推为全部 TileOps case。",
        "",
        f"- SOL commit: `{payload['sol_commit']}`",
        f"- Physical GPU: `{payload['gpu']['physical_gpu']}`",
        f"- GPU: `{payload['gpu']['name']}`",
        f"- warmup/rep: `{payload['warmup']}/{payload['rep']}`",
        f"- cold L2 cache: `{payload['cold_l2_cache']}`",
        f"- duration: `{payload['duration_seconds']:.2f} s`",
        "",
        "| Case | Status | Strict pass | SOL median (us) | Event median (us) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        sol_summary = case["official_sol_summary"]
        sol_median = (
            f"{sol_summary['median_us']:.3f}" if sol_summary is not None else "error"
        )
        event_median = f"{case['event_summary']['median_us']:.3f}"
        lines.append(
            f"| `{case['name']}` | `{case['status']}` | "
            f"{case['strict_pass_count']}/{payload['rep']} | "
            f"{sol_median} | {event_median} |"
        )
    lines.extend(
        [
            "",
            "## 首轮观察",
            "",
            f"- 固定 single-kernel、固定不同名/同名 multi-kernel、host gap 和 joined multi-stream case 均为 `{payload['rep']}/{payload['rep']}` strict pass。",
            f"- 动态增加不同名 kernel：官方 SOL error 为 `{dynamic_different['official_sol_error']}`，但 strict validator 为 `{dynamic_different['strict_pass_count']}/{payload['rep']}`；每轮业务 activity 为 2，SOL 只选 1。",
            f"- 动态增加同名 kernel：官方 SOL error 为 `{dynamic_same['official_sol_error']}`，但 strict validator 为 `{dynamic_same['strict_pass_count']}/{payload['rep']}`；说明只校验 identity/count of selected set 不能发现额外同名 activity。",
            f"- 动态顺序交替：官方 SOL error 为 `{dynamic_order['official_sol_error']}`，strict validator 为 `{dynamic_order['strict_pass_count']}/{payload['rep']}`；原版 fallback matcher 可接受相同 multiset 的不同顺序。",
            f"- Unjoined multi-stream 中 SOL median 为 `{unjoined['official_sol_summary']['median_us']:.3f} us`，default-stream Event median 为 `{unjoined['event_summary']['median_us']:.3f} us`；SOL 的 per-iteration device synchronize 捕获了 worker streams，而 Event 没有 completion join。",
            "- Joined multi-stream 的 Event/SOL 数值仍有差异；本实验把它记录为待解释现象，不据此判断任一路径更准确，需用独立 CUPTI on/off 扰动实验分析。",
            "",
            "## 解释约束",
            "",
            "- `PASS_STATIC_SOL` 只表示本次合成 case 的每轮 raw/selected/count/order 校验通过。",
            "- `FAIL_DYNAMIC_DISCOVERY` 表示官方 SOL 可能仍返回 latency，但 strict validator 发现 timed dispatch 与 discovery 不一致。",
            "- `FAIL_ASYNC_BOUNDARY` 是 callable completion contract 不完整，不代表 CUPTI 没捕到 GPU work。",
            "- CUDA Event 是另一 measurement contract；多 stream 只有 joined case 才覆盖全部 worker streams。",
            "- 完整逐轮 activities 位于同目录 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument(
        "--warm-l2-cache",
        action="store_true",
        help="Disable SOL's per-iteration L2 cache clearing.",
    )
    parser.add_argument(
        "--allow-external-gpu-processes",
        action="store_true",
        help="Allow a diagnostic run despite other compute processes on GPU 1.",
    )
    args = parser.parse_args()
    if args.warmup < 1 or args.rep < 1:
        raise ValueError("warmup and rep must be positive")

    require_gpu1()
    torch, timing, cupti_utils, sol_head = load_sol_modules()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    torch.cuda.set_device(0)
    torch.cuda.synchronize()
    preflight = external_gpu_process_summary()
    if (
        preflight["external_compute_process_count"]
        and not args.allow_external_gpu_processes
    ):
        raise RuntimeError(
            "Physical GPU 1 is not exclusive: found "
            f"{preflight['external_compute_process_count']} external compute process(es) "
            f"holding {preflight['external_compute_memory_mib']} MiB. "
            "Wait for an exclusive window; use --allow-external-gpu-processes only "
            "for non-evidence diagnostic runs."
        )

    elements = 4 * 1024 * 1024
    small_elements = 1024 * 1024
    discovery_call = args.warmup + 1
    cases = [
        {
            "name": "single_add",
            "purpose": "A1: one business kernel.",
            "sol_factory": single_add_factory(torch, elements),
            "event_factory": single_add_factory(torch, elements),
            "expected_business_count": 1,
        },
        *[
            {
                "name": f"single_sleep_{cycles}_cycles",
                "purpose": "A2/A3: controlled single CUDA sleep kernel.",
                "sol_factory": single_sleep_factory(torch, cycles),
                "event_factory": single_sleep_factory(torch, cycles),
                "expected_business_count": 1,
            }
            for cycles in (2_000, 20_000, 200_000)
        ],
        {
            "name": "fixed_distinct_add_sin",
            "purpose": "A4: fixed two-kernel sequence with distinct identities.",
            "sol_factory": add_sin_factory(torch, elements),
            "event_factory": add_sin_factory(torch, elements),
            "expected_business_count": 2,
        },
        {
            "name": "fixed_same_name_two_adds",
            "purpose": "A5: same kernel identity repeated twice per call.",
            "sol_factory": two_adds_factory(torch, elements),
            "event_factory": two_adds_factory(torch, elements),
            "expected_business_count": 2,
        },
        {
            "name": "two_adds_with_host_gap",
            "purpose": "A8: fixed sequence with a 200 us host/device idle gap.",
            "sol_factory": two_adds_factory(torch, small_elements, 0.0002),
            "event_factory": two_adds_factory(torch, small_elements, 0.0002),
            "expected_business_count": 2,
        },
        {
            "name": "two_stream_joined_sleep",
            "purpose": "A9: two worker streams explicitly joined to caller stream.",
            "sol_factory": two_stream_sleep_factory(torch, 500_000, join=True),
            "event_factory": two_stream_sleep_factory(torch, 500_000, join=True),
            "expected_business_count": 2,
        },
        {
            "name": "two_stream_unjoined_sleep",
            "purpose": "A10: callable returns without joining worker streams.",
            "sol_factory": two_stream_sleep_factory(torch, 500_000, join=False),
            "event_factory": two_stream_sleep_factory(torch, 500_000, join=False),
            "expected_business_count": 2,
            "forced_status": "FAIL_ASYNC_BOUNDARY",
        },
        {
            "name": "dynamic_extra_different_name",
            "purpose": "A11: discovery sees add; timed iterations add an extra sin.",
            "sol_factory": dynamic_extra_factory(
                torch,
                elements,
                discovery_call=discovery_call,
                same_name=False,
            ),
            "event_factory": add_sin_factory(torch, elements),
            "expected_business_count": 2,
        },
        {
            "name": "dynamic_extra_same_name",
            "purpose": "A12: discovery sees one add; timed iterations launch two adds.",
            "sol_factory": dynamic_extra_factory(
                torch,
                elements,
                discovery_call=discovery_call,
                same_name=True,
            ),
            "event_factory": two_adds_factory(torch, elements),
            "expected_business_count": 2,
        },
        {
            "name": "dynamic_order_add_sin",
            "purpose": "A13: timed activity order alternates after fixed discovery.",
            "sol_factory": alternating_order_factory(
                torch,
                elements,
                discovery_call=discovery_call,
            ),
            "event_factory": add_sin_factory(torch, elements),
            "expected_business_count": 2,
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
                cold_l2_cache=not args.warm_l2_cache,
                **case,
            )
        )

    payload = {
        "experiment": "02_sol_applicability_phase_a",
        "timestamp_unix": started,
        "duration_seconds": time.time() - started,
        "sol_commit": sol_head,
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cupti_python_version": "12.8.0 compatibility binding",
        "official_sol_declared_cupti_requirement": ">=13.0.1",
        "warmup": args.warmup,
        "rep": args.rep,
        "cold_l2_cache": not args.warm_l2_cache,
        "gpu_preflight": preflight,
        "gpu": gpu_metadata(torch),
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(markdown_summary(payload))
        print(f"wrote {args.summary_output}")


if __name__ == "__main__":
    main()

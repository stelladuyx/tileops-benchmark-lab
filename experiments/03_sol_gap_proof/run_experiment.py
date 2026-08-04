#!/usr/bin/env python3
"""Prove that SOL's selected activity span includes inter-kernel GPU gap."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any


def load_phase_a_helpers():
    helper_path = (
        Path(__file__).resolve().parents[1]
        / "02_sol_applicability"
        / "run_phase_a.py"
    )
    spec = importlib.util.spec_from_file_location("sol_phase_a_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Phase A helpers from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize_us(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list")
    ordered = sorted(values)
    return {
        "min_us": ordered[0],
        "median_us": statistics.median(ordered),
        "mean_us": statistics.mean(ordered),
        "max_us": ordered[-1],
        "max_abs_us": max(abs(value) for value in ordered),
    }


def linear_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("linear_fit requires equal lists with at least two points")
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    ss_x = sum((value - mean_x) ** 2 for value in xs)
    if ss_x == 0:
        raise ValueError("Cannot fit a constant x series")
    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(xs, ys, strict=True)
    ) / ss_x
    intercept = mean_y - slope * mean_x
    predictions = [slope * value + intercept for value in xs]
    ss_residual = sum(
        (actual - predicted) ** 2
        for actual, predicted in zip(ys, predictions, strict=True)
    )
    ss_total = sum((value - mean_y) ** 2 for value in ys)
    r_squared = 1.0 - ss_residual / ss_total if ss_total else 1.0
    return {
        "slope": slope,
        "intercept_us": intercept,
        "r_squared": r_squared,
    }


def busy_wait_us(gap_us: int) -> None:
    deadline = time.perf_counter_ns() + gap_us * 1000
    while time.perf_counter_ns() < deadline:
        pass


def add_gap_sin_factory(torch, elements: int, gap_us: int):
    def factory():
        source = torch.randn(elements, device="cuda")
        add_output = torch.empty_like(source)
        sin_output = torch.empty_like(source)

        def run():
            torch.add(source, 1.0, out=add_output)
            busy_wait_us(gap_us)
            torch.sin(source, out=sin_output)

        return run

    return factory


def proof_rows(case_result: dict[str, Any]) -> list[dict[str, Any]]:
    official_times_ms = case_result["official_sol_times_ms"]
    windows = case_result["windows"]
    if len(official_times_ms) != len(windows):
        raise RuntimeError("Official SOL samples and traced windows differ in length")

    rows = []
    for official_ms, window in zip(official_times_ms, windows, strict=True):
        activities = sorted(
            window["business_activities"], key=lambda item: item["start_ns"]
        )
        if len(activities) != 2:
            rows.append(
                {
                    "iteration": window["iteration"],
                    "error": f"expected 2 business activities, got {len(activities)}",
                }
            )
            continue

        first, second = activities
        duration_a_us = (first["end_ns"] - first["start_ns"]) / 1000
        duration_b_us = (second["end_ns"] - second["start_ns"]) / 1000
        actual_gap_us = (second["start_ns"] - first["end_ns"]) / 1000
        duration_sum_us = duration_a_us + duration_b_us
        manual_span_us = (
            max(first["end_ns"], second["end_ns"])
            - min(first["start_ns"], second["start_ns"])
        ) / 1000
        official_span_us = official_ms * 1000
        closure_error_us = manual_span_us - duration_sum_us - actual_gap_us
        rows.append(
            {
                "iteration": window["iteration"],
                "s1_ns": first["start_ns"],
                "e1_ns": first["end_ns"],
                "s2_ns": second["start_ns"],
                "e2_ns": second["end_ns"],
                "business_activity_count": window["business_activity_count"],
                "selected_activity_count": window["selected_activity_count"],
                "selection_complete": window["selection_complete"],
                "ordered_signature_matches_discovery": window[
                    "ordered_signature_matches_discovery"
                ],
                "duration_a_us": duration_a_us,
                "duration_b_us": duration_b_us,
                "duration_sum_us": duration_sum_us,
                "actual_gpu_gap_us": actual_gap_us,
                "manual_span_us": manual_span_us,
                "official_sol_span_us": official_span_us,
                "official_minus_manual_us": official_span_us - manual_span_us,
                "span_minus_sum_us": manual_span_us - duration_sum_us,
                "closure_error_us": closure_error_us,
                "strict_pass": window["strict_pass"],
            }
        )
    return rows


def proof_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if "error" not in row]
    if not valid:
        raise RuntimeError("No valid proof rows")
    return {
        "valid_rows": len(valid),
        "error_rows": len(rows) - len(valid),
        "strict_pass_rows": sum(row["strict_pass"] for row in valid),
        "actual_gpu_gap": summarize_us(
            [row["actual_gpu_gap_us"] for row in valid]
        ),
        "duration_sum": summarize_us([row["duration_sum_us"] for row in valid]),
        "manual_span": summarize_us([row["manual_span_us"] for row in valid]),
        "span_minus_sum": summarize_us(
            [row["span_minus_sum_us"] for row in valid]
        ),
        "closure_error": summarize_us(
            [row["closure_error_us"] for row in valid]
        ),
        "official_minus_manual": summarize_us(
            [row["official_minus_manual_us"] for row in valid]
        ),
        "minimum_actual_gpu_gap_us": min(
            row["actual_gpu_gap_us"] for row in valid
        ),
    }


def compact_payload(payload: dict[str, Any]) -> None:
    """Remove duplicated trace structures after proof rows are materialized."""
    for case in payload["cases"]:
        for row in case["proof_rows"]:
            # Older/in-memory payloads may contain these repeated long strings.
            row.pop("first_identity", None)
            row.pop("second_identity", None)
        for field in ("windows", "official_sol_times_ms", "event_times_ms"):
            case.pop(field, None)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    all_rows = []
    all_case_pass = True
    for case in payload["cases"]:
        rows = proof_rows(case)
        case["proof_rows"] = rows
        case["proof_summary"] = proof_summary(rows)
        all_rows.extend(row for row in rows if "error" not in row)
        all_case_pass = all_case_pass and case["status"] == "PASS_STATIC_SOL"

    actual_gaps = [row["actual_gpu_gap_us"] for row in all_rows]
    spans = [row["manual_span_us"] for row in all_rows]
    sums = [row["duration_sum_us"] for row in all_rows]
    span_fit = linear_fit(actual_gaps, spans)
    sum_fit = linear_fit(actual_gaps, sums)
    max_closure_error = max(abs(row["closure_error_us"]) for row in all_rows)
    max_official_error = max(
        abs(row["official_minus_manual_us"]) for row in all_rows
    )
    minimum_gap = min(actual_gaps)

    checks = {
        "all_cases_pass_static_sol": all_case_pass,
        "all_rows_strict_pass": all(row["strict_pass"] for row in all_rows),
        "all_gaps_nonnegative": minimum_gap >= -0.001,
        "closure_error_le_0_001_us": max_closure_error <= 0.001,
        "official_matches_manual_le_0_001_us": max_official_error <= 0.001,
        "span_gap_slope_0_95_to_1_05": 0.95 <= span_fit["slope"] <= 1.05,
        "span_gap_r_squared_ge_0_99": span_fit["r_squared"] >= 0.99,
        "duration_sum_gap_abs_slope_le_0_05": abs(sum_fit["slope"]) <= 0.05,
    }
    return {
        "overall_pass": all(checks.values()),
        "checks": checks,
        "row_count": len(all_rows),
        "minimum_actual_gpu_gap_us": minimum_gap,
        "maximum_abs_closure_error_us": max_closure_error,
        "maximum_abs_official_minus_manual_us": max_official_error,
        "span_vs_actual_gap_fit": span_fit,
        "duration_sum_vs_actual_gap_fit": sum_fit,
    }


def markdown_summary(payload: dict[str, Any]) -> str:
    analysis = payload["analysis"]
    lines = [
        "# Experiment 03 results: SOL inter-kernel gap proof",
        "",
        f"结论：`{'PASS' if analysis['overall_pass'] else 'FAIL'}`。",
        "",
        f"- SOL commit: `{payload['sol_commit']}`",
        f"- Physical GPU: `{payload['gpu']['physical_gpu']}`",
        f"- GPU: `{payload['gpu']['name']}`",
        f"- warmup/rep/sessions: `{payload['warmup']}/{payload['rep']}/{payload['sessions']}`",
        f"- pooled proof rows: `{analysis['row_count']}`",
        f"- external compute processes at every session preflight: `{[item['external_compute_process_count'] for item in payload['gpu_preflights']]}`",
        "",
        "| Session | Requested gap (us) | Strict pass | Actual GPU gap median (us) | Duration sum median (us) | SOL span median (us) | Span-sum median (us) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in sorted(
        payload["cases"], key=lambda item: (item["session"], item["requested_gap_us"])
    ):
        proof = case["proof_summary"]
        lines.append(
            f"| {case['session']} | {case['requested_gap_us']} | "
            f"{proof['strict_pass_rows']}/{payload['rep']} | "
            f"{proof['actual_gpu_gap']['median_us']:.3f} | "
            f"{proof['duration_sum']['median_us']:.3f} | "
            f"{proof['manual_span']['median_us']:.3f} | "
            f"{proof['span_minus_sum']['median_us']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Closure 和线性检查",
            "",
            "```text",
            "SOL span - duration sum = actual GPU gap",
            "```",
            "",
            f"- maximum absolute closure error: `{analysis['maximum_abs_closure_error_us']:.9f} us`",
            f"- maximum official SOL minus manual span: `{analysis['maximum_abs_official_minus_manual_us']:.9f} us`",
            f"- `SOL span ~ actual gap`: slope `{analysis['span_vs_actual_gap_fit']['slope']:.6f}`, R² `{analysis['span_vs_actual_gap_fit']['r_squared']:.9f}`",
            f"- `duration sum ~ actual gap`: slope `{analysis['duration_sum_vs_actual_gap_fit']['slope']:.6f}`, R² `{analysis['duration_sum_vs_actual_gap_fit']['r_squared']:.9f}`",
            "",
            "## Pass checks",
            "",
        ]
    )
    for name, passed in analysis["checks"].items():
        lines.append(f"- `[{'x' if passed else ' '}]` `{name}`")
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "实验直接使用 CUPTI activity timestamps，而不是请求的 host busy-wait 作为 ground truth。",
            "当实际 GPU gap 增长时，SOL span 以约 1:1 增长，而两条 kernel 的 duration sum 基本不随 gap 增长。",
            "这验证的是固定、完整选中的两条 activities；如果 SOL 漏选 activity，结论不成立，必须先通过 strict validator。",
            "CUDA Event 是辅助对照，不参与上述 closure proof。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument(
        "--gaps-us",
        type=int,
        nargs="+",
        default=[0, 20, 50, 100, 200, 500, 1000],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.rep < 1 or args.sessions < 1:
        raise ValueError("warmup, rep, and sessions must be positive")
    if len(set(args.gaps_us)) < 2 or any(value < 0 for value in args.gaps_us):
        raise ValueError("gaps-us must contain at least two distinct nonnegative values")

    helpers = load_phase_a_helpers()
    helpers.require_gpu1()
    torch, timing, cupti_utils, sol_head = helpers.load_sol_modules()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    torch.cuda.set_device(0)

    elements = 4 * 1024 * 1024
    started = time.time()
    cases = []
    preflights = []
    run_order = []
    for session in range(args.sessions):
        torch.cuda.synchronize()
        preflight = helpers.external_gpu_process_summary()
        preflights.append(preflight)
        if preflight["external_compute_process_count"]:
            raise RuntimeError(
                "Physical GPU 1 is not exclusive before session "
                f"{session}: {preflight['external_compute_process_count']} external "
                "compute process(es)."
            )

        session_gaps = list(args.gaps_us)
        random.Random(20260803 + session).shuffle(session_gaps)
        run_order.append({"session": session, "requested_gaps_us": session_gaps})
        for gap_us in session_gaps:
            print(
                f"running session {session}, requested gap {gap_us} us on GPU 1",
                flush=True,
            )
            factory = add_gap_sin_factory(torch, elements, gap_us)
            case = helpers.run_case(
                torch,
                timing,
                cupti_utils,
                name=f"gap_{gap_us}_us",
                purpose="Two distinct same-stream kernels separated by host busy-wait.",
                sol_factory=factory,
                event_factory=factory,
                expected_business_count=2,
                warmup=args.warmup,
                rep=args.rep,
                cold_l2_cache=True,
            )
            case["session"] = session
            case["requested_gap_us"] = gap_us
            cases.append(case)

    payload = {
        "experiment": "03_sol_gap_proof",
        "timestamp_unix": started,
        "duration_seconds": time.time() - started,
        "sol_commit": sol_head,
        "python": helpers.sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cupti_python_version": "12.8.0 compatibility binding",
        "warmup": args.warmup,
        "rep": args.rep,
        "sessions": args.sessions,
        "requested_gaps_us": args.gaps_us,
        "run_order": run_order,
        "gpu_preflights": preflights,
        "gpu": helpers.gpu_metadata(torch),
        "cases": cases,
    }
    payload["analysis"] = analyze(payload)
    compact_payload(payload)

    # Reject NaN/inf before producing public JSON.
    for fit in (
        payload["analysis"]["span_vs_actual_gap_fit"],
        payload["analysis"]["duration_sum_vs_actual_gap_fit"],
    ):
        if not all(math.isfinite(value) for value in fit.values()):
            raise RuntimeError(f"Non-finite regression result: {fit}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(markdown_summary(payload))
        print(f"wrote {args.summary_output}")

    if not payload["analysis"]["overall_pass"]:
        raise SystemExit("Experiment completed but one or more proof checks failed")


if __name__ == "__main__":
    main()

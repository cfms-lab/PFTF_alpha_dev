"""Command-line runner for the first B0-P2 synthetic benchmark."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from .baselines import BaselineID, BaselineResult, BenchmarkConfig, run_case_benchmarks
from .calibration import (
    AdaptiveCalibrationResult,
    P2ConfidenceCalibrationResult,
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
)
from .synthetic import PanelSplit, make_minimal_panel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run PFTF-alpha B0-P2 on the six-case synthetic panel."
    )
    parser.add_argument(
        "--split",
        choices=[split.value for split in PanelSplit],
        default=PanelSplit.CALIBRATION.value,
    )
    parser.add_argument("--point-count", type=int, default=64)
    parser.add_argument("--reference-count", type=int, default=1024)
    parser.add_argument("--surface-samples", type=int, default=192)
    parser.add_argument("--b3-candidate-budget", type=int, default=16)
    parser.add_argument("--adaptive-knn", type=int, default=12)
    parser.add_argument("--b4-scale-multiplier", type=float, default=None)
    parser.add_argument("--b5-scale-multiplier", type=float, default=None)
    parser.add_argument("--b5-normal-penalty", type=float, default=4.0)
    parser.add_argument("--p1-scale-multiplier", type=float, default=None)
    parser.add_argument("--p1-relation-gain", type=float, default=2.0)
    parser.add_argument("--p1-max-condition", type=float, default=9.0)
    parser.add_argument("--p1-density-contrast-scale", type=float, default=0.5)
    parser.add_argument("--p1-imbalance-weight", type=float, default=0.5)
    parser.add_argument("--p2-scale-multiplier", type=float, default=None)
    parser.add_argument("--p2-confidence-threshold", type=float, default=None)
    parser.add_argument("--p2-target-fallback-fraction", type=float, default=0.25)
    parser.add_argument(
        "--calibrate-adaptive",
        action="store_true",
        help=(
            "freeze a reference-free P2 confidence threshold when needed, then "
            "select one B4/B5/P1/P2 multiplier on the calibration panel"
        ),
    )
    parser.add_argument("--adaptive-calibration-budget", type=int, default=24)
    parser.add_argument("--resample-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method.value for method in BaselineID],
        default=[method.value for method in BaselineID],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (default: benchmark-out/b0_p2_<split>.json)",
    )
    return parser


def _alpha_text(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _parameter_text(result: BaselineResult) -> str:
    if result.selection_parameter_name is not None:
        value = result.selection_parameter_value
        rendered = "n/a" if value is None else f"{value:.6g}"
        return f"{result.selection_parameter_name}={rendered}"
    return f"alpha^2={_alpha_text(result.alpha_squared)}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    split = PanelSplit(args.split)
    methods = tuple(BaselineID(method) for method in args.methods)
    adaptive_methods = tuple(
        method
        for method in methods
        if method
        in (
            BaselineID.B4_DENSITY_SCALED,
            BaselineID.B5_PCA_ANISOTROPIC,
            BaselineID.P1_PFTF_LOCAL_SPD,
            BaselineID.P2_CONFIDENCE_FALLBACK,
        )
    )
    if args.calibrate_adaptive and (
        args.b4_scale_multiplier is not None
        or args.b5_scale_multiplier is not None
        or args.p1_scale_multiplier is not None
        or args.p2_scale_multiplier is not None
    ):
        parser.error(
            "--calibrate-adaptive cannot be combined with explicit B4/B5/P1/P2 "
            "scale multipliers"
        )
    if args.calibrate_adaptive and not adaptive_methods:
        parser.error("--calibrate-adaptive requires B4, B5, P1, and/or P2 in --methods")
    if not 0.0 < args.p2_target_fallback_fraction < 1.0:
        parser.error("--p2-target-fallback-fraction must lie strictly between 0 and 1")

    initial_p2_threshold = (
        0.5 if args.p2_confidence_threshold is None else args.p2_confidence_threshold
    )

    config = BenchmarkConfig(
        surface_sample_count=args.surface_samples,
        b3_candidate_budget=args.b3_candidate_budget,
        resample_repeats=args.resample_repeats,
        adaptive_k_neighbors=args.adaptive_knn,
        b4_scale_multiplier=args.b4_scale_multiplier,
        b5_scale_multiplier=args.b5_scale_multiplier,
        p1_scale_multiplier=args.p1_scale_multiplier,
        p2_scale_multiplier=args.p2_scale_multiplier,
        p2_confidence_threshold=initial_p2_threshold,
        p1_relation_gain=args.p1_relation_gain,
        p1_max_condition_number=args.p1_max_condition,
        p1_density_contrast_scale=args.p1_density_contrast_scale,
        p1_receiver_imbalance_weight=args.p1_imbalance_weight,
        b5_max_normal_penalty=args.b5_normal_penalty,
        seed=args.seed,
    )
    calibrations: list[AdaptiveCalibrationResult] = []
    p2_confidence_calibration: P2ConfidenceCalibrationResult | None = None
    if args.calibrate_adaptive:
        calibration_cases = make_minimal_panel(
            split=PanelSplit.CALIBRATION,
            point_count=args.point_count,
            reference_count=args.reference_count,
            seed=args.seed,
        )
        if (
            BaselineID.P2_CONFIDENCE_FALLBACK in adaptive_methods
            and args.p2_confidence_threshold is None
        ):
            print("[calibration] freezing reference-free P2 confidence threshold")
            p2_confidence_calibration = calibrate_p2_confidence_threshold(
                calibration_cases,
                config=config,
                target_fallback_fraction=args.p2_target_fallback_fraction,
            )
            config = replace(
                config,
                p2_confidence_threshold=p2_confidence_calibration.threshold,
            )
            print(
                "  froze confidence_threshold="
                f"{p2_confidence_calibration.threshold:.6g}; "
                "fallback_fraction="
                f"{p2_confidence_calibration.achieved_fallback_fraction:.3f}"
            )
        for method in adaptive_methods:
            print(f"[calibration] selecting one {method.value} multiplier")
            calibration = calibrate_adaptive_multiplier(
                calibration_cases,
                method,
                config=config,
                candidate_budget=args.adaptive_calibration_budget,
            )
            calibrations.append(calibration)
            if method is BaselineID.B4_DENSITY_SCALED:
                config = replace(
                    config,
                    b4_scale_multiplier=calibration.multiplier,
                )
            elif method is BaselineID.B5_PCA_ANISOTROPIC:
                config = replace(
                    config,
                    b5_scale_multiplier=calibration.multiplier,
                )
            elif method is BaselineID.P1_PFTF_LOCAL_SPD:
                config = replace(
                    config,
                    p1_scale_multiplier=calibration.multiplier,
                )
            else:
                config = replace(
                    config,
                    p2_scale_multiplier=calibration.multiplier,
                )
            print(
                f"  froze local_scale_multiplier={calibration.multiplier:.6g} "
                f"from {calibration.candidate_count} candidates"
            )

    cases = make_minimal_panel(
        split=split,
        point_count=args.point_count,
        reference_count=args.reference_count,
        seed=args.seed,
    )

    reports = []
    for case in cases:
        print(f"[{case.family.value}] running {', '.join(args.methods)}")
        report = run_case_benchmarks(
            case,
            config=config,
            methods=methods,
        )
        reports.append(report)
        summary = ", ".join(
            (
                f"{result.method.value}:{_parameter_text(result)} "
                f"F={result.endpoints.fscore:.3f} "
                f"components={result.endpoints.connected_components}"
            )
            for result in report.results
        )
        print(f"  {summary}")

    output = args.output
    if output is None:
        output = Path("benchmark-out") / f"b0_p2_{split.value}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 9,
        "created_utc": datetime.now(UTC).isoformat(),
        "topology_endpoint_contract": {
            "homology_coefficients": "GF(2)",
            "betti_target_role": "evaluation_only",
            "selection_topology_term": "component_error",
            "false_bridges_semantics": "component_merge_proxy",
            "false_splits_semantics": "component_split_proxy",
        },
        "selection_contract": {
            "B0": "no selection",
            "B1": "fixed normalized radius; no reference",
            "B2": "exhaustive top-simplex critical scan; dense reference oracle",
            "B3": "topology persistence plus point fit and resampling; no reference",
            "B4": (
                "kNN density scale; per-case oracle unless a multiplier is "
                "explicitly set or frozen on the calibration panel"
            ),
            "B5": (
                "density-normalized PCA anisotropy; per-case oracle unless a "
                "multiplier is explicitly set or frozen on the calibration panel"
            ),
            "P1": (
                "directed-relation bounded local SPD with confidence softening; "
                "per-case oracle unless a multiplier is explicitly set or frozen "
                "on the calibration panel"
            ),
            "P2": (
                "P1 with an explicit or reference-free calibration-only confidence "
                "threshold frozen before multiplier selection; low-confidence cells "
                "must also pass the trusted B4 density score via max(P1, B4); "
                "per-case oracle unless a multiplier is explicitly set or frozen "
                "on the calibration panel; conservative prototype, not exact CGAL"
            ),
        },
        "p2_confidence_calibration": {
            "enabled": p2_confidence_calibration is not None,
            "source_split": (
                PanelSplit.CALIBRATION.value
                if p2_confidence_calibration is not None
                else None
            ),
            "explicit_threshold": args.p2_confidence_threshold is not None,
            "requested_target_fallback_fraction": args.p2_target_fallback_fraction,
            "result": (
                p2_confidence_calibration.to_dict()
                if p2_confidence_calibration is not None
                else None
            ),
        },
        "adaptive_calibration": {
            "enabled": args.calibrate_adaptive,
            "source_split": (
                PanelSplit.CALIBRATION.value if args.calibrate_adaptive else None
            ),
            "requested_candidate_budget": (
                args.adaptive_calibration_budget if args.calibrate_adaptive else None
            ),
            "results": [calibration.to_dict() for calibration in calibrations],
        },
        "config": asdict(config),
        "cases": [report.to_dict() for report in reports],
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

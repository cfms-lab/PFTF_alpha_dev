"""Command-line runner for the first B0-P2 synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from .baselines import BaselineID, BaselineResult, BenchmarkConfig, run_case_benchmarks
from .calibration import (
    AdaptiveCalibrationResult,
    BoundaryBridgeLocalizationResult,
    BoundaryOwnerInterventionAblationResult,
    BoundaryRegionCutAblationResult,
    BridgePenaltyAblationResult,
    P2ConfidenceCalibrationResult,
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
    evaluate_boundary_bridge_localization,
    evaluate_boundary_owner_intervention,
    evaluate_boundary_region_cut_ablation,
    evaluate_bridge_penalty_ablation,
)
from .exact import ExactPredicatePanelAudit, audit_exact_predicate_panel
from .exact_backend import (
    ExactConstructionPanelResult,
    evaluate_exact_construction_panel,
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
    parser.add_argument(
        "--bridge-probe-normal-coherence-threshold",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--bridge-probe-normal-edge-threshold",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--bridge-probe-length-edge-threshold",
        type=float,
        default=1.8,
    )
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
    parser.add_argument(
        "--evaluate-bridge-penalty",
        action="store_true",
        help=("run a calibration-only soft bridge-penalty audit without deploying it"),
    )
    parser.add_argument(
        "--bridge-penalty-strengths",
        nargs="+",
        type=float,
        default=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8),
        help="evaluation-only penalty strengths; must include zero",
    )
    parser.add_argument(
        "--evaluate-boundary-bridges",
        action="store_true",
        help="evaluate label-free boundary and dual bridge localization",
    )
    parser.add_argument(
        "--evaluate-boundary-intervention",
        action="store_true",
        help="run a calibration-only iterative boundary-owner pruning audit",
    )
    parser.add_argument(
        "--boundary-intervention-rounds",
        nargs="+",
        type=int,
        default=(0, 1, 2, 4),
        help="evaluation-only pruning depths; must include zero",
    )
    parser.add_argument(
        "--evaluate-boundary-region-cuts",
        action="store_true",
        help="run a calibration-only connected boundary-region/cut audit",
    )
    parser.add_argument(
        "--evaluate-exact-predicates",
        action="store_true",
        help=(
            "audit SciPy/Qhull connectivity with exact rational orientation and "
            "in-sphere signs; does not construct or select an exact complex"
        ),
    )
    parser.add_argument(
        "--evaluate-exact-construction",
        action="store_true",
        help="validate an optional exact backend handoff without using its cells",
    )
    parser.add_argument("--exact-backend", type=Path, default=None)
    parser.add_argument(
        "--exact-backend-arg",
        action="append",
        default=[],
        help="argument appended to the explicitly supplied exact backend command",
    )
    parser.add_argument(
        "--exact-backend-timeout",
        type=float,
        default=60.0,
        help="per-case exact backend timeout in seconds",
    )
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
    if (
        args.evaluate_bridge_penalty
        and BaselineID.P2_CONFIDENCE_FALLBACK not in methods
    ):
        parser.error("--evaluate-bridge-penalty requires P2 in --methods")
    if (
        args.evaluate_bridge_penalty
        and not args.calibrate_adaptive
        and args.p2_scale_multiplier is None
    ):
        parser.error(
            "--evaluate-bridge-penalty requires --calibrate-adaptive or an explicit "
            "--p2-scale-multiplier"
        )
    if (
        args.evaluate_boundary_bridges
        and BaselineID.P2_CONFIDENCE_FALLBACK not in methods
    ):
        parser.error("--evaluate-boundary-bridges requires P2 in --methods")
    if (
        args.evaluate_boundary_bridges
        and not args.calibrate_adaptive
        and args.p2_scale_multiplier is None
    ):
        parser.error(
            "--evaluate-boundary-bridges requires --calibrate-adaptive or an "
            "explicit --p2-scale-multiplier"
        )
    if (
        args.evaluate_boundary_intervention
        and BaselineID.P2_CONFIDENCE_FALLBACK not in methods
    ):
        parser.error("--evaluate-boundary-intervention requires P2 in --methods")
    if (
        args.evaluate_boundary_intervention
        and not args.calibrate_adaptive
        and args.p2_scale_multiplier is None
    ):
        parser.error(
            "--evaluate-boundary-intervention requires --calibrate-adaptive or an "
            "explicit --p2-scale-multiplier"
        )
    if (
        args.evaluate_boundary_region_cuts
        and BaselineID.P2_CONFIDENCE_FALLBACK not in methods
    ):
        parser.error("--evaluate-boundary-region-cuts requires P2 in --methods")
    if (
        args.evaluate_boundary_region_cuts
        and not args.calibrate_adaptive
        and args.p2_scale_multiplier is None
    ):
        parser.error(
            "--evaluate-boundary-region-cuts requires --calibrate-adaptive or an "
            "explicit --p2-scale-multiplier"
        )
    if not 0.0 < args.p2_target_fallback_fraction < 1.0:
        parser.error("--p2-target-fallback-fraction must lie strictly between 0 and 1")
    if args.exact_backend is not None and not args.evaluate_exact_construction:
        parser.error("--exact-backend requires --evaluate-exact-construction")
    if args.exact_backend_arg and args.exact_backend is None:
        parser.error("--exact-backend-arg requires --exact-backend")
    if not math.isfinite(args.exact_backend_timeout) or args.exact_backend_timeout <= 0:
        parser.error("--exact-backend-timeout must be finite and positive")

    initial_p2_threshold = (
        0.5 if args.p2_confidence_threshold is None else args.p2_confidence_threshold
    )

    config = BenchmarkConfig(
        surface_sample_count=args.surface_samples,
        b3_candidate_budget=args.b3_candidate_budget,
        resample_repeats=args.resample_repeats,
        adaptive_k_neighbors=args.adaptive_knn,
        bridge_probe_normal_coherence_threshold=args.bridge_probe_normal_coherence_threshold,
        bridge_probe_normal_edge_threshold=args.bridge_probe_normal_edge_threshold,
        bridge_probe_length_edge_threshold=args.bridge_probe_length_edge_threshold,
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
    calibration_cases = None
    bridge_penalty_ablation: BridgePenaltyAblationResult | None = None
    boundary_bridge_localization: BoundaryBridgeLocalizationResult | None = None
    boundary_owner_intervention: BoundaryOwnerInterventionAblationResult | None = None
    boundary_region_cut_ablation: BoundaryRegionCutAblationResult | None = None
    exact_predicate_audit: ExactPredicatePanelAudit | None = None
    exact_construction_result: ExactConstructionPanelResult | None = None

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

    if args.evaluate_bridge_penalty:
        if calibration_cases is None:
            calibration_cases = make_minimal_panel(
                split=PanelSplit.CALIBRATION,
                point_count=args.point_count,
                reference_count=args.reference_count,
                seed=args.seed,
            )
        print("[calibration] evaluating bridge-penalty proxy alignment")
        bridge_penalty_ablation = evaluate_bridge_penalty_ablation(
            calibration_cases,
            config=config,
            strengths=args.bridge_penalty_strengths,
        )
        flagged_spearman = _alpha_text(
            bridge_penalty_ablation.selected_flagged_vs_false_bridge_spearman
        )
        risk_spearman = _alpha_text(
            bridge_penalty_ablation.selected_mean_risk_vs_false_bridge_spearman
        )
        print(
            "  promotion_supported="
            f"{bridge_penalty_ablation.promotion_supported}; "
            f"flagged/edge Spearman={flagged_spearman}; "
            f"risk/edge Spearman={risk_spearman}"
        )
    if args.evaluate_boundary_intervention:
        if calibration_cases is None:
            calibration_cases = make_minimal_panel(
                split=PanelSplit.CALIBRATION,
                point_count=args.point_count,
                reference_count=args.reference_count,
                seed=args.seed,
            )
        print("[calibration] evaluating iterative boundary-owner intervention")
        boundary_owner_intervention = evaluate_boundary_owner_intervention(
            calibration_cases,
            config=config,
            rounds=args.boundary_intervention_rounds,
        )
        print(
            "  promotion_supported="
            f"{boundary_owner_intervention.promotion_supported}; "
            "eligible_rounds="
            f"{list(boundary_owner_intervention.eligible_rounds)}"
        )
    if args.evaluate_boundary_region_cuts:
        if calibration_cases is None:
            calibration_cases = make_minimal_panel(
                split=PanelSplit.CALIBRATION,
                point_count=args.point_count,
                reference_count=args.reference_count,
                seed=args.seed,
            )
        print("[calibration] evaluating connected boundary regions and safe cuts")
        boundary_region_cut_ablation = evaluate_boundary_region_cut_ablation(
            calibration_cases,
            config=config,
        )
        print(
            "  promotion_supported="
            f"{boundary_region_cut_ablation.promotion_supported}; "
            "eligible_strategies="
            f"{list(boundary_region_cut_ablation.eligible_strategies)}"
        )
    cases = make_minimal_panel(
        split=split,
        point_count=args.point_count,
        reference_count=args.reference_count,
        seed=args.seed,
    )

    if args.evaluate_exact_predicates:
        print(f"[{split.value}] auditing exact orientation/in-sphere signs")
        exact_predicate_audit = audit_exact_predicate_panel(
            ((case.family.value, case.points) for case in cases),
            evaluation_split=split.value,
        )
        totals = exact_predicate_audit.to_dict()["totals"]
        print(
            "  predicate_consistent="
            f"{exact_predicate_audit.all_predicates_consistent}; "
            "unique_combinatorics="
            f"{exact_predicate_audit.all_unique_delaunay_combinatorics_supported}; "
            f"local_violations={totals['exact_local_delaunay_violation_count']}; "
            f"cospherical={totals['exact_cospherical_interior_facet_count']}"
        )

    if args.evaluate_exact_construction:
        backend_command = None
        if args.exact_backend is not None:
            backend_command = (str(args.exact_backend), *args.exact_backend_arg)
        print(f"[{split.value}] validating optional exact construction handoff")
        exact_construction_result = evaluate_exact_construction_panel(
            ((case.family.value, case.points) for case in cases),
            evaluation_split=split.value,
            backend_command=backend_command,
            timeout_seconds=args.exact_backend_timeout,
        )
        print(
            f"  backend_requested={exact_construction_result.backend_requested}; "
            f"accepted_cases={exact_construction_result.accepted_case_count}; "
            f"blocking={list(exact_construction_result.blocking_reasons)}"
        )
    if args.evaluate_boundary_bridges:
        print(f"[{split.value}] evaluating boundary/dual bridge localization")
        boundary_bridge_localization = evaluate_boundary_bridge_localization(
            cases,
            config=config,
        )
        print(
            "  face AUC="
            f"{_alpha_text(boundary_bridge_localization.pooled_face_auc)}; "
            f"recall={_alpha_text(boundary_bridge_localization.pooled_face_recall)}; "
            "FPR="
            f"{_alpha_text(boundary_bridge_localization.pooled_face_false_positive_rate)}"
        )
        print(
            "  edge AUC="
            f"{_alpha_text(boundary_bridge_localization.pooled_edge_auc)}; "
            f"recall={_alpha_text(boundary_bridge_localization.pooled_edge_recall)}; "
            "FPR="
            f"{_alpha_text(boundary_bridge_localization.pooled_edge_false_positive_rate)}"
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
        "schema_version": 17,
        "created_utc": datetime.now(UTC).isoformat(),
        "topology_endpoint_contract": {
            "homology_coefficients": "GF(2)",
            "betti_target_role": "evaluation_only",
            "selection_topology_term": "component_error",
            "false_bridges_semantics": "component_merge_proxy",
            "false_splits_semantics": "component_split_proxy",
            "labeled_false_bridge_role": "evaluation_only",
            "labeled_false_bridge_semantics": (
                "unique mesh edges and faces spanning declared synthetic "
                "point-component labels"
            ),
        },
        "bridge_risk_probe_contract": {
            "role": "evaluation_only",
            "risk_inputs": "observed_points_only",
            "risk_uses_reference_or_labels": False,
            "labels_role": "AUC_recall_FPR_evaluation_only",
            "flag_rule": "risk > 1.0",
            "selection_effect": "none",
        },
        "bridge_penalty_ablation_contract": {
            "role": "calibration_only_evaluation_no_selection",
            "base_method": "P2",
            "penalty_formula": "score * (1 + strength * max(risk - 1, 0))",
            "risk_inputs": "observed_points_only",
            "endpoint_labels_role": "evaluation_only_promotion_gate",
            "selection_effect": "none",
            "held_out_tuning": "prohibited",
        },
        "boundary_bridge_localization_contract": {
            "role": "evaluation_only_no_selection",
            "base_method": "P2",
            "risk_inputs": "observed_points_and_frozen_P2_boundary_only",
            "edge_risk": "route_specific_normal_or_normalized_length_risk",
            "face_risk": "maximum_incident_boundary_edge_risk",
            "flag_rule": "risk > 1.0",
            "dual_cut_role": "audit_only_not_fused_into_geometric_risk",
            "reference_geometry_use": False,
            "component_labels_role": "AUC_recall_FPR_evaluation_only",
            "selection_effect": "none",
        },
        "boundary_owner_intervention_contract": {
            "role": "calibration_only_evaluation_no_selection",
            "base_method": "P2",
            "owner_rule": (
                "all unique owners of current boundary faces with risk > 1.0"
            ),
            "boundary_recomputation": "after_every_round",
            "intervention_inputs": ("observed_points_and_frozen_P2_boundary_only"),
            "endpoint_labels_role": "evaluation_only_promotion_gate",
            "promotion_gate": (
                "objective_geometry_component_Betti_nonregression_and_strict_"
                "labeled_edge_face_bridge_improvement"
            ),
            "selection_effect": "none",
            "held_out_tuning": "prohibited",
        },
        "boundary_region_cut_ablation_contract": {
            "role": "calibration_only_evaluation_no_selection",
            "base_method": "P2",
            "strategies": [
                "baseline",
                "largest_risk_region",
                "safe_backbone_cut",
            ],
            "region_adjacency": "flagged_faces_share_flagged_boundary_edge",
            "safe_backbone_cut": (
                "flagged_edge_endpoints_in_distinct_safe_edge_components"
            ),
            "intervention_inputs": ("observed_points_and_frozen_P2_boundary_only"),
            "endpoint_labels_role": "evaluation_only_promotion_gate",
            "promotion_gate": (
                "objective_geometry_component_Betti_nonregression_and_strict_"
                "labeled_edge_face_bridge_improvement"
            ),
            "selection_effect": "none",
            "held_out_tuning": "prohibited",
        },
        "exact_predicate_audit_contract": {
            "role": "readiness_audit_no_selection",
            "coordinate_model": "binary64_values_as_exact_rationals",
            "triangulation_source": "SciPy_Qhull_floating_point",
            "predicates": ["orientation_3", "in_sphere_3"],
            "construction_effect": "none",
            "selection_effect": "none",
            "claim_boundary": (
                "audits supplied connectivity only; not exact construction, an "
                "exact alpha complex, or a CGAL certificate"
            ),
            "promotion_gate": "blocked_until_exact_construction_backend_integrated",
        },
        "exact_construction_backend_contract": {
            "role": "optional_backend_handoff_validation_no_selection",
            "protocol_version": 1,
            "transport": "one_JSON_request_on_stdin_one_JSON_response_on_stdout",
            "request_binding": "SHA256_of_canonical_request_must_be_echoed",
            "coordinate_model": "binary64_values_as_exact_rationals",
            "response_connectivity": "tetrahedron_vertex_indices",
            "backend_attestation": "name_version_kernel_exact_construction_true",
            "host_validation": [
                "all_points_used_and_face_incidence",
                "exact_convex_hull_support_and_volume_coverage",
                "exact_orientation_and_in_sphere_predicates",
            ],
            "construction_effect": "validated_connectivity_not_applied",
            "selection_effect": "none",
            "fail_closed": "missing_failed_or_rejected_backend_blocks_promotion",
            "held_out_tuning": "prohibited",
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
        "bridge_penalty_ablation": {
            "enabled": bridge_penalty_ablation is not None,
            "source_split": (
                PanelSplit.CALIBRATION.value
                if bridge_penalty_ablation is not None
                else None
            ),
            "requested_strengths": (
                list(args.bridge_penalty_strengths)
                if args.evaluate_bridge_penalty
                else None
            ),
            "result": (
                bridge_penalty_ablation.to_dict()
                if bridge_penalty_ablation is not None
                else None
            ),
        },
        "boundary_owner_intervention": {
            "enabled": boundary_owner_intervention is not None,
            "source_split": (
                PanelSplit.CALIBRATION.value
                if boundary_owner_intervention is not None
                else None
            ),
            "requested_rounds": (
                list(args.boundary_intervention_rounds)
                if args.evaluate_boundary_intervention
                else None
            ),
            "result": (
                boundary_owner_intervention.to_dict()
                if boundary_owner_intervention is not None
                else None
            ),
        },
        "boundary_region_cut_ablation": {
            "enabled": boundary_region_cut_ablation is not None,
            "source_split": (
                PanelSplit.CALIBRATION.value
                if boundary_region_cut_ablation is not None
                else None
            ),
            "requested_strategies": (
                ["baseline", "largest_risk_region", "safe_backbone_cut"]
                if args.evaluate_boundary_region_cuts
                else None
            ),
            "result": (
                boundary_region_cut_ablation.to_dict()
                if boundary_region_cut_ablation is not None
                else None
            ),
        },
        "exact_predicate_audit": {
            "enabled": exact_predicate_audit is not None,
            "source_split": split.value if exact_predicate_audit is not None else None,
            "result": (
                exact_predicate_audit.to_dict()
                if exact_predicate_audit is not None
                else None
            ),
        },
        "exact_construction_backend": {
            "enabled": exact_construction_result is not None,
            "source_split": split.value
            if exact_construction_result is not None
            else None,
            "backend_executable_explicit": args.exact_backend is not None,
            "requested_timeout_seconds": args.exact_backend_timeout,
            "result": (
                exact_construction_result.to_dict()
                if exact_construction_result is not None
                else None
            ),
        },
        "config": asdict(config),
        "boundary_bridge_localization": {
            "enabled": boundary_bridge_localization is not None,
            "source_split": (
                split.value if boundary_bridge_localization is not None else None
            ),
            "result": (
                boundary_bridge_localization.to_dict()
                if boundary_bridge_localization is not None
                else None
            ),
        },
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

"""Numerical baselines and safety primitives for PFTF-alpha research."""

from .adaptive import (
    AdaptiveCellFiltration,
    LocalNeighborhoodGeometry,
    density_scaled_filtration,
    knn_scales,
    local_neighborhood_geometry,
    pca_anisotropic_filtration,
    pftf_confidence_fallback_filtration,
    pftf_local_metric_filtration,
)
from .baselines import (
    BaselineID,
    BaselineResult,
    BenchmarkConfig,
    CaseBenchmark,
    run_case_benchmarks,
)
from .calibration import (
    AdaptiveCalibrationResult,
    CalibrationPoint,
    P2ConfidenceCalibrationResult,
    calibrate_adaptive_multiplier,
    calibrate_p2_confidence_threshold,
)
from .conventions import (
    AlphaConvention,
    alpha_to_squared_radius,
    squared_radius_to_alpha,
)
from .filtration import (
    AlphaFiltration,
    BoundaryMode,
    ComplexStatistics,
    SimplexRecord,
)
from .geometry import Circumsphere, intrinsic_circumsphere
from .metrics import (
    LocalMetricField,
    SimplexMetricDecision,
    hard_alpha_gate,
    metric_circumradius_squared,
    soft_alpha_gate,
)
from .pftf import (
    PFTFRelationField,
    directed_scale_contrast,
    pftf_relation_field,
)
from .selection import (
    AlphaEvaluation,
    ObjectiveTerms,
    ObjectiveWeights,
    scan_critical_alphas,
    select_best_alpha,
)
from .surface import (
    MeshStatistics,
    SurfaceDistanceMetrics,
    SurfaceEndpointMetrics,
    SurfaceMesh,
    alpha_surface,
    convex_hull_surface,
    evaluate_surface,
    mesh_statistics,
    sample_triangle_mesh,
    surface_distance_metrics,
)
from .synthetic import (
    PanelSplit,
    SyntheticCase,
    SyntheticFamily,
    make_minimal_panel,
    make_synthetic_case,
)

__all__ = [
    "AdaptiveCalibrationResult",
    "P2ConfidenceCalibrationResult",
    "CalibrationPoint",
    "AdaptiveCellFiltration",
    "LocalNeighborhoodGeometry",
    "density_scaled_filtration",
    "knn_scales",
    "local_neighborhood_geometry",
    "pca_anisotropic_filtration",
    "pftf_confidence_fallback_filtration",
    "AlphaConvention",
    "pftf_local_metric_filtration",
    "AlphaEvaluation",
    "AlphaFiltration",
    "BoundaryMode",
    "Circumsphere",
    "ComplexStatistics",
    "LocalMetricField",
    "ObjectiveTerms",
    "PFTFRelationField",
    "ObjectiveWeights",
    "SimplexMetricDecision",
    "SimplexRecord",
    "alpha_to_squared_radius",
    "hard_alpha_gate",
    "directed_scale_contrast",
    "intrinsic_circumsphere",
    "metric_circumradius_squared",
    "scan_critical_alphas",
    "select_best_alpha",
    "soft_alpha_gate",
    "squared_radius_to_alpha",
    "BaselineID",
    "BaselineResult",
    "BenchmarkConfig",
    "CaseBenchmark",
    "MeshStatistics",
    "PanelSplit",
    "SurfaceDistanceMetrics",
    "SurfaceEndpointMetrics",
    "SurfaceMesh",
    "SyntheticCase",
    "SyntheticFamily",
    "alpha_surface",
    "calibrate_adaptive_multiplier",
    "calibrate_p2_confidence_threshold",
    "convex_hull_surface",
    "evaluate_surface",
    "make_minimal_panel",
    "make_synthetic_case",
    "mesh_statistics",
    "run_case_benchmarks",
    "pftf_relation_field",
    "sample_triangle_mesh",
    "surface_distance_metrics",
]

"""Run the first G0-G1 baseline on a tiny 2D point cloud."""

from __future__ import annotations

import numpy as np

from pftf_alpha import (
    AlphaFiltration,
    ObjectiveTerms,
    scan_critical_alphas,
    select_best_alpha,
)

points = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
)
filtration = AlphaFiltration.from_points(points)


def toy_objective(_filtration, _alpha_squared, statistics):
    """Require filled 2D cells, then prefer the smaller complex."""

    return ObjectiveTerms(
        geometry=0.0 if statistics.top_simplices else 1.0,
        topology=abs(statistics.connected_components - 1),
        complexity=0.01 * statistics.top_simplices,
    )


evaluations = scan_critical_alphas(filtration, toy_objective)
selected = select_best_alpha(evaluations)

print("critical alpha^2:", filtration.critical_values().tolist())
for row in evaluations:
    print(
        f"alpha^2={row.alpha_squared:.6g} "
        f"loss={row.total:.6g} "
        f"simplices={row.statistics.simplex_counts}"
    )
print("selected alpha^2:", selected.alpha_squared)

"""Observed-only local multi-scan support for ETH Gazebo reconstruction."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .eth_gazebo_reconstruction_protocol import (
    FUSION_VOXEL_METERS,
    REFERENCE_VOXEL_METERS,
    ROI_LOWER_QUANTILE,
    ROI_MARGIN_METERS,
    ROI_UPPER_QUANTILE,
    SOURCE_VOXEL_METERS,
)
from .eth_gazebo_reconstruction_shadow import (
    ReconstructionEndpoint,
    _alpha_surface,
    _endpoint,
    _split_source_points,
    _transform_and_crop,
    _voxel_downsample,
)
from .eth_gazebo_validation_protocol import SCAN_MEMBERS
from .eth_open3d_fgr_pipeline import _load_xyz

FloatArray = np.ndarray


@dataclass(frozen=True)
class SourceReconstructionInputs:
    source_index: int
    pair_count: int
    accepted_pair_count: int
    rejected_pair_count: int
    anchor_points: FloatArray
    reference_points: FloatArray
    target_points: FloatArray
    target_provenance: np.ndarray
    lower: FloatArray
    upper: FloatArray
    characteristic_length: float


@dataclass(frozen=True)
class LocalSupportRoute:
    anchor_cell_count: int
    target_cell_count: int
    overlapping_target_cell_count: int
    target_only_cell_count: int
    corroborated_target_only_cell_count: int
    rejected_target_only_cell_count: int
    mean_target_support: float
    mean_target_dispersion_meters: float
    scan_fused_points: FloatArray
    local_points: FloatArray


@dataclass(frozen=True)
class LocalSupportEndpoint:
    route: LocalSupportRoute
    endpoint: ReconstructionEndpoint


def load_hokuyo_scans(
    archive_path: str | Path,
) -> tuple[tuple[FloatArray, ...], tuple[str, ...]]:
    """Open exactly the frozen Hokuyo members and no registration label."""

    scans: list[FloatArray] = []
    opened: list[str] = []
    with zipfile.ZipFile(Path(archive_path)) as source:
        for member in SCAN_MEMBERS:
            opened.append(member)
            with source.open(member) as stream:
                scans.append(_load_xyz(stream))
    if tuple(opened) != SCAN_MEMBERS:
        raise RuntimeError("local-support evaluator opened an unexpected member set")
    return tuple(scans), tuple(opened)


def prepare_source_inputs(
    o3d: object,
    raw_scans: Sequence[FloatArray],
    downsampled_scans: Sequence[FloatArray],
    predictions: Sequence[Mapping[str, object]],
    accept_by_pair: Mapping[tuple[int, int], bool],
    *,
    source_index: int,
) -> SourceReconstructionInputs:
    """Create a source-view heldout case without using any reference for routing."""

    source_observed_raw, source_reference_raw = _split_source_points(
        raw_scans[source_index]
    )
    source_observed = _voxel_downsample(
        o3d,
        source_observed_raw,
        SOURCE_VOXEL_METERS,
    )
    reference = _voxel_downsample(
        o3d,
        source_reference_raw,
        REFERENCE_VOXEL_METERS,
    )
    lower = (
        np.quantile(source_observed, ROI_LOWER_QUANTILE, axis=0)
        - ROI_MARGIN_METERS
    )
    upper = (
        np.quantile(source_observed, ROI_UPPER_QUANTILE, axis=0)
        + ROI_MARGIN_METERS
    )
    inside = np.all(
        (source_observed >= lower) & (source_observed <= upper),
        axis=1,
    )
    anchor = np.ascontiguousarray(source_observed[inside])
    reference_inside = np.all((reference >= lower) & (reference <= upper), axis=1)
    reference = np.ascontiguousarray(reference[reference_inside])
    target_chunks: list[FloatArray] = []
    provenance_chunks: list[np.ndarray] = []
    accepted_count = 0
    for prediction in predictions:
        target_index = int(prediction["target_index"])
        pair = (source_index, target_index)
        if not accept_by_pair[pair]:
            continue
        transformed = _transform_and_crop(
            downsampled_scans[target_index],
            prediction["target_to_source_matrix"],
            lower,
            upper,
        )
        target_chunks.append(transformed)
        provenance_chunks.append(
            np.full(transformed.shape[0], target_index, dtype=np.int64)
        )
        accepted_count += 1
    if not target_chunks:
        raise ValueError("local-support source has no accepted target scans")
    characteristic_length = float(np.linalg.norm(upper - lower))
    return SourceReconstructionInputs(
        source_index=source_index,
        pair_count=len(predictions),
        accepted_pair_count=accepted_count,
        rejected_pair_count=len(predictions) - accepted_count,
        anchor_points=anchor,
        reference_points=reference,
        target_points=np.ascontiguousarray(np.vstack(target_chunks)),
        target_provenance=np.ascontiguousarray(np.concatenate(provenance_chunks)),
        lower=np.ascontiguousarray(lower),
        upper=np.ascontiguousarray(upper),
        characteristic_length=characteristic_length,
    )


def _cell_summary(
    points: FloatArray,
    lower: FloatArray,
) -> tuple[np.ndarray, np.ndarray, FloatArray, np.ndarray]:
    keys = np.floor((points - lower) / FUSION_VOXEL_METERS).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique_keys.shape[0]).astype(float)
    sums = np.stack(
        [
            np.bincount(
                inverse,
                weights=points[:, axis],
                minlength=unique_keys.shape[0],
            )
            for axis in range(3)
        ],
        axis=1,
    )
    centroids = sums / counts[:, None]
    return unique_keys, inverse, centroids, counts


def local_support_route(
    anchor_points: FloatArray,
    target_points: FloatArray,
    target_provenance: np.ndarray,
    lower: FloatArray,
    *,
    minimum_support: int,
    maximum_dispersion_meters: float,
) -> LocalSupportRoute:
    """Keep anchor cells and add only corroborated target-only spatial cells."""

    if minimum_support < 2:
        raise ValueError("minimum_support must be at least two scans")
    if maximum_dispersion_meters <= 0.0:
        raise ValueError("maximum_dispersion_meters must be positive")
    if target_provenance.shape != (target_points.shape[0],):
        raise ValueError("target provenance must align with target points")
    anchor_keys, _, anchor_centroids, _ = _cell_summary(anchor_points, lower)
    target_keys, inverse, target_centroids, counts = _cell_summary(
        target_points,
        lower,
    )
    cell_provenance = np.unique(
        np.column_stack((inverse, target_provenance)),
        axis=0,
    )
    support = np.bincount(
        cell_provenance[:, 0],
        minlength=target_keys.shape[0],
    )
    squared_norm = np.sum(target_points * target_points, axis=1)
    squared_sum = np.bincount(
        inverse,
        weights=squared_norm,
        minlength=target_keys.shape[0],
    )
    dispersion = np.sqrt(
        np.maximum(
            squared_sum / counts - np.sum(target_centroids * target_centroids, axis=1),
            0.0,
        )
    )
    anchor_key_set = {tuple(key) for key in anchor_keys.tolist()}
    target_only = np.asarray(
        [tuple(key) not in anchor_key_set for key in target_keys.tolist()],
        dtype=bool,
    )
    corroborated = (
        target_only
        & (support >= minimum_support)
        & (dispersion <= maximum_dispersion_meters)
    )
    scan_fused = np.ascontiguousarray(
        np.vstack((anchor_centroids, target_centroids[target_only]))
    )
    local = np.ascontiguousarray(
        np.vstack((anchor_centroids, target_centroids[corroborated]))
    )
    return LocalSupportRoute(
        anchor_cell_count=int(anchor_keys.shape[0]),
        target_cell_count=int(target_keys.shape[0]),
        overlapping_target_cell_count=int(np.count_nonzero(~target_only)),
        target_only_cell_count=int(np.count_nonzero(target_only)),
        corroborated_target_only_cell_count=int(np.count_nonzero(corroborated)),
        rejected_target_only_cell_count=int(
            np.count_nonzero(target_only & ~corroborated)
        ),
        mean_target_support=float(np.mean(support)),
        mean_target_dispersion_meters=float(np.mean(dispersion)),
        scan_fused_points=scan_fused,
        local_points=local,
    )


def evaluate_local_support(
    o3d: object,
    inputs: SourceReconstructionInputs,
    *,
    minimum_support: int,
    maximum_dispersion_meters: float,
    seed: int,
) -> LocalSupportEndpoint:
    route = local_support_route(
        inputs.anchor_points,
        inputs.target_points,
        inputs.target_provenance,
        inputs.lower,
        minimum_support=minimum_support,
        maximum_dispersion_meters=maximum_dispersion_meters,
    )
    endpoint = _endpoint(
        _alpha_surface(o3d, route.local_points),
        inputs.reference_points,
        characteristic_length=inputs.characteristic_length,
        seed=seed,
    )
    return LocalSupportEndpoint(route=route, endpoint=endpoint)


def evaluate_anchor_and_scan_baselines(
    o3d: object,
    inputs: SourceReconstructionInputs,
    *,
    seed: int,
) -> tuple[ReconstructionEndpoint, ReconstructionEndpoint, LocalSupportRoute]:
    route = local_support_route(
        inputs.anchor_points,
        inputs.target_points,
        inputs.target_provenance,
        inputs.lower,
        minimum_support=2,
        maximum_dispersion_meters=np.finfo(float).max,
    )
    anchor = _endpoint(
        _alpha_surface(o3d, route.scan_fused_points[: route.anchor_cell_count]),
        inputs.reference_points,
        characteristic_length=inputs.characteristic_length,
        seed=seed,
    )
    scan = _endpoint(
        _alpha_surface(o3d, route.scan_fused_points),
        inputs.reference_points,
        characteristic_length=inputs.characteristic_length,
        seed=seed,
    )
    return anchor, scan, route

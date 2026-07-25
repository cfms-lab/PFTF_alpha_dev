import struct

import numpy as np

from pftf_alpha.real_heldout import (
    _normalized,
    load_mesh,
    make_real_case,
)
from pftf_alpha.surface import SurfaceMesh, mesh_statistics


def _write_binary_stl(path, triangles) -> None:
    with open(path, "wb") as handle:
        handle.write(b"\x00" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            handle.write(struct.pack("<3f", 0.0, 0.0, 0.0))
            for vertex in tri:
                handle.write(struct.pack("<3f", *vertex))
            handle.write(struct.pack("<H", 0))


def _tetra_triangles():
    v = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
    return [
        (v[0], v[1], v[2]),
        (v[0], v[1], v[3]),
        (v[0], v[2], v[3]),
        (v[1], v[2], v[3]),
    ]


def test_load_binary_stl(tmp_path) -> None:
    path = tmp_path / "tetra.stl"
    _write_binary_stl(path, _tetra_triangles())
    mesh = load_mesh(path)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape[0] == 4
    assert mesh_statistics(mesh).connected_components == 1


def test_load_npz(tmp_path) -> None:
    path = tmp_path / "tetra.npz"
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    facets = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    np.savez(path, vertices=vertices, facets=facets)
    mesh = load_mesh(path)
    assert mesh.vertices.shape == (4, 3)
    assert mesh.faces.shape[0] == 4


def test_normalization_unit_diagonal() -> None:
    mesh = SurfaceMesh(
        vertices=np.array(
            [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64
        ),
        faces=np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
        ),
    )
    normalized = _normalized(mesh)
    extent = normalized.vertices.max(axis=0) - normalized.vertices.min(axis=0)
    assert np.isclose(np.linalg.norm(extent), 1.0)


def test_make_real_case_is_deterministic() -> None:
    mesh = SurfaceMesh(
        vertices=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
        ),
        faces=np.array(
            [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64
        ),
    )
    first = make_real_case(
        mesh, "t", observed_count=200, reference_count=800,
        noise_fraction=0.005, seed=3,
    )
    second = make_real_case(
        mesh, "t", observed_count=200, reference_count=800,
        noise_fraction=0.005, seed=3,
    )
    np.testing.assert_array_equal(first.points, second.points)
    assert first.points.shape == (200, 3)
    assert first.characteristic_length == 1.0

"""Verified, dependency-light intake for Open3D's DemoICPPointClouds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import urllib.request
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DATASET_URL = (
    "https://github.com/isl-org/open3d_downloads/releases/download/"
    "20220301-data/DemoICPPointClouds.zip"
)
ARCHIVE_NAME = "DemoICPPointClouds.zip"
EXPECTED_ARCHIVE_MD5 = "596cffe5f9c587045e7397ad70754de9"
EXPECTED_ARCHIVE_SHA256 = (
    "b94e0146c1d48c5edfc11af71b4af39ffca604485668c55a127c3b43203a6bd5"
)
EXPECTED_MEMBERS = (
    "cloud_bin_0.pcd",
    "cloud_bin_1.pcd",
    "cloud_bin_2.pcd",
    "init.log",
)
EXPECTED_PCD_FIELDS = (
    "x",
    "y",
    "z",
    "rgb",
    "normal_x",
    "normal_y",
    "normal_z",
    "curvature",
)


@dataclass(frozen=True)
class DemoICPArchiveVerification:
    archive_path: str
    byte_count: int
    md5: str
    sha256: str
    members: tuple[str, ...]
    verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransformationLogEntry:
    source_index: int
    target_index: int
    information_count: int
    logged_matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.logged_matrix, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("logged_matrix must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
            raise ValueError("logged_matrix must be an affine transform")
        if not math.isfinite(float(np.linalg.det(matrix[:3, :3]))):
            raise ValueError("logged_matrix rotation block must be finite")
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1.0e-12:
            raise ValueError("logged_matrix must be invertible")

    @property
    def source_to_target_matrix(self) -> FloatArray:
        """Return the frame-i to frame-j transform for an ``i j`` log entry."""

        return np.linalg.inv(np.asarray(self.logged_matrix, dtype=np.float64))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "information_count": self.information_count,
            "logged_matrix": self.logged_matrix.tolist(),
            "source_to_target_matrix": self.source_to_target_matrix.tolist(),
        }


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_demo_icp_archive(
    archive_path: str | Path,
) -> DemoICPArchiveVerification:
    """Verify the frozen official archive and its exact member list."""

    resolved = Path(archive_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    md5 = _hash_file(resolved, "md5")
    sha256 = _hash_file(resolved, "sha256")
    if md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"DemoICP archive MD5 mismatch: {md5}")
    if sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError(f"DemoICP archive SHA-256 mismatch: {sha256}")
    with zipfile.ZipFile(resolved) as archive:
        members = tuple(info.filename for info in archive.infolist())
        if set(members) != set(EXPECTED_MEMBERS) or len(members) != len(
            EXPECTED_MEMBERS
        ):
            raise ValueError(f"unexpected DemoICP archive members: {members}")
        unsafe_member = any(
            info.is_dir() or Path(info.filename).name != info.filename
            for info in archive.infolist()
        )
        if unsafe_member:
            raise ValueError("DemoICP archive contains a directory or unsafe path")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"DemoICP archive CRC failure: {bad_member}")
    return DemoICPArchiveVerification(
        archive_path=str(resolved),
        byte_count=resolved.stat().st_size,
        md5=md5,
        sha256=sha256,
        members=members,
        verified=True,
    )


def fetch_demo_icp_archive(data_root: str | Path) -> Path:
    """Download the official archive only when no local archive exists."""

    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / ARCHIVE_NAME
    if archive_path.exists():
        verify_demo_icp_archive(archive_path)
        return archive_path
    partial = archive_path.with_suffix(archive_path.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"stale partial download requires review: {partial}")
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=60) as response:  # noqa: S310
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        verify_demo_icp_archive(partial)
        os.replace(partial, archive_path)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return archive_path


def extract_demo_icp_archive(
    archive_path: str | Path,
    output_dir: str | Path,
) -> DemoICPArchiveVerification:
    """Extract only the four verified, top-level dataset members."""

    resolved = Path(archive_path)
    verification = verify_demo_icp_archive(resolved)
    target_root = Path(output_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(resolved) as archive:
        for name in EXPECTED_MEMBERS:
            info = archive.getinfo(name)
            target = target_root / name
            if target.exists():
                payload = target.read_bytes()
                if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                    raise ValueError(f"existing extracted member differs: {target}")
                continue
            target.write_bytes(archive.read(name))
    return verification


def _parse_pcd_header(stream: object) -> dict[str, tuple[str, ...]]:
    header: dict[str, tuple[str, ...]] = {}
    total_bytes = 0
    while True:
        line = stream.readline(4097)  # type: ignore[attr-defined]
        if not line:
            raise ValueError("PCD header ended before DATA")
        if len(line) > 4096:
            raise ValueError("PCD header line is too long")
        total_bytes += len(line)
        if total_bytes > 65536:
            raise ValueError("PCD header is too large")
        try:
            text = line.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PCD header must be ASCII") from error
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        key = parts[0].upper()
        if key in header:
            raise ValueError(f"duplicate PCD header field: {key}")
        header[key] = tuple(parts[1:])
        if key == "DATA":
            return header


def read_binary_pcd_xyz(path: str | Path) -> FloatArray:
    """Read XYZ coordinates from the exact binary PCD schema in DemoICP."""

    resolved = Path(path)
    with resolved.open("rb") as stream:
        header = _parse_pcd_header(stream)
        fields = header.get("FIELDS")
        if fields != EXPECTED_PCD_FIELDS:
            raise ValueError(f"unsupported PCD fields: {fields}")
        field_count = len(EXPECTED_PCD_FIELDS)
        if header.get("SIZE") != ("4",) * field_count:
            raise ValueError("DemoICP PCD SIZE must be float32 for every field")
        if header.get("TYPE") != ("F",) * field_count:
            raise ValueError("DemoICP PCD TYPE must be floating point")
        if header.get("COUNT") != ("1",) * field_count:
            raise ValueError("DemoICP PCD COUNT must be one for every field")
        if header.get("DATA") != ("binary",):
            raise ValueError("only binary DemoICP PCD files are supported")
        try:
            width = int(header["WIDTH"][0])
            height = int(header["HEIGHT"][0])
            point_count = int(header["POINTS"][0])
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("invalid PCD dimensions") from error
        if width <= 0 or height <= 0 or width * height != point_count:
            raise ValueError("PCD WIDTH, HEIGHT, and POINTS disagree")
        payload = stream.read()
    expected_bytes = point_count * field_count * 4
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PCD payload length mismatch: {len(payload)} != {expected_bytes}"
        )
    rows = np.frombuffer(payload, dtype="<f4").reshape(point_count, field_count)
    points = rows[:, :3].astype(np.float64)
    if not np.all(np.isfinite(points)):
        raise ValueError("PCD XYZ coordinates must be finite")
    return np.ascontiguousarray(points)


def read_transformation_log(path: str | Path) -> tuple[TransformationLogEntry, ...]:
    """Parse Open3D's pair headers and 4x4 transformation matrices."""

    text = Path(path).read_text(encoding="ascii")
    lines = [line.strip() for line in text.splitlines()]
    rows = [line for line in lines if line and not line.startswith("#")]
    entries: list[TransformationLogEntry] = []
    index = 0
    while index < len(rows):
        header = rows[index].split()
        if len(header) != 3:
            raise ValueError(f"invalid transformation-log header: {rows[index]}")
        try:
            source_index, target_index, information_count = map(int, header)
        except ValueError as error:
            raise ValueError(
                "transformation-log header must contain integers"
            ) from error
        if index + 4 >= len(rows):
            raise ValueError("transformation log ended inside a matrix")
        try:
            matrix_rows = [
                [float(value) for value in rows[index + offset].split()]
                for offset in range(1, 5)
            ]
            matrix = np.asarray(matrix_rows, dtype=np.float64)
        except ValueError as error:
            raise ValueError("transformation-log matrix must be numeric") from error
        if matrix.shape != (4, 4):
            raise ValueError("transformation-log matrix must be 4x4")
        if source_index < 0 or target_index < 0 or information_count < 0:
            raise ValueError(
                "transformation-log indices and count must be non-negative"
            )
        entries.append(
            TransformationLogEntry(
                source_index=source_index,
                target_index=target_index,
                information_count=information_count,
                logged_matrix=matrix,
            )
        )
        index += 5
    if not entries:
        raise ValueError("transformation log contains no entries")
    return tuple(entries)


def transform_points(points: FloatArray, matrix: FloatArray) -> FloatArray:
    """Apply a finite affine 4x4 transform to an ``(n, 3)`` point array."""

    selected = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if selected.ndim != 2 or selected.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("matrix must be finite and 4x4")
    result = selected @ transform[:3, :3].T + transform[:3, 3]
    if not np.all(np.isfinite(result)):
        raise ValueError("transformed points must be finite")
    return np.ascontiguousarray(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark-data/open3d_demo_icp"),
    )
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    archive_path = args.data_root / ARCHIVE_NAME
    if args.download:
        archive_path = fetch_demo_icp_archive(args.data_root)
    verification = extract_demo_icp_archive(archive_path, args.data_root)
    payload = verification.to_dict()
    payload["extracted_root"] = str(args.data_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

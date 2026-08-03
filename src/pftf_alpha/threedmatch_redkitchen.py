"""Verified intake for the 3DMatch redkitchen registration benchmark."""

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
from pathlib import Path, PurePosixPath

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

SCENE_NAME = "7-scenes-redkitchen"
EVALUATION_NAME = f"{SCENE_NAME}-evaluation"
FRAGMENT_ARCHIVE_NAME = f"{SCENE_NAME}.zip"
EVALUATION_ARCHIVE_NAME = f"{EVALUATION_NAME}.zip"
FRAGMENT_URL = (
    "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/"
    f"scene-fragments/{FRAGMENT_ARCHIVE_NAME}"
)
EVALUATION_URL = (
    "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/"
    f"scene-fragments/{EVALUATION_ARCHIVE_NAME}"
)
FRAGMENT_ARCHIVE_MD5 = "2b7ba86ec2a370a9a5b989b63cc515f2"
FRAGMENT_ARCHIVE_SHA256 = (
    "7cb9a1c9236e6833e910692b1d3f572b970c3fc3493e7641c28f1a45841fa51c"
)
EVALUATION_ARCHIVE_MD5 = "20dc1a06956e01886f378109eb6df3bc"
EVALUATION_ARCHIVE_SHA256 = (
    "ff3eaa243025a0cdf6dd1ca5364a726acf7c08b36444e49c685e1f014bc4f16e"
)
FRAGMENT_COUNT = 60


@dataclass(frozen=True)
class ThreeDMatchArchiveVerification:
    role: str
    archive_path: str
    byte_count: int
    md5: str
    sha256: str
    file_count: int
    verified: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RegistrationLogEntry:
    source_index: int
    target_index: int
    fragment_count: int
    source_to_target_matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.source_to_target_matrix, dtype=np.float64)
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError("registration indices must be non-negative")
        if self.source_index == self.target_index:
            raise ValueError("registration pair indices must differ")
        if self.fragment_count <= max(self.source_index, self.target_index):
            raise ValueError("fragment_count does not cover the pair indices")
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("registration matrix must be finite and 4x4")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-12):
            raise ValueError("registration matrix must be affine")
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1.0e-12:
            raise ValueError("registration matrix must be invertible")

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source_index, self.target_index)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_index": self.source_index,
            "target_index": self.target_index,
            "fragment_count": self.fragment_count,
            "source_to_target_matrix": self.source_to_target_matrix.tolist(),
        }


@dataclass(frozen=True)
class RegistrationInfoEntry:
    source_index: int
    target_index: int
    fragment_count: int
    information_matrix: FloatArray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.information_matrix, dtype=np.float64)
        if self.source_index < 0 or self.target_index < 0:
            raise ValueError("information indices must be non-negative")
        if self.fragment_count <= max(self.source_index, self.target_index):
            raise ValueError("fragment_count does not cover information indices")
        if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
            raise ValueError("information matrix must be finite and 6x6")
        if matrix[0, 0] <= 0.0:
            raise ValueError("information normalization must be positive")

    @property
    def pair(self) -> tuple[int, int]:
        return (self.source_index, self.target_index)


def _fragment_members() -> tuple[str, ...]:
    return (f"{SCENE_NAME}/",) + tuple(
        f"{SCENE_NAME}/cloud_bin_{index}.ply"
        for index in range(FRAGMENT_COUNT)
    )


def _evaluation_members() -> tuple[str, ...]:
    return (
        f"{EVALUATION_NAME}/",
        f"{EVALUATION_NAME}/3dmatch.log",
        f"{EVALUATION_NAME}/gt.info",
        f"{EVALUATION_NAME}/gt.log",
    )


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_redkitchen_archive(
    archive_path: str | Path,
    *,
    role: str,
) -> ThreeDMatchArchiveVerification:
    """Verify one frozen archive, its hashes, CRCs, and exact allowlist."""

    resolved = Path(archive_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if role == "fragments":
        expected_md5 = FRAGMENT_ARCHIVE_MD5
        expected_sha256 = FRAGMENT_ARCHIVE_SHA256
        expected_members = _fragment_members()
    elif role == "evaluation":
        expected_md5 = EVALUATION_ARCHIVE_MD5
        expected_sha256 = EVALUATION_ARCHIVE_SHA256
        expected_members = _evaluation_members()
    else:
        raise ValueError("role must be fragments or evaluation")
    md5 = _hash_file(resolved, "md5")
    sha256 = _hash_file(resolved, "sha256")
    if md5 != expected_md5:
        raise ValueError(f"{role} archive MD5 mismatch: {md5}")
    if sha256 != expected_sha256:
        raise ValueError(f"{role} archive SHA-256 mismatch: {sha256}")
    with zipfile.ZipFile(resolved) as archive:
        infos = archive.infolist()
        members = tuple(info.filename for info in infos)
        if set(members) != set(expected_members) or len(members) != len(
            expected_members
        ):
            raise ValueError(f"unexpected {role} archive members")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe {role} archive path: {info.filename}")
            if info.is_dir() != info.filename.endswith("/"):
                raise ValueError(f"ambiguous archive member: {info.filename}")
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"{role} archive CRC failure: {bad_member}")
    return ThreeDMatchArchiveVerification(
        role=role,
        archive_path=str(resolved),
        byte_count=resolved.stat().st_size,
        md5=md5,
        sha256=sha256,
        file_count=sum(not info.is_dir() for info in infos),
        verified=True,
    )


def _fetch(url: str, target: Path, *, role: str) -> Path:
    if target.exists():
        verify_redkitchen_archive(target, role=role)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"stale partial download requires review: {partial}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        verify_redkitchen_archive(partial, role=role)
        os.replace(partial, target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target


def fetch_redkitchen_archives(data_root: str | Path) -> tuple[Path, Path]:
    """Fetch both official archives, reusing only verified local copies."""

    root = Path(data_root)
    fragments = _fetch(
        FRAGMENT_URL,
        root / FRAGMENT_ARCHIVE_NAME,
        role="fragments",
    )
    evaluation = _fetch(
        EVALUATION_URL,
        root / EVALUATION_ARCHIVE_NAME,
        role="evaluation",
    )
    return fragments, evaluation


def _extract_archive(
    archive_path: Path,
    output_root: Path,
    *,
    role: str,
) -> ThreeDMatchArchiveVerification:
    verification = verify_redkitchen_archive(archive_path, role=role)
    output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            target = output_root.joinpath(*relative.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                payload = target.read_bytes()
                if len(payload) != info.file_size or zlib.crc32(payload) != info.CRC:
                    raise ValueError(f"existing extracted member differs: {target}")
                continue
            target.write_bytes(archive.read(info.filename))
    return verification


def prepare_redkitchen_data(
    data_root: str | Path,
) -> tuple[ThreeDMatchArchiveVerification, ThreeDMatchArchiveVerification]:
    """Verify and safely extract both locally present archives."""

    root = Path(data_root)
    fragment_verification = _extract_archive(
        root / FRAGMENT_ARCHIVE_NAME,
        root,
        role="fragments",
    )
    evaluation_verification = _extract_archive(
        root / EVALUATION_ARCHIVE_NAME,
        root,
        role="evaluation",
    )
    return fragment_verification, evaluation_verification


def read_binary_ply_xyz(path: str | Path) -> FloatArray:
    """Read the exact binary little-endian XYZ-only fragment PLY schema."""

    resolved = Path(path)
    with resolved.open("rb") as stream:
        first = stream.readline(4097)
        if first != b"ply\n" and first != b"ply\r\n":
            raise ValueError("PLY must begin with the ply magic line")
        format_seen = False
        vertex_count: int | None = None
        properties: list[tuple[str, str]] = []
        current_element: str | None = None
        header_bytes = len(first)
        while True:
            line = stream.readline(4097)
            if not line:
                raise ValueError("PLY header ended before end_header")
            if len(line) > 4096:
                raise ValueError("PLY header line is too long")
            header_bytes += len(line)
            if header_bytes > 65536:
                raise ValueError("PLY header is too large")
            try:
                text = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError("PLY header must be ASCII") from error
            parts = text.split()
            if not parts or parts[0] in {"comment", "obj_info"}:
                continue
            if parts[0] == "format":
                format_seen = parts == ["format", "binary_little_endian", "1.0"]
            elif parts[0] == "element":
                if len(parts) != 3:
                    raise ValueError("invalid PLY element declaration")
                current_element = parts[1]
                if current_element != "vertex":
                    raise ValueError("only the vertex element is supported")
                vertex_count = int(parts[2])
            elif parts[0] == "property":
                if current_element != "vertex" or len(parts) != 3:
                    raise ValueError("invalid PLY property declaration")
                properties.append((parts[1], parts[2]))
            elif parts[0] == "end_header":
                break
            else:
                raise ValueError(f"unsupported PLY header declaration: {parts[0]}")
        payload = stream.read()
    if not format_seen:
        raise ValueError("PLY must use binary_little_endian 1.0")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY vertex count must be positive")
    if properties != [("float", "x"), ("float", "y"), ("float", "z")]:
        raise ValueError(f"unsupported PLY vertex properties: {properties}")
    expected_bytes = vertex_count * 3 * 4
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PLY payload length mismatch: {len(payload)} != {expected_bytes}"
        )
    points = np.frombuffer(payload, dtype="<f4").reshape(vertex_count, 3)
    result = points.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("PLY coordinates must be finite")
    return np.ascontiguousarray(result)


def _nonempty_text_lines(payload: str) -> list[str]:
    lines = payload.splitlines()
    return [line.strip() for line in lines if line.strip()]


def parse_registration_log(payload: str) -> tuple[RegistrationLogEntry, ...]:
    """Parse a 3DMatch/ElasticReconstruction five-line registration log."""

    lines = _nonempty_text_lines(payload)
    if len(lines) % 5 != 0:
        raise ValueError("registration log must contain five-line records")
    entries = []
    for offset in range(0, len(lines), 5):
        try:
            source_index, target_index, fragment_count = map(
                int,
                lines[offset].split(),
            )
            matrix = np.asarray(
                [
                    [float(value) for value in lines[offset + row].split()]
                    for row in range(1, 5)
                ],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ValueError("invalid registration-log record") from error
        entries.append(
            RegistrationLogEntry(
                source_index=source_index,
                target_index=target_index,
                fragment_count=fragment_count,
                source_to_target_matrix=matrix,
            )
        )
    pairs = [entry.pair for entry in entries]
    if len(set(pairs)) != len(pairs):
        raise ValueError("registration log contains duplicate pairs")
    return tuple(entries)


def read_registration_log(path: str | Path) -> tuple[RegistrationLogEntry, ...]:
    """Read a 3DMatch/ElasticReconstruction five-line registration log."""

    return parse_registration_log(Path(path).read_text(encoding="ascii"))


def parse_registration_info(payload: str) -> tuple[RegistrationInfoEntry, ...]:
    """Parse seven-line pair headers and 6x6 information matrices."""

    lines = _nonempty_text_lines(payload)
    if len(lines) % 7 != 0:
        raise ValueError("registration info must contain seven-line records")
    entries = []
    for offset in range(0, len(lines), 7):
        try:
            source_index, target_index, fragment_count = map(
                int,
                lines[offset].split(),
            )
            matrix = np.asarray(
                [
                    [float(value) for value in lines[offset + row].split()]
                    for row in range(1, 7)
                ],
                dtype=np.float64,
            )
        except ValueError as error:
            raise ValueError("invalid registration-info record") from error
        entries.append(
            RegistrationInfoEntry(
                source_index=source_index,
                target_index=target_index,
                fragment_count=fragment_count,
                information_matrix=matrix,
            )
        )
    pairs = [entry.pair for entry in entries]
    if len(set(pairs)) != len(pairs):
        raise ValueError("registration info contains duplicate pairs")
    return tuple(entries)


def read_registration_info(path: str | Path) -> tuple[RegistrationInfoEntry, ...]:
    """Read seven-line pair headers and 6x6 information matrices."""

    return parse_registration_info(Path(path).read_text(encoding="ascii"))


def official_transformation_error(
    ground_truth: RegistrationLogEntry,
    prediction: RegistrationLogEntry,
    information: RegistrationInfoEntry,
) -> float:
    """Reproduce mrEvaluateRegistration's normalized quadratic error."""

    if ground_truth.pair != prediction.pair or ground_truth.pair != information.pair:
        raise ValueError("ground truth, prediction, and information pairs disagree")
    relative = np.linalg.inv(ground_truth.source_to_target_matrix) @ (
        prediction.source_to_target_matrix
    )
    rotation = relative[:3, :3]
    trace_term = max(1.0 + float(np.trace(rotation)), 0.0)
    scalar = 0.5 * math.sqrt(trace_term)
    if scalar > 1.0e-12:
        quaternion_vector = np.asarray(
            (
                (rotation[2, 1] - rotation[1, 2]) / (4.0 * scalar),
                (rotation[0, 2] - rotation[2, 0]) / (4.0 * scalar),
                (rotation[1, 0] - rotation[0, 1]) / (4.0 * scalar),
            ),
            dtype=np.float64,
        )
    else:
        from scipy.spatial.transform import Rotation

        quaternion = Rotation.from_matrix(rotation).as_quat()
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        quaternion_vector = quaternion[:3]
    error_vector = np.concatenate((relative[:3, 3], quaternion_vector))
    matrix = information.information_matrix
    error = float(error_vector @ matrix @ error_vector / matrix[0, 0])
    if not math.isfinite(error) or error < -1.0e-12:
        raise ValueError("official transformation error must be finite")
    return max(error, 0.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("benchmark-data/3dmatch_redkitchen"),
    )
    parser.add_argument("--download", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.download:
        fetch_redkitchen_archives(args.data_root)
    fragments, evaluation = prepare_redkitchen_data(args.data_root)
    print(
        json.dumps(
            {
                "fragments": fragments.to_dict(),
                "evaluation": evaluation.to_dict(),
                "data_root": str(args.data_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

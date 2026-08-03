"""Verified archive intake for a frozen 3DMatch benchmark scene."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from .threedmatch_redkitchen import ThreeDMatchArchiveVerification


@dataclass(frozen=True)
class ThreeDMatchSceneSpec:
    """Immutable identity and exact archive contract for one scene."""

    scene_name: str
    evaluation_name: str
    fragment_archive_name: str
    evaluation_archive_name: str
    fragment_url: str
    evaluation_url: str
    fragment_archive_md5: str
    fragment_archive_sha256: str
    evaluation_archive_md5: str
    evaluation_archive_sha256: str
    fragment_count: int
    dataset_source: str
    dataset_license_boundary: str

    def __post_init__(self) -> None:
        text_fields = (
            self.scene_name,
            self.evaluation_name,
            self.fragment_archive_name,
            self.evaluation_archive_name,
            self.fragment_url,
            self.evaluation_url,
            self.dataset_source,
            self.dataset_license_boundary,
        )
        if any(not value.strip() for value in text_fields):
            raise ValueError("scene specification text fields must be non-empty")
        if self.fragment_count <= 0:
            raise ValueError("fragment_count must be positive")
        for name, value, length in (
            ("fragment MD5", self.fragment_archive_md5, 32),
            ("fragment SHA-256", self.fragment_archive_sha256, 64),
            ("evaluation MD5", self.evaluation_archive_md5, 32),
            ("evaluation SHA-256", self.evaluation_archive_sha256, 64),
        ):
            if len(value) != length or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def expected_members(self, role: str) -> tuple[str, ...]:
        if role == "fragments":
            return (f"{self.scene_name}/",) + tuple(
                f"{self.scene_name}/cloud_bin_{index}.ply"
                for index in range(self.fragment_count)
            )
        if role == "evaluation":
            return (
                f"{self.evaluation_name}/",
                f"{self.evaluation_name}/3dmatch.log",
                f"{self.evaluation_name}/gt.info",
                f"{self.evaluation_name}/gt.log",
            )
        raise ValueError("role must be fragments or evaluation")

    def archive_identity(self, role: str) -> tuple[str, str, str, str]:
        if role == "fragments":
            return (
                self.fragment_archive_name,
                self.fragment_url,
                self.fragment_archive_md5,
                self.fragment_archive_sha256,
            )
        if role == "evaluation":
            return (
                self.evaluation_archive_name,
                self.evaluation_url,
                self.evaluation_archive_md5,
                self.evaluation_archive_sha256,
            )
        raise ValueError("role must be fragments or evaluation")


MARYLAND_HOTEL3_SPEC = ThreeDMatchSceneSpec(
    scene_name="sun3d-hotel_umd-maryland_hotel3",
    evaluation_name="sun3d-hotel_umd-maryland_hotel3-evaluation",
    fragment_archive_name="sun3d-hotel_umd-maryland_hotel3.zip",
    evaluation_archive_name=(
        "sun3d-hotel_umd-maryland_hotel3-evaluation.zip"
    ),
    fragment_url=(
        "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/"
        "scene-fragments/sun3d-hotel_umd-maryland_hotel3.zip"
    ),
    evaluation_url=(
        "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/"
        "scene-fragments/"
        "sun3d-hotel_umd-maryland_hotel3-evaluation.zip"
    ),
    fragment_archive_md5="6edd29d020fa164939141ad218973aba",
    fragment_archive_sha256=(
        "2dd600fad0cfd98968b9ff1684430f3647f241b3690004154c56d1c058c6f5bc"
    ),
    evaluation_archive_md5="ad9128fae730de3c29ccf76575577a7a",
    evaluation_archive_sha256=(
        "180bf3749c7353f5e0a0a17220f9760944ed86613aaf787ab1e841e7007912b7"
    ),
    fragment_count=37,
    dataset_source="SUN3D via the official 3DMatch benchmark",
    dataset_license_boundary=(
        "the accessed 3DMatch and SUN3D pages request dataset citation but do "
        "not state an explicit SUN3D data license; do not redistribute archives"
    ),
)


def _hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_threedmatch_scene_archive(
    archive_path: str | Path,
    scene: ThreeDMatchSceneSpec,
    *,
    role: str,
) -> ThreeDMatchArchiveVerification:
    """Verify hashes, CRCs, path safety, and the exact member allowlist."""

    resolved = Path(archive_path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    _, _, expected_md5, expected_sha256 = scene.archive_identity(role)
    expected_members = scene.expected_members(role)
    md5 = _hash_file(resolved, "md5")
    sha256 = _hash_file(resolved, "sha256")
    if md5 != expected_md5:
        raise ValueError(f"{scene.scene_name} {role} archive MD5 mismatch: {md5}")
    if sha256 != expected_sha256:
        raise ValueError(
            f"{scene.scene_name} {role} archive SHA-256 mismatch: {sha256}"
        )
    with zipfile.ZipFile(resolved) as archive:
        infos = archive.infolist()
        members = tuple(info.filename for info in infos)
        if set(members) != set(expected_members) or len(members) != len(
            expected_members
        ):
            raise ValueError(
                f"unexpected {scene.scene_name} {role} archive members"
            )
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


def _fetch(
    url: str,
    target: Path,
    scene: ThreeDMatchSceneSpec,
    *,
    role: str,
) -> Path:
    if target.exists():
        verify_threedmatch_scene_archive(target, scene, role=role)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        raise FileExistsError(f"stale partial download requires review: {partial}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            with partial.open("wb") as output:
                shutil.copyfileobj(response, output)
        verify_threedmatch_scene_archive(partial, scene, role=role)
        os.replace(partial, target)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise
    return target


def fetch_threedmatch_scene_archives(
    data_root: str | Path,
    scene: ThreeDMatchSceneSpec,
) -> tuple[Path, Path]:
    """Fetch both official archives, reusing only verified copies."""

    root = Path(data_root)
    fragment_name, fragment_url, _, _ = scene.archive_identity("fragments")
    evaluation_name, evaluation_url, _, _ = scene.archive_identity("evaluation")
    fragments = _fetch(
        fragment_url,
        root / fragment_name,
        scene,
        role="fragments",
    )
    evaluation = _fetch(
        evaluation_url,
        root / evaluation_name,
        scene,
        role="evaluation",
    )
    return fragments, evaluation


def _extract_archive(
    archive_path: Path,
    output_root: Path,
    scene: ThreeDMatchSceneSpec,
    *,
    role: str,
) -> ThreeDMatchArchiveVerification:
    verification = verify_threedmatch_scene_archive(
        archive_path,
        scene,
        role=role,
    )
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
                crc = zlib.crc32(payload) & 0xFFFFFFFF
                if len(payload) != info.file_size or crc != info.CRC:
                    raise ValueError(f"existing extracted member differs: {target}")
                continue
            target.write_bytes(archive.read(info.filename))
    return verification


def prepare_threedmatch_scene_data(
    data_root: str | Path,
    scene: ThreeDMatchSceneSpec,
) -> tuple[ThreeDMatchArchiveVerification, ThreeDMatchArchiveVerification]:
    """Verify and safely extract both locally present scene archives."""

    root = Path(data_root)
    fragment_name, _, _, _ = scene.archive_identity("fragments")
    evaluation_name, _, _, _ = scene.archive_identity("evaluation")
    fragments = _extract_archive(
        root / fragment_name,
        root,
        scene,
        role="fragments",
    )
    evaluation = _extract_archive(
        root / evaluation_name,
        root,
        scene,
        role="evaluation",
    )
    return fragments, evaluation

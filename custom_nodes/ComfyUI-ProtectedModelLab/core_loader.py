from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.hashing import sha256_file
from core_format.spcore import ProtectedCore, read_core


@dataclass(frozen=True)
class ProtectedIdentity:
    carrier_path: Path
    carrier_sha256: str
    architecture: str


@dataclass(frozen=True)
class ProtectedCoreHandle:
    path: Path
    core: ProtectedCore


def load_bound_core(path: str | Path, identity: ProtectedIdentity) -> ProtectedCoreHandle:
    core_path = Path(path)
    core = read_core(core_path, identity.carrier_sha256)
    if core.architecture != identity.architecture:
        raise ValueError(f"architecture mismatch: core={core.architecture}, carrier={identity.architecture}")
    return ProtectedCoreHandle(core_path, core)

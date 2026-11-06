"""Huella SHA-256 de un archivo: sirve para auditar qué datos se usaron."""
import hashlib
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
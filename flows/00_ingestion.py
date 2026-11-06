"""Ingesta: fuente remota -> capa Bronze.

Bronze conserva los valores tal como llegan. Los metadatos de ingesta van en
un archivo aparte (_ingestion_metadata.json); nunca se alteran los datos.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    BRONZE_METADATA_NAME,
    BRONZE_UNLABELED_NAME,
)

log = get_logger(__name__)


def read_source(source: str) -> pd.DataFrame:
    """Lee la fuente completa (URL o ruta local) sin transformar valores."""
    return pd.read_csv(source)


def split_labeled(df: pd.DataFrame, n_labeled: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa los pasajeros con etiqueta (primeras n filas) de los que no la tienen."""
    labeled = df.iloc[:n_labeled].copy()
    unlabeled = df.iloc[n_labeled:].copy().reset_index(drop=True)
    return labeled, unlabeled


def run(source: str | None = None, n_labeled: int | None = None, out_dir: Path = BRONZE) -> dict:
    """Descarga, divide y guarda en Bronze. Devuelve los metadatos de la ingesta."""
    source = source or settings.local_path or settings.data_url
    n_labeled = n_labeled or settings.n_labeled

    log.info("Leyendo fuente: %s", source)
    df = read_source(source)
    labeled, unlabeled = split_labeled(df, n_labeled)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / BRONZE_LABELED_NAME
    unlabeled_path = out_dir / BRONZE_UNLABELED_NAME
    labeled.to_csv(labeled_path, index=False)
    unlabeled.to_csv(unlabeled_path, index=False)

    metadata = {
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "rows_labeled": len(labeled),
        "rows_unlabeled": len(unlabeled),
        "columns": list(df.columns),
        "sha256_labeled": sha256_of(labeled_path),
        "sha256_unlabeled": sha256_of(unlabeled_path),
    }
    (out_dir / BRONZE_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Bronze listo: %d etiquetados, %d sin etiqueta", len(labeled), len(unlabeled))
    return metadata
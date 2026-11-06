"""Rutas y nombres de archivos del proyecto (sin rutas absolutas)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

BRONZE_LABELED_NAME = "titanic_labeled.csv"
BRONZE_UNLABELED_NAME = "titanic_unlabeled.csv"
BRONZE_METADATA_NAME = "_ingestion_metadata.json"
SILVER_NAME = "titanic_silver.parquet"          # completo y auditable
SILVER_FINAL_NAME = "titanic_final.parquet"     # numérico, sin columnas originales
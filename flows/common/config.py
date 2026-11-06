"""Configuración del proyecto: valores por defecto + .env opcional."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from flows.common.paths import ROOT

load_dotenv(ROOT / ".env")

DEFAULT_URL = "https://paste.c-net.org/MarvelJinxed"


@dataclass(frozen=True)
class Settings:
    data_url: str
    local_path: str
    n_labeled: int
    random_state: int
    test_size: float
    cv_folds: int


def load_settings() -> Settings:
    return Settings(
        data_url=os.getenv("TITANIC_URL", DEFAULT_URL),
        local_path=os.getenv("TITANIC_LOCAL_PATH", ""),
        n_labeled=int(os.getenv("TITANIC_N_LABELED", "891")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        cv_folds=int(os.getenv("CV_FOLDS", "5")),
    )


settings = load_settings()
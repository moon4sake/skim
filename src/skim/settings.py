from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

BFCL_DIR = DATA_DIR / "bfcl"
APIBANK_DIR = DATA_DIR / "apibank"
XLAM_DIR = DATA_DIR / "xlam"

CACHE_DIR = DATA_DIR / "caches"
RECORD_CACHE_DIR = CACHE_DIR / "records"

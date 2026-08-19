import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("COSMOS_CDS_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
WORK_DIR = Path(os.environ.get("COSMOS_CDS_WORK_DIR", str(DATA_DIR / "work"))).resolve()
TEMP_DIR = Path(tempfile.gettempdir())

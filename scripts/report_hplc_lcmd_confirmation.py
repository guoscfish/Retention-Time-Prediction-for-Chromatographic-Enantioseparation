"""Write reports from the already frozen and revealed confirmation artifacts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hplc_al.lcmd_confirmation import STUDY, report


if __name__ == "__main__":
    (STUDY / "results").mkdir(parents=True, exist_ok=True)
    report()

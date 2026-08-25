import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud_detection import run_pipeline

if __name__ == "__main__":
    print(json.dumps(run_pipeline(ROOT / "artifacts"), indent=2))


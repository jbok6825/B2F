import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import training.createTrainedNetwork as createTrainedNetwork

if __name__ == "__main__":
    createTrainedNetwork.main()

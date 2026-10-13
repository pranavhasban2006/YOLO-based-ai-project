"""Runs every code cell of colab_upload/train_colab.ipynb locally, in test mode.

Test mode: CPU, 1-2 epochs, a small subset of each (dataset, split), outputs under
project/colab_upload/test_run/, which is deleted afterwards. Set YOLO_TEST_E3=1 to also run E3.
Usage: python test_notebook.py [--e3]
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

from common import project_path

sys.stdout.reconfigure(encoding="utf-8")
NB = project_path("colab_upload", "train_colab.ipynb")
UPLOAD = project_path("colab_upload")
TEST_ROOT = UPLOAD / "test_run"


def main() -> int:
    if TEST_ROOT.exists():
        shutil.rmtree(TEST_ROOT)
    TEST_ROOT.mkdir(parents=True)
    os.environ["YOLO_TEST"] = "1"
    os.environ["YOLO_TEST_ROOT"] = str(TEST_ROOT)
    os.environ["YOLO_UPLOAD_DIR"] = str(UPLOAD)
    os.environ["YOLO_TEST_E3"] = "1" if "--e3" in sys.argv else "0"

    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    src = "\n\n".join(cells)
    print(f"running {len(cells)} code cells, E3={os.environ['YOLO_TEST_E3']}")
    t = time.time()
    status = 0
    try:
        exec(compile(src, str(NB), "exec"), {"__name__": "__main__"})
    except Exception:
        import traceback
        traceback.print_exc()
        status = 1
    finally:
        print(f"elapsed {time.time() - t:.0f}s")
        shutil.rmtree(TEST_ROOT, ignore_errors=True)
        print("test_run deleted:", not TEST_ROOT.exists())
    print("RESULT:", "PASS" if status == 0 else "FAIL")
    return status


if __name__ == "__main__":
    sys.exit(main())

"""Run the fixed seed-525 IVR/Hybrid screen; reveal only after global freeze."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hplc_al.ivr_hybrid_screen import prepare, run, reveal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'run', 'reveal', 'all'])
    stage = parser.parse_args().stage
    if stage == 'prepare':
        prepare()
    elif stage == 'run':
        run()
    elif stage == 'reveal':
        reveal()
    else:
        run()
        reveal()

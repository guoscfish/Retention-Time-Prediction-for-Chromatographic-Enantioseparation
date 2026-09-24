"""Resume the explicitly bounded representation study."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hplc_al.representation_runner import diagnostics, run, reveal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['diagnostics', 'run', 'reveal', 'all'])
    args = parser.parse_args()
    if args.stage in ('diagnostics', 'all'):
        diagnostics()
    if args.stage in ('run', 'all'):
        run()
    if args.stage in ('reveal', 'all'):
        reveal()

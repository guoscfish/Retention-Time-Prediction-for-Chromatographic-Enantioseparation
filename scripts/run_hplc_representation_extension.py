"""Continue all four frozen HPLC trajectories under a validation-only stop rule."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hplc_al.representation_extension import run, reveal

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['run','reveal','all'])
    stage = parser.parse_args().stage
    if stage in ('run','all'):
        run()
    if stage in ('reveal','all'):
        reveal()

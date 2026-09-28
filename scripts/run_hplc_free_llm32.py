"""Explicit phase-1 commands; no automatic third round, seed, hybrid or test evaluation."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hplc_al.free_llm_runner import prepare, dry_run, run_selection, advance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','dry-run','select','advance','report'])
    parser.add_argument('--round', type=int, choices=[0,1])
    args = parser.parse_args()
    if args.command in ('select','advance') and args.round is None:
        parser.error('--round is required')
    if args.command == 'prepare': prepare()
    elif args.command == 'dry-run': dry_run()
    elif args.command == 'select': run_selection(args.round)
    elif args.command == 'advance': advance(args.round)
    else:
        from hplc_al.free_llm_report import report
        report()

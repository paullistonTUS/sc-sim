"""Command-line entry point.

  python run.py inputs/sc_inputs.xlsx --check
  python run.py inputs/sc_inputs.xlsx
  python run.py inputs/sc_inputs.xlsx --variants beer_base beer_info_sharing
"""
import argparse
import sys

from scsim.config import InputError, load_config
from scsim.runner import run_all


def main():
    ap = argparse.ArgumentParser(description="Run the supply chain lab models from an input workbook.")
    ap.add_argument("inputs")
    ap.add_argument("--check", action="store_true", help="validate the workbook and stop")
    ap.add_argument("--variants", nargs="+", help="variants to run (default: all)")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--viewer", default="viewer/traces", help="folder for viewer traces, or '' to skip")
    a = ap.parse_args()
    try:
        cfg = load_config(a.inputs)
    except InputError as e:
        print("The input workbook has problems:\n" + str(e))
        sys.exit(1)
    print(f"Inputs OK: {len(cfg.variants)} variants")
    if a.check:
        return
    out = run_all(cfg, a.variants, a.out, a.viewer or None, input_path=a.inputs)
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()

"""Command line:

    python -m rotation_fit train   [--config config.yaml]
    python -m rotation_fit score   --pairs pairs.xlsx [--out scored.xlsx]
    python -m rotation_fit synthetic [--out data/synthetic]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import load_config
from .data import load_hr_data, read_table


def _train(args):
    from .report import save_outputs, write_report
    from .train import train

    cfg = load_config(args.config)
    result = train(cfg, progress=lambda m: print(f"  - {m}", flush=True))
    paths = save_outputs(result, cfg["paths"]["outputs"])
    report = write_report(result, cfg["paths"]["report"])

    print("\nINCLUSION FUNNEL")
    for k, v in result.funnel.items():
        print(f"  {k:<46} {v:>6,}")
    print(f"\nBEST (development set): {result.best['model']} / {result.best['scaler']}")
    print("\nHOLD-OUT")
    print(result.holdout.round(3).to_string(index=False))
    print("\nSIGNIFICANCE (likelihood-ratio test, development rows)")
    print(result.significance.round(4).to_string(index=False))
    print("\nIMPORTANCE (95% interval across splits)")
    print(result.importance.round(4).to_string(index=False))
    r = result.reduced
    print(f"\nSIGNIFICANT-ATTRIBUTES MODEL: {r['features']}")
    print(f"  loss vs full {r['mean_loss']:+.3f} (SE {r['se_loss']:.3f}, tolerance "
          f"{r['tolerance']}) -> {'accepted' if r['not_worse'] else 'too costly'}")
    print(f"  deployed: {result.artifact['metadata']['deployed_variant']} model")
    for n in result.notes:
        print(f"NOTE: {n}")
    print(f"\nmodel + cohort (PERSONAL DATA, keep local): {paths['model'].parent}/")
    print(f"shareable report (aggregates only):          {report}/")


def _score(args):
    from .predict import RotationScorer

    cfg = load_config(args.config)
    hr = load_hr_data(cfg)
    model = args.model or Path(cfg["paths"]["outputs"]) / "rotation_fit_model.joblib"
    scorer = RotationScorer.from_path(model, hr)
    pairs = read_table(args.pairs)
    out = scorer.score_pairs(pairs)
    dest = Path(args.out or Path(cfg["paths"]["outputs"]) / "scored_pairs.xlsx")
    dest.parent.mkdir(parents=True, exist_ok=True)
    (out.to_csv if dest.suffix == ".csv" else out.to_excel)(dest, index=False)
    print(out[["NIK", "destination_position", "success_pct", "out_of_distribution"]]
          .to_string(index=False))
    print(f"\nwrote {dest}")


def _synthetic(args):
    from .synthetic import make_synthetic

    paths = make_synthetic(args.out, n_employees=args.n, seed=args.seed)
    print("synthetic data written:")
    for k, v in paths.items():
        print(f"  {k:<18} {v}")
    print("\nTo use it, point config.yaml `paths:` at these files "
          "(or copy config.yaml and pass --config).")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rotation_fit")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="build cohort, run the experiment, save model and report")
    sc = sub.add_parser("score", help="score a spreadsheet of employee x role pairs")
    sc.add_argument("--pairs", required=True, help="xlsx/csv with NIK, destination_position, "
                                                   "destination_org_unit [, rotation_date]")
    sc.add_argument("--model", default=None)
    sc.add_argument("--out", default=None)
    sy = sub.add_parser("synthetic", help="write fake data with the real layout")
    sy.add_argument("--out", default="data/synthetic")
    sy.add_argument("--n", type=int, default=900)
    sy.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    {"train": _train, "score": _score, "synthetic": _synthetic}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

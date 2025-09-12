# run_example.py
# -*- coding: utf-8 -*-
import argparse, json, os
from solver.ocp_bev import build_and_solve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="path to task json")
    ap.add_argument("--patch", default=None, help="path to bev patch json (optional)")
    ap.add_argument("--out_csv", default="results/traj.csv")
    ap.add_argument("--out_png", default="results/traj.png")
    args = ap.parse_args()

    with open(args.task, "r", encoding="utf-8") as f:
        task = json.load(f)

    patch = None
    if args.patch and os.path.exists(args.patch):
        with open(args.patch, "r", encoding="utf-8") as f:
            patch = json.load(f)

    build_and_solve(task, patch, out_csv=args.out_csv, out_png=args.out_png)
    print("[OK] wrote", args.out_csv, "and", args.out_png)

if __name__ == "__main__":
    main()

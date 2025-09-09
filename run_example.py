# -*- coding: utf-8 -*-
import os, json, argparse
from solver.ocp_bev import build_and_solve

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",  type=str, default="tasks/task_simple.json")
    ap.add_argument("--patch", type=str, default="examples/patch_example.json")
    ap.add_argument("--out_csv", type=str, default="results/traj.csv")
    ap.add_argument("--out_png", type=str, default="results/traj.png")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    task  = json.load(open(args.task,  "r", encoding="utf-8"))
    patch = json.load(open(args.patch, "r", encoding="utf-8"))
    build_and_solve(task, patch, out_csv=args.out_csv, out_png=args.out_png)
    print("[OK] wrote", args.out_csv, "and", args.out_png)

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse, json, os, sys
from solver.ocp_bev import build_and_solve

def load_json_strict(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="path to task json")
    ap.add_argument("--patch", default=None, help="path to bev patch json (optional)")
    ap.add_argument("--out_csv", default="results/traj.csv")
    ap.add_argument("--out_png", default="results/traj.png")
    args = ap.parse_args()

    task = load_json_strict(args.task)

    patch = None
    if args.patch:
        if os.path.exists(args.patch):
            patch = load_json_strict(args.patch)
        else:
            print(f"[WARN] patch file not found: {args.patch}")

    # 调试打印（看清到底读到了什么）
    print("[OCP-BEV] task-costs", "ENABLED" if ("costs" in task and task["costs"]) else "DISABLED")
    if patch is None:
        print("[DEBUG] no patch provided (patch=None)")
    else:
        pm = patch.get("spatial_masks") or []
        print(f"[DEBUG] bev_patch spatial_masks: {len(pm)}")

    # 直接把 task 与 patch 交给 solver；solver 内部会统一解析
    result = build_and_solve(task, patch, out_csv=args.out_csv, out_png=args.out_png)
    print("[OK] wrote", args.out_csv, "and", args.out_png)
    return 0

if __name__ == "__main__":
    sys.exit(main())

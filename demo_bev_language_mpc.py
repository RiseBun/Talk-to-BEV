# -*- coding: utf-8 -*-
import argparse, json
import numpy as np
from semantics.lang_parser import parse_instruction
from semantics.spatial_dsl import build_patch_from_bev
from perception.detectors import fake_bev_objects
from perception.bev_infer import load_risk_map
from perception.uncert import pseudo_variance_from_risk
from riskmap.inflation import compute_inflation
from riskmap.composer import fuse_risk_and_inflation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instr', type=str, required=True)
    ap.add_argument('--risk', type=str, required=True)
    ap.add_argument('--speed', type=float, default=0.6)
    ap.add_argument('--out', type=str, default='examples/patch_example.json')
    args = ap.parse_args()

    slots = parse_instruction(args.instr)
    bev_objs = fake_bev_objects()
    risk, res = load_risk_map(args.risk)
    var = pseudo_variance_from_risk(risk)
    infl = compute_inflation(var, risk, speed=args.speed)
    fused = fuse_risk_and_inflation(risk, infl)

    patch = build_patch_from_bev(slots, bev_objs, risk_map_path=args.risk, base_margin=slots.get('margin_m') or 0.2)
    patch['__debug__'] = {'infl_mean': float(infl.mean()), 'risk_mean': float(risk.mean())}

    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(patch, f, ensure_ascii=False, indent=2)
    print('[OK] Patch saved:', args.out)
    print(json.dumps(patch, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()

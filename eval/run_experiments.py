# -*- coding: utf-8 -*-
import argparse, json, os, random, numpy as np, pandas as pd
from glob import glob

def simulate_once(scn: dict, method: str, seed: int=0):
    random.seed(seed); np.random.seed(seed)
    # 这里是占位的“统计模拟”，仅用于打通聚合与制表流程；请替换为真实调用。
    base = {'SR': 0.65, 'CR': 0.3, 'MD': 0.18, 'EPE': 0.45, 'Smooth': 0.61, 'ST': 0.09}
    bump = {
        'R1': {'SR': +0.07,'CR': -0.08,'MD': +0.03,'EPE': -0.03,'Smooth': +0.03,'ST': +0.03},
        'R2': {'SR': +0.15,'CR': -0.15,'MD': +0.08,'EPE': -0.07,'Smooth': +0.10,'ST': +0.06},
        'R3': {'SR': +0.24,'CR': -0.21,'MD': +0.14,'EPE': -0.11,'Smooth': +0.13,'ST': +0.08},
        'R4': {'SR': +0.28,'CR': -0.26,'MD': +0.23,'EPE': -0.16,'Smooth': +0.18,'ST': +0.11},
        'R5': {'SR': +0.29,'CR': -0.27,'MD': +0.24,'EPE': -0.17,'Smooth': +0.20,'ST': +0.13},
    }
    res = base.copy()
    if method in bump:
        for k,v in bump[method].items():
            res[k] = res[k] + v + np.random.randn()*0.01
    # 简化：场景难度影响
    if scn.get('density')=='high': res['SR']-=0.05; res['CR']+=0.05
    if scn.get('occlusion',0)>0.5: res['CR']+=0.05; res['MD']-=0.02
    if scn.get('dynamic'): res['CR']+=0.03
    if scn.get('lang_noise'): res['SR']-=0.06; res['CR']+=0.06
    res['SR']=max(0,min(1,res['SR'])); res['CR']=max(0,min(1,res['CR']))
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', type=str, default='eval/scenario_configs/*.json')
    ap.add_argument('--methods', type=str, default='R0,R1,R2,R3,R4,R5')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--out', type=str, default='results/result_stub.csv')
    args = ap.parse_args()

    files = sorted(glob(args.scenarios))
    methods = args.methods.split(',')
    rows = []
    for fp in files:
        scn = json.load(open(fp, 'r', encoding='utf-8'))
        name = os.path.splitext(os.path.basename(fp))[0]
        for m in methods:
            for ep in range(args.episodes):
                r = simulate_once(scn, m, seed=ep)
                r.update({'scenario': name, 'method': m, 'episode': ep})
                rows.append(r)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print('[OK] wrote', args.out)

if __name__=='__main__':
    main()

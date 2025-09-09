# -*- coding: utf-8 -*-
import argparse, pandas as pd, numpy as np, os, matplotlib.pyplot as plt

def aggregate(df):
    # 只对数值列做均值聚合，避免把 'scenario' 文本列卷进去
    numeric_cols = ['SR', 'CR', 'MD', 'EPE', 'Smooth', 'ST']
    grp = (
        df.groupby(['scenario', 'method'])[numeric_cols]
          .mean()
          .reset_index()
    )
    return grp

def to_latex_main(grp):
    order = ['R0','R1','R2','R3','R4','R5']
    numeric_cols = ['SR', 'CR', 'MD', 'EPE', 'Smooth', 'ST']
    # 对 method 汇总时也只均值数值列；reindex 保障缺项不会 KeyError
    dfm = (
        grp.groupby('method')[numeric_cols]
           .mean()
           .reindex(order)
    )
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Method & SR$\\uparrow$ & CR$\\downarrow$ & MD$\\uparrow$ & EPE$\\downarrow$ & Smooth$\\uparrow$ & ST$\\downarrow$ \\\\",
        r"\midrule",
    ]
    for m, row in dfm.iterrows():
        if isinstance(m, str):
            lines.append(
                f"{m} & {row.SR*100:.1f} & {row.CR*100:.1f} & {row.MD:.2f} & {row.EPE:.2f} & {row.Smooth:.2f} & {row.ST:.2f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)



def to_latex_ablation(grp_full, grp_drop):
    lines = ['\\begin{tabular}{lrrrr}','\\toprule','Variant & SR(%) & CR(%) & MD(m) & Smooth \\','\\midrule']
    for name,g in [('Full',grp_full),('w/o BEV',grp_drop['wobeV']),('w/o Robust',grp_drop['worob']),('w/o Infl',grp_drop['woinfl']),('w/o Hybrid',grp_drop['wohyb']),('w/o Cons.',grp_drop['wocons'])]:
        r = g.mean()
        lines.append(f"{name} & {r.SR*100:.1f} & {r.CR*100:.1f} & {r.MD:.2f} & {r.Smooth:.2f} \\")
    lines += ['\\bottomrule','\\end{tabular}']
    return '\n'.join(lines)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True)
    ap.add_argument('--outdir', type=str, default='results/')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    grp = aggregate(df)
    latex_main = to_latex_main(grp)
    with open(os.path.join(args.outdir,'main_table.tex'),'w',encoding='utf-8') as f: f.write(latex_main)
    print('[OK] wrote main_table.tex')
    # 简单画一个 solve time 箱线图示例
    plt.figure()
    df.boxplot(column='ST', by='method')
    plt.title('Solve Time by Method'); plt.suptitle('')
    plt.ylabel('seconds')
    plt.savefig(os.path.join(args.outdir,'solve_time_boxplot.png'), bbox_inches='tight')

if __name__=='__main__':
    main()

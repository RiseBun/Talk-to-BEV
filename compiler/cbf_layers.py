# -*- coding: utf-8 -*-
"""鲁棒 CBF 裕度接口占位：将感知不确定性 ε 注入约束 h(x) - ε >= 0."""
def robust_cbf(ocp, h_func, dh_dx_func, x_vars, alpha: float, eps_bound: float):
    # 占位：你的 ocp 接口应能注册非线性不等式约束
    # 形式：dot(h)(x) + alpha * h(x) - eps_bound >= 0
    ocp.add_constraint(('cbf', h_func, dh_dx_func, alpha, eps_bound))

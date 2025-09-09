# -*- coding: utf-8 -*-
"""OCP 注入示例（伪实现）：展示如何把 spatial_masks / risk_map 转化为 OCP 约束与代价。
将这些函数对接到你现有 ocp_builder（CasADi/Ipopt 等）中。
"""
from typing import Dict, Any, List, Tuple
import numpy as np

def point_to_polygon_signed_distance(pt: Tuple[float,float], poly: List[Tuple[float,float]]) -> float:
    # 简化版：多边形外为正，内为负（占位，真实实现请用稳健的几何库）
    # 这里只返回近似“距离-是否在内”的指标
    x, y = pt; # 占位：返回正数表示在外 + 距离，负数在内
    return 0.5  # TODO: 替换为真实几何函数

def inject_spatial_masks(ocp, traj_vars, masks: List[Dict[str,Any]], hybrid_threshold: float=0.0):
    """将 keepout/keepin 注入 OCP。
    - 硬约束：要求采样点对多边形的 signed_distance >= margin
    - 软约束：惩罚项 max(0, margin - signed_distance)*weight
    - Hybrid：低违规走软罚，高违规切硬约束/大罚
    """
    for m in masks:
        mode = m['mode']; hardness = m.get('hardness','hybrid')
        margin = float(m.get('margin_m', 0.2))
        poly = m['points']
        w = float(m.get('weight', 5.0))
        for t, x_t in enumerate(traj_vars):  # x_t: (x,y,theta, ...)
            sd = point_to_polygon_signed_distance((x_t[0], x_t[1]), poly)
            if mode == 'keepout':
                if hardness == 'hard':
                    ocp.add_constraint(sd - margin >= 0.0)
                elif hardness == 'soft':
                    ocp.add_cost(max(0.0, margin - sd) * w)
                else:  # hybrid
                    ocp.add_cost(max(0.0, margin - sd) * w)
                    if sd < -hybrid_threshold:
                        ocp.add_constraint(sd - margin >= 0.0)
            elif mode == 'keepin':
                if hardness == 'hard':
                    ocp.add_constraint(sd + margin <= 0.0)  # sd<0 considered inside (示意)
                elif hardness == 'soft':
                    ocp.add_cost(max(0.0, sd + margin) * w)
                else:
                    ocp.add_cost(max(0.0, sd + margin) * w)
                    if sd > hybrid_threshold:
                        ocp.add_constraint(sd + margin <= 0.0)

def inject_risk_map_cost(ocp, traj_vars, risk_map: np.ndarray, origin_xy=(0.0,0.0), res=0.1, weight=1.0, hard_thr: float=None):
    """沿轨迹采样 risk(x_t) 并加到代价；若 hard_thr 给定，超过阈值处加硬约束或大罚。"""
    H, W = risk_map.shape
    ox, oy = origin_xy
    for t, x_t in enumerate(traj_vars):
        ix = int(round((x_t[0] - ox)/res)); iy = int(round((x_t[1] - oy)/res))
        if 0 <= ix < W and 0 <= iy < H:
            r = float(risk_map[iy, ix])
            ocp.add_cost(r * weight)
            if hard_thr is not None and r >= hard_thr:
                ocp.add_constraint(r <= hard_thr)

def inject_terminal_box(ocp, terminal_state, box_cxy=(0.0,0.0), box_hw=(0.2,0.2), v_limit=0.05, weight=10.0):
    """终端盒（已有思想）：在终点附近要求低速停车，提升“到点且停住”。"""
    ex, ey = terminal_state[0], terminal_state[1]
    cx, cy = box_cxy; hwx, hwy = box_hw
    # 软约束示意：超出盒子则惩罚
    ocp.add_cost(max(0.0, abs(ex - cx) - hwx) * weight + max(0.0, abs(ey - cy) - hwy) * weight)
    ocp.add_constraint(abs(terminal_state[2]) <= v_limit)  # 简化：速度变量索引为2

# -*- coding: utf-8 -*-
import os, json
import numpy as np
import casadi as ca
from typing import Dict, Any, List, Tuple
from .ackermann_model import AckermannModel

# -----------------------
# 小工具
# -----------------------
def _pad5(v):
    v = list(v)
    if len(v) < 5:
        v = v + [0.0] * (5 - len(v))
    return v[:5]

# -----------------------
# AABB SDF（可微）
# -----------------------
def aabb_from_polygon(points, margin=0.0):
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    hx = (xmax - xmin) * 0.5 + margin
    hy = (ymax - ymin) * 0.5 + margin
    return cx, cy, hx, hy

def sdf_aabb(x, y, cx, cy, hx, hy):
    # outside>0, inside<0；safe-sqrt 防抖
    dx = ca.fabs(x - cx) - hx
    dy = ca.fabs(y - cy) - hy
    ax = ca.fmax(dx, 0.0)
    ay = ca.fmax(dy, 0.0)
    outside = ca.sqrt(ca.fmax(ax*ax + ay*ay, 1e-12))
    inside  = ca.fmin(ca.fmax(dx, dy), 0.0)
    return outside + inside

def smooth_hinge(z, rho=0.02):
    # softplus(z/rho)*rho，裁剪输入避免 exp 溢出
    t = z / rho
    t = ca.fmin(ca.fmax(t, -40.0), 40.0)
    return rho * ca.log1p(ca.exp(t))

def soft_keepout(sd, margin, w=10.0):  # sd < margin → penalty
    return w * smooth_hinge(margin - sd, rho=0.02)

def soft_keepin(sd, margin, w=10.0):   # sd > -margin → penalty
    return w * smooth_hinge(sd + margin, rho=0.02)

# -----------------------
# risk_map → AABB masks（简化占位）
# -----------------------
def risk_to_masks(risk_cfg, thr=0.35, min_area=6):
    if not isinstance(risk_cfg, dict): return []
    if risk_cfg.get("type") != "grid": return []
    path = risk_cfg.get("values")
    if not path or (not os.path.exists(path)): return []
    res = float(risk_cfg.get("resolution_m", 0.1))
    ox, oy = (risk_cfg.get("origin_xy") or [0.0, 0.0])
    risk = np.load(path).astype("float32")
    H, W = risk.shape
    vis = np.zeros_like(risk, dtype=bool)
    masks=[]
    for i in range(H):
        for j in range(W):
            if vis[i,j] or risk[i,j] < thr: continue
            stack=[(i,j)]; vis[i,j]=True; cells=[]
            while stack:
                y,x=stack.pop(); cells.append((y,x))
                for ny,nx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                    if 0<=ny<H and 0<=nx<W and (not vis[ny,nx]) and risk[ny,nx]>=thr:
                        vis[ny,nx]=True; stack.append((ny,nx))
            if len(cells) < min_area: continue
            ys=[c[0] for c in cells]; xs=[c[1] for c in cells]
            xmin, xmax = min(xs)*res+ox, (max(xs)+1)*res+ox
            ymin, ymax = min(ys)*res+oy, (max(ys)+1)*res+oy
            masks.append({
                "id": f"ko_risk_{len(masks)}",
                "shape": "polygon",
                "mode": "keepout",
                "points": [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]],
                "margin_m": 0.0, "hardness": "soft", "weight": 3.0,
                "note": "risk block"
            })
    return masks

# -----------------------
# 主函数：构建并求解
# -----------------------
def build_and_solve(task: Dict[str, Any], bev_patch: Dict[str, Any] = None,
                    out_csv="results/traj.csv", out_png="results/traj.png"):
    # 任务超参
    start = _pad5(task.get("start", [0, 0, 0, 0, 0]))
    goal  = _pad5(task.get("goal",  [5, 0, 0, 0, 0]))
    N  = int(task.get("horizon", 40))
    dt = float(task.get("dt", 0.1))
    w_state = task.get("weights", {}).get("state", [6, 6, 0.5, 0.05, 0.05])
    w_ctrl  = task.get("weights", {}).get("control", [0.005, 0.01])
    term_scale = float(task.get("terminal_scale", 4.0))
    u_rate_w = float(task.get("u_rate_weight", 0.1))

    model = AckermannModel()
    nx, nu = model.nx, model.nu
    X = ca.MX.sym("X", nx, N + 1)
    U = ca.MX.sym("U", nu, N)

    g_list, lbg, ubg = [], [], []
    obj = 0

    # 初值约束
    g_list.append(X[:, 0] - ca.DM(start)); lbg += [0.0] * nx; ubg += [0.0] * nx

    Q  = ca.diag(ca.DM(w_state))
    R  = ca.diag(ca.DM(w_ctrl))
    Qf = Q * term_scale

    # -------- 动力学 + 分段参考（via-point）--------
    mid = [(start[0] + goal[0]) / 2.0, (start[1] + goal[1]) / 2.0, 0.0, 0.0, 0.0]

    for k in range(N):
        xk = X[:, k]; uk = U[:, k]
        xnext = model.forward(xk, uk, dt)
        g_list.append(X[:, k + 1] - xnext); lbg += [0.0] * nx; ubg += [0.0] * nx

        ref = mid if k < N // 2 else goal
        obj += ca.mtimes([(xk - ca.DM(ref)).T, Q, (xk - ca.DM(ref))]) + ca.mtimes([uk.T, R, uk])

        if u_rate_w > 0 and k >= 1:
            du = U[:, k] - U[:, k - 1]
            obj += u_rate_w * ca.mtimes(du.T, du)

    # 终端代价（加 终端速度/转角 罚）
    xN = X[:, -1]
    obj += ca.mtimes([(xN - ca.DM(goal)).T, Qf, (xN - ca.DM(goal))])
    obj += 2.0 * (xN[3] ** 2) + 1.0 * (xN[4] ** 2)

    # —— 轻微正则，稳定数值 ——
    obj += 1e-6 * ca.sumsqr(X) + 1e-6 * ca.sumsqr(U)

    # -------- 注入 BEV masks（含 risk blocks）--------
    if bev_patch:
        masks = list(bev_patch.get("spatial_masks") or [])
        if bev_patch.get("risk_map"):
            masks += risk_to_masks(bev_patch["risk_map"], thr=0.35, min_area=6)

        for m in masks:
            if str(m.get("shape", "polygon")) != "polygon":
                continue
            mode = str(m.get("mode", "keepout")).lower()
            hard = str(m.get("hardness", "hybrid")).lower()
            base_margin = float(m.get("margin_m", 0.2))
            w = float(m.get("weight", 8.0))
            pts = m.get("points", [])
            if not pts or len(pts) < 3:
                continue
            cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)

            for k in range(N + 1):
                xk = X[:, k]
                # 速度相关通胀：margin += 0.25 * |v|
                margin_k = base_margin + 0.25 * ca.fabs(xk[3])
                sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)

                if mode == "keepout":
                    if hard == "hard":
                        g_list.append(sd - margin_k); lbg.append(0.0); ubg.append(ca.inf)
                    elif hard == "soft":
                        obj += soft_keepout(sd, margin_k, w=w)
                    else:  # hybrid
                        obj += soft_keepout(sd, margin_k, w=w)
                        g_list.append(sd - (margin_k - 0.10)); lbg.append(0.0); ubg.append(ca.inf)

                elif mode == "keepin":
                    if hard == "hard":
                        g_list.append(-(sd + margin_k)); lbg.append(0.0); ubg.append(ca.inf)
                    elif hard == "soft":
                        obj += soft_keepin(sd, margin_k, w=w)
                    else:
                        obj += soft_keepin(sd, margin_k, w=w)
                        g_list.append(-(sd + (margin_k - 0.10))); lbg.append(0.0); ubg.append(ca.inf)

    # -------- 进度约束（关键）--------
    # ① 终端盒：必须到达目标附近（硬约束）
    eps_pos = float(task.get("terminal_box_eps", 0.08))
    g_list += [
        (X[0, -1] - goal[0]) - eps_pos,
        -(X[0, -1] - goal[0]) - eps_pos,
        (X[1, -1] - goal[1]) - eps_pos,
        -(X[1, -1] - goal[1]) - eps_pos,
    ]
    lbg += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
    ubg += [0.0, 0.0, 0.0, 0.0]

    # ② 无回退（沿目标方向的位移 ≥ -eps）
    goal_xy = ca.DM(goal[:2])
    eps_back = 1e-4
    for k in range(N):
        p_k  = X[0:2, k]
        p_k1 = X[0:2, k+1]
        dir_goal = (goal_xy - p_k) / (1e-6 + ca.norm_2(goal_xy - p_k))
        forward_step = ca.dot(p_k1 - p_k, dir_goal)
        g_list.append(forward_step + eps_back)   # >= 0
        lbg.append(0.0); ubg.append(ca.inf)

    # ③ 起步最小速度（前 1/3 步）
    v_min0 = float(task.get("vmin_warmup", 0.05))
    warmN = max(1, N // 3)
    for k in range(warmN):
        g_list.append(X[3, k] - v_min0)         # v_k >= v_min0
        lbg.append(0.0); ubg.append(ca.inf)

    # -------- 控制界 --------
    a_min, a_max = -1.0, +1.0
    d_min, d_max = -0.5, +0.5
    for k in range(N):
        g_list.append(U[:, k] - ca.DM([a_min, d_min])); lbg += [0.0, 0.0]; ubg += [ca.inf, ca.inf]
        g_list.append(ca.DM([a_max, d_max]) - U[:, k]); lbg += [0.0, 0.0]; ubg += [ca.inf, ca.inf]

    # -------- 组装 NLP --------
    vars_ = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g_list)
    nlp = {"x": vars_, "f": obj, "g": g}

    # -------- 初值（很关键）--------
    X_init = np.zeros((nx, N + 1), dtype=float)
    U_init = np.zeros((nu, N), dtype=float)
    for k in range(N + 1):
        tau = k / float(N)
        X_init[0, k] = (1 - tau) * start[0] + tau * goal[0]
        X_init[1, k] = (1 - tau) * start[1] + tau * goal[1]
        X_init[2, k] = 0.0
        X_init[3, k] = 0.5
        X_init[4, k] = 0.0
    U_init[0, :] = 0.2
    U_init[1, :] = 0.0
    x0_guess = np.concatenate([X_init.ravel(), U_init.ravel()])

    # -------- IPOPT 求解 --------
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 800,
        "ipopt.acceptable_tol": 1e-4
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg), x0=x0_guess)

    z = sol["x"].full().ravel()
    Xsol = z[: (nx * (N + 1))].reshape((nx, N + 1))
    Usol = z[(nx * (N + 1)) :].reshape((nu, N))

    # -------- 输出保存 --------
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["k", "x", "y", "theta", "v", "delta"])
        for k in range(N + 1):
            wr.writerow([k] + [float(v) for v in Xsol[:, k]])

    if out_png:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(Xsol[0, :], Xsol[1, :], marker=".")
        if bev_patch:
            masks = bev_patch.get("spatial_masks") or []
            if bev_patch.get("risk_map"):
                masks += risk_to_masks(bev_patch["risk_map"])
            for m in masks:
                if str(m.get("shape", "polygon")) != "polygon":
                    continue
                pts = m.get("points", [])
                xs = [p[0] for p in pts] + [pts[0][0]]
                ys = [p[1] for p in pts] + [pts[0][1]]
                plt.plot(xs, ys)
        plt.axis("equal")
        plt.title("Trajectory with BEV masks")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")

    return Xsol, Usol

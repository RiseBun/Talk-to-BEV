# -*- coding: utf-8 -*-
import os, json
import numpy as np
import casadi as ca
from typing import Dict, Any, List, Tuple
from .ackermann_model import AckermannModel

# --------- utils ----------
def _pad5(v):
    v = list(v)
    if len(v) < 5:
        v = v + [0.0] * (5 - len(v))
    return v[:5]

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
    dx = ca.fabs(x - cx) - hx
    dy = ca.fabs(y - cy) - hy
    ax = ca.fmax(dx, 0.0)
    ay = ca.fmax(dy, 0.0)
    outside = ca.sqrt(ca.fmax(ax*ax + ay*ay, 1e-12))
    inside  = ca.fmin(ca.fmax(dx, dy), 0.0)
    return outside + inside

def smooth_hinge(z, rho=0.02):
    t = z / rho
    t = ca.fmin(ca.fmax(t, -40.0), 40.0)
    return rho * ca.log1p(ca.exp(t))

def soft_keepout(sd, margin, w=10.0):
    return w * smooth_hinge(margin - sd, rho=0.02)

def soft_keepin(sd, margin, w=10.0):
    return w * smooth_hinge(sd + margin, rho=0.02)

# ----- risk_map → masks（可选） -----
def risk_to_masks(risk_cfg, thr=0.35, min_area=6):
    if not isinstance(risk_cfg, dict):
        return []
    if risk_cfg.get("type") != "grid":
        return []
    path = risk_cfg.get("values")
    if not path or (not os.path.exists(path)):
        return []
    res = float(risk_cfg.get("resolution_m", 0.1))
    ox, oy = (risk_cfg.get("origin_xy") or [0.0, 0.0])
    risk = np.load(path).astype("float32")
    H, W = risk.shape
    vis = np.zeros_like(risk, dtype=bool)
    masks = []
    for i in range(H):
        for j in range(W):
            if vis[i, j] or risk[i, j] < thr:
                continue
            stack = [(i, j)]
            vis[i, j] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]:
                    if 0 <= ny < H and 0 <= nx < W and (not vis[ny, nx]) and risk[ny, nx] >= thr:
                        vis[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) < min_area:
                continue
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            xmin, xmax = min(xs)*res+ox, (max(xs)+1)*res+ox
            ymin, ymax = min(ys)*res+oy, (max(ys)+1)*res+oy
            masks.append({
                "id": f"ko_risk_{len(masks)}",
                "shape": "polygon",
                "mode": "keepout",
                "points": [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                "margin_m": 0.0, "hardness": "soft", "weight": 3.0
            })
    return masks

# ----- 初值构造（与模型无关的几何路由） -----
def rect_from_points(pts, extra_margin=0.0):
    cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)
    return (cx, cy, hx + extra_margin, hy + extra_margin)

def seg_intersects_rect(p0, p1, rect):
    cx, cy, hx, hy = rect
    rx0, rx1 = cx - hx, cx + hx
    ry0, ry1 = cy - hy, cy + hy
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    p = [-dx, dx, -dy, dy]
    q = [x0 - rx0, rx1 - x0, y0 - ry0, ry1 - y0]
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return True
            else:
                continue
        t = -qi / pi
        if pi < 0:
            if t > u1:
                return False
            if t > u0:
                u0 = t
        else:
            if t < u0:
                return False
            if t < u1:
                u1 = t
    return True if u0 <= u1 and (u0 != 0.0 or u1 != 1.0) else False

def route_around_rects(start_xy, goal_xy, rects):
    pts = [tuple(start_xy), tuple(goal_xy)]
    for _ in range(16):
        collided = False
        new_pts = [pts[0]]
        for i in range(len(pts)-1):
            a, b = pts[i], pts[i+1]
            hit_rect = None
            for R in rects:
                if seg_intersects_rect(a, b, R):
                    hit_rect = R
                    break
            if hit_rect is None:
                new_pts.append(b)
                continue
            cx, cy, hx, hy = hit_rect
            eps = 0.20
            corners = [
                (cx - hx - eps, cy - hy - eps),
                (cx + hx + eps, cy - hy - eps),
                (cx + hx + eps, cy + hy + eps),
                (cx - hx - eps, cy + hy + eps),
            ]
            best = None
            bestlen = 1e18
            for c in corners:
                if seg_intersects_rect(a, c, hit_rect) or seg_intersects_rect(c, b, hit_rect):
                    continue
                L = abs(a[0]-c[0]) + abs(a[1]-c[1]) + abs(c[0]-b[0]) + abs(c[1]-b[1])
                if L < bestlen:
                    best, bestlen = [a, c, b], L
            if best is None:
                midx = cx + (hx + 0.4) * (1 if b[0] >= a[0] else -1)
                midy = cy + (hy + 0.4) * (1 if b[1] >= a[1] else -1)
                best = [a, (midx, a[1]), (midx, midy), (b[0], midy), b]
            new_pts += best[1:]
            collided = True
        pts = new_pts
        if not collided:
            break
    simp = [pts[0]]
    for q in pts[1:]:
        if abs(q[0]-simp[-1][0]) + abs(q[1]-simp[-1][1]) > 1e-3:
            simp.append(q)
    return simp

def catmull_rom(points, samples_per_seg=12):
    if len(points) < 2:
        return points
    P = [points[0]] + points + [points[-1]]
    out = []
    for i in range(len(P)-3):
        p0, p1, p2, p3 = P[i], P[i+1], P[i+2], P[i+3]
        for j in range(samples_per_seg):
            t = j/float(samples_per_seg)
            t2 = t*t
            t3 = t*t*t
            x = 0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            out.append((x, y))
    out.append(points[-1])
    return out

def resample_polyline(poly, N):
    if len(poly) == 1:
        return [poly[0]] * (N+1)
    segs = []
    total = 0.0
    for i in range(len(poly)-1):
        a, b = poly[i], poly[i+1]
        L = ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
        segs.append((a, b, L))
        total += L
    if total < 1e-6:
        return [poly[0]] * (N+1)
    targets = [total * k / float(N) for k in range(N+1)]
    out = []
    acc = 0.0
    idx = 0
    cur = segs[0][0]
    out.append(cur)
    for s in targets[1:]:
        while idx < len(segs) and acc + segs[idx][2] < s:
            acc += segs[idx][2]
            cur = segs[idx][1]
            idx += 1
        if idx >= len(segs):
            out.append(segs[-1][1])
            continue
        a, b, L = segs[idx]
        t = (s - acc) / max(L, 1e-9)
        x = a[0] + t*(b[0]-a[0])
        y = a[1] + t*(b[1]-a[1])
        out.append((x, y))
    return out

def heading_and_speed(points, dt, v_target=0.4):
    xs, ys, ths, vs = [], [], [], []
    for i, p in enumerate(points):
        xs.append(p[0])
        ys.append(p[1])
        if i == 0:
            dx = points[1][0] - p[0]
            dy = points[1][1] - p[1]
        else:
            dx = p[0] - points[i-1][0]
            dy = p[1] - points[i-1][1]
        ths.append(np.arctan2(dy, dx + 1e-9))
        dist = (dx*dx + dy*dy) ** 0.5
        vs.append(max(min(dist / dt, v_target), 0.05))
    return xs, ys, ths, vs

def build_initial_guess(start, goal, bev_patch, N, dt, v_target=0.4):
    rects = []
    if bev_patch:
        masks = list(bev_patch.get("spatial_masks") or [])
        if bev_patch.get("risk_map"):
            masks += risk_to_masks(bev_patch["risk_map"], thr=0.35, min_area=6)
        for m in masks:
            if str(m.get("mode", "keepout")).lower() != "keepout":
                continue
            pts = m.get("points", [])
            if not pts:
                continue
            infl = float(m.get("margin_m", 0.2)) + 0.30
            rects.append(rect_from_points(pts, extra_margin=infl))
    start_xy = (start[0], start[1])
    goal_xy  = (goal[0], goal[1])
    coarse = route_around_rects(start_xy, goal_xy, rects)
    smooth = catmull_rom(coarse, samples_per_seg=12)
    pts = resample_polyline(smooth, N)
    xs, ys, ths, vs = heading_and_speed(pts, dt, v_target=v_target)
    nx, nu = 5, 2
    X_init = np.zeros((nx, N+1))
    U_init = np.zeros((nu, N))
    for k in range(N+1):
        X_init[0, k] = xs[k]
        X_init[1, k] = ys[k]
        X_init[2, k] = ths[k]
        X_init[3, k] = vs[k]
        X_init[4, k] = 0.0
    for k in range(N):
        dv  = X_init[3, k+1] - X_init[3, k]
        dth = X_init[2, k+1] - X_init[2, k]
        while dth >  np.pi: dth -= 2*np.pi
        while dth < -np.pi: dth += 2*np.pi
        U_init[0, k] = np.clip(dv/dt, -0.4, 0.6)
        U_init[1, k] = np.clip(dth,     -0.2, 0.2)
    return X_init, U_init

print("[OCP-BEV] v1.3 (geom-vmax, soft terminal, time-ref, jerk, L-BFGS)")

def build_and_solve(task: Dict[str, Any], bev_patch: Dict[str, Any] = None,
                    out_csv="results/traj.csv", out_png="results/traj.png",
                    debug_save_init="results/init_guess.png"):
    # ---------- task ----------
    start = _pad5(task.get("start", [0, 0, 0, 0, 0]))
    goal  = _pad5(task.get("goal",  [5, 0, 0, 0, 0]))
    N  = int(task.get("horizon", 60))
    dt = float(task.get("dt", 0.05))
    w_state = task.get("weights", {}).get("state", [2.5, 2.5, 0.2, 0.02, 0.02])
    w_ctrl  = task.get("weights", {}).get("control", [0.10, 0.10])
    term_scale = float(task.get("terminal_scale", 2.5))
    u_rate_w = float(task.get("u_rate_weight", 0.5))
    term_eps  = float(task.get("terminal_box_eps", 0.25))
    term_mode = str(task.get("terminal_box_mode", "soft")).lower()  # "soft"|"hard"
    enable_no_backtrack = bool(task.get("no_backtrack", False))
    vmin_warmup = float(task.get("vmin_warmup", 0.0))

    model = AckermannModel()
    nx, nu = model.nx, model.nu
    X = ca.MX.sym("X", nx, N + 1)
    U = ca.MX.sym("U", nu, N)

    g_list, lbg, ubg = [], [], []
    obj = 0

    # initial state
    g_list.append(X[:, 0] - ca.DM(start))
    lbg += [0.0] * nx
    ubg += [0.0] * nx

    Q  = ca.diag(ca.DM(w_state))
    R  = ca.diag(ca.DM(w_ctrl))
    Qf = Q * term_scale

    print(f"[OCP-BEV] goal=({goal[0]:.3f},{goal[1]:.3f}), N={N}, dt={dt}")

    # dynamics + time-varying reference
    jerk_w = 0.5
    for k in range(N):
        xk = X[:, k]
        uk = U[:, k]
        xnext = model.forward(xk, uk, dt)
        g_list.append(X[:, k+1] - xnext)
        lbg += [0.0] * nx
        ubg += [0.0] * nx

        tau = (k + 1) / float(N + 1)
        x_ref = (1 - tau) * start[0] + tau * goal[0]
        y_ref = (1 - tau) * start[1] + tau * goal[1]
        ref = [x_ref, y_ref, 0.0, 0.0, 0.0]
        obj += ca.mtimes([(xk - ca.DM(ref)).T, Q, (xk - ca.DM(ref))]) + ca.mtimes([uk.T, R, uk])

        if u_rate_w > 0 and k >= 1:
            du = U[:, k] - U[:, k-1]
            obj += u_rate_w * ca.mtimes(du.T, du)
        if k >= 1:
            ddelta = U[1, k] - U[1, k-1]
            obj += jerk_w * (ddelta * ddelta)

    # terminal cost (soft) + optional hard box
    xN = X[:, -1]
    obj += ca.mtimes([(xN - ca.DM(goal)).T, Qf, (xN - ca.DM(goal))])
    obj += 2.0 * (xN[3] ** 2) + 1.0 * (xN[4] ** 2)  # prefer v,delta -> 0

    if term_mode == "hard":
        g_list += [
            (X[0, -1] - goal[0]) - term_eps,
            -(X[0, -1] - goal[0]) - term_eps,
            (X[1, -1] - goal[1]) - term_eps,
            -(X[1, -1] - goal[1]) - term_eps,
        ]
        lbg += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
        ubg += [0.0, 0.0, 0.0, 0.0]
    else:
        # soft terminal (guarantees feasibility)
        obj += 50.0 * ((X[0, -1] - goal[0])**2 + (X[1, -1] - goal[1])**2)

    # BEV masks (soft by default for stability)
    if bev_patch:
        masks = list(bev_patch.get("spatial_masks") or [])
        if bev_patch.get("risk_map"):
            masks += risk_to_masks(bev_patch["risk_map"], thr=0.35, min_area=6)
        for m in masks:
            if str(m.get("shape", "polygon")) != "polygon":
                continue
            mode = str(m.get("mode", "keepout")).lower()
            hard = str(m.get("hardness", "soft")).lower()
            base_margin = float(m.get("margin_m", 0.2))
            w = float(m.get("weight", 8.0))
            pts = m.get("points", [])
            if not pts or len(pts) < 3:
                continue
            cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)
            for k in range(N + 1):
                xk = X[:, k]
                margin_k = base_margin + 0.25 * ca.fabs(xk[3])  # speed-aware inflation
                sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)
                if mode == "keepout":
                    if hard == "hard":
                        g_list.append(sd - margin_k)
                        lbg.append(0.0)
                        ubg.append(ca.inf)
                    elif hard == "soft":
                        obj += soft_keepout(sd, margin_k, w=w)
                    else:
                        obj += soft_keepout(sd, margin_k, w=w)
                        g_list.append(sd - (margin_k - 0.10))
                        lbg.append(0.0)
                        ubg.append(ca.inf)
                elif mode == "keepin":
                    if hard == "hard":
                        g_list.append(-(sd + margin_k))
                        lbg.append(0.0)
                        ubg.append(ca.inf)
                    elif hard == "soft":
                        obj += soft_keepin(sd, margin_k, w=w)
                    else:
                        obj += soft_keepin(sd, margin_k, w=w)
                        g_list.append(-(sd + (margin_k - 0.10)))
                        lbg.append(0.0)
                        ubg.append(ca.inf)

    # optional monotonic progress
    if enable_no_backtrack:
        goal_xy = ca.DM(goal[:2])
        eps_back = 1e-4
        for k in range(N):
            p_k  = X[0:2, k]
            p_k1 = X[0:2, k+1]
            dir_goal = (goal_xy - p_k) / (1e-6 + ca.norm_2(goal_xy - p_k))
            forward_step = ca.dot(p_k1 - p_k, dir_goal)
            g_list.append(forward_step + eps_back)   # >= 0
            lbg.append(0.0)
            ubg.append(ca.inf)

    if vmin_warmup > 0:
        warmN = max(1, N // 3)
        for k in range(warmN):
            g_list.append(X[3, k] - vmin_warmup)
            lbg.append(0.0)
            ubg.append(ca.inf)

    # controls bounds
    a_min, a_max = -0.3, +0.3
    d_min, d_max = -0.5, +0.5
    for k in range(N):
        g_list.append(U[:, k] - ca.DM([a_min, d_min]))
        lbg += [0.0, 0.0]
        ubg += [ca.inf, ca.inf]
        g_list.append(ca.DM([a_max, d_max]) - U[:, k])
        lbg += [0.0, 0.0]
        ubg += [ca.inf, ca.inf]

    # geometric speed cap  ||p_{k+1}-p_k|| <= v_max*dt   (model-layout independent)
    v_max = float(task.get("vmax_geom", 1.0))
    for k in range(N):
        step = X[0:2, k+1] - X[0:2, k]
        move = ca.norm_2(step)
        g_list.append(v_max * dt - move)  # >= 0
        lbg.append(0.0)
        ubg.append(ca.inf)

    # pack NLP
    vars_ = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
    g = ca.vertcat(*g_list)
    nlp = {"x": vars_, "f": obj, "g": g}

    # initial guess
    X_init, U_init = build_initial_guess(start, goal, bev_patch, N=N, dt=dt, v_target=0.35)
    x0_guess = np.concatenate([X_init.ravel(), U_init.ravel()])

    if debug_save_init:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(X_init[0, :], X_init[1, :], marker=".")
        if bev_patch:
            masks = bev_patch.get("spatial_masks") or []
            for m in masks:
                if str(m.get("shape", "polygon")) != "polygon":
                    continue
                pts = m.get("points", [])
                if pts:
                    xs = [p[0] for p in pts] + [pts[0][0]]
                    ys = [p[1] for p in pts] + [pts[0][1]]
                    plt.plot(xs, ys)
        plt.axis("equal")
        plt.title("Initial guess")
        os.makedirs(os.path.dirname(debug_save_init), exist_ok=True)
        plt.savefig(debug_save_init, bbox_inches="tight")

    # IPOPT options
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.max_iter": 1200,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.nlp_scaling_method": "gradient-based"
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    sol = solver(lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg), x0=x0_guess)

    z = sol["x"].full().ravel()
    Xsol = z[: (nx * (N + 1))].reshape((nx, N + 1))
    Usol = z[(nx * (N + 1)) :].reshape((nu, N))

    print(f"[OCP-BEV] xN=({float(Xsol[0,-1]):.3f},{float(Xsol[1,-1]):.3f})")

    # dump
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
                if pts:
                    xs = [p[0] for p in pts] + [pts[0][0]]
                    ys = [p[1] for p in pts] + [pts[0][1]]
                    plt.plot(xs, ys)
        plt.axis("equal")
        plt.title("Trajectory with BEV masks")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")

    return Xsol, Usol

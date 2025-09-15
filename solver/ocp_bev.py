# -*- coding: utf-8 -*-
import os, json
import numpy as np
import casadi as ca
from typing import Dict, Any, List, Tuple
from .ackermann_model import AckermannModel

# ======================
# Utils
# ======================

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

# --- geometry helpers for initial guess ---
def rect_from_points(pts, extra_margin=0.0):
    cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)
    return (cx, cy, hx + extra_margin, hy + extra_margin)

# ---- robust segment-rectangle intersection ----
def _point_in_aabb(px, py, cx, cy, hx, hy):
    return (cx - hx) <= px <= (cx + hx) and (cy - hy) <= py <= (cy + hy)

def _segments_intersect(p1, p2, q1, q2):
    # standard orientation test
    def orient(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and
                min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)
    o1 = orient(p1, p2, q1)
    o2 = orient(p1, p2, q2)
    o3 = orient(q1, q2, p1)
    o4 = orient(q1, q2, p2)
    if (o1 == 0 and on_seg(p1, p2, q1)) or (o2 == 0 and on_seg(p1, p2, q2)) \
       or (o3 == 0 and on_seg(q1, q2, p1)) or (o4 == 0 and on_seg(q1, q2, p2)):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def seg_intersects_rect(p0, p1, rect):
    # rect: (cx, cy, hx, hy)
    cx, cy, hx, hy = rect
    # 快速剔除 (两点都在同一侧外部且与盒无重叠投影)
    if (p0[0] < cx - hx and p1[0] < cx - hx) or (p0[0] > cx + hx and p1[0] > cx + hx) \
       or (p0[1] < cy - hy and p1[1] < cy - hy) or (p0[1] > cy + hy and p1[1] > cy + hy):
        # 仍可能对角切到，所以不能直接返回 False；继续判断
        pass

    # 端点在盒内 -> 相交
    if _point_in_aabb(p0[0], p0[1], cx, cy, hx, hy) or _point_in_aabb(p1[0], p1[1], cx, cy, hx, hy):
        return True

    # 与四条边相交？
    r = [(cx-hx, cy-hy), (cx+hx, cy-hy), (cx+hx, cy+hy), (cx-hx, cy+hy)]
    edges = [(r[0], r[1]), (r[1], r[2]), (r[2], r[3]), (r[3], r[0])]
    for e0, e1 in edges:
        if _segments_intersect(p0, p1, e0, e1):
            return True
    return False



def route_around_rects(start_xy, goal_xy, rects):
    # 尝试从直线开始，遇撞则加“绕角”节点，最多迭代若干次
    pts = [tuple(start_xy), tuple(goal_xy)]
    for _ in range(24):
        collided = False
        new_pts = [pts[0]]
        for i in range(len(pts)-1):
            a, b = pts[i], pts[i+1]
            hit = None
            for R in rects:
                if seg_intersects_rect(a, b, R):
                    hit = R
                    break
            if hit is None:
                new_pts.append(b)
                continue
            cx, cy, hx, hy = hit
            eps = 0.35  # 比之前更大的安全偏置
            candidates = [
                (cx - hx - eps, cy - hy - eps),
                (cx + hx + eps, cy - hy - eps),
                (cx + hx + eps, cy + hy + eps),
                (cx - hx - eps, cy + hy + eps),
            ]
            best = None
            best_cost = 1e18
            for c in candidates:
                if seg_intersects_rect(a, c, hit) or seg_intersects_rect(c, b, hit):
                    continue
                # 简单 L1 距离做代价，避免不必要弯曲
                cost = (abs(a[0]-c[0]) + abs(a[1]-c[1]) +
                        abs(c[0]-b[0]) + abs(c[1]-b[1]))
                if cost < best_cost:
                    best = [a, c, b]
                    best_cost = cost
            if best is None:
                # 实在绕不过，走上/下/左/右四向“凹”字
                up    = (cx, cy + hy + eps)
                down  = (cx, cy - hy - eps)
                left  = (cx - hx - eps, cy)
                right = (cx + hx + eps, cy)
                for mid in (up, down, left, right):
                    if not seg_intersects_rect(a, mid, hit) and not seg_intersects_rect(mid, b, hit):
                        best = [a, mid, b]; break
                if best is None:
                    # 兜底：从四角里选离 a+b 最近的一角
                    corners = [(cx - hx - eps, cy - hy - eps),
                               (cx + hx + eps, cy - hy - eps),
                               (cx + hx + eps, cy + hy + eps),
                               (cx - hx - eps, cy + hy + eps)]
                    corners.sort(key=lambda p: (abs(a[0]-p[0])+abs(a[1]-p[1])+
                                                abs(b[0]-p[0])+abs(b[1]-p[1])))
                    best = [a, corners[0], b]
            new_pts += best[1:]
            collided = True
        pts = new_pts
        if not collided:
            break

    # 简单抽稀
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
    # task.masks 也参与初值避障（当其是 rect）
    # 同时 patch.spatial_masks 也参与
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
            infl = float(m.get("margin_m", 0.2)) + 0.45
            rects.append(rect_from_points(pts, extra_margin=infl))
    # NOTE: 如果你想让 task.masks 也纳入初值几何避障，可以在 run_example.py 里把 task 先转 polygon 再传进来

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

# =============== 新增：把 rect-mask 规范化为 polygon，并统一收集 ===============

def rect_to_poly(center, size, yaw=0.0):
    cx, cy = float(center[0]), float(center[1])
    hx, hy = float(size[0]) * 0.5, float(size[1]) * 0.5
    # yaw 忽略（按 axis-aligned）
    return [[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]]

def collect_polygon_masks(task: Dict[str, Any], bev_patch: Dict[str, Any]):
    polys = []
    # 1) task.masks（rect -> polygon）
    for m in task.get("masks", []) or []:
        if str(m.get("type", "rect")).lower() == "rect":
            poly = rect_to_poly(m.get("center", [0,0]), m.get("size", [1,1]), m.get("yaw", 0.0))
            polys.append({
                "id": m.get("id", f"task_rect_{len(polys)}"),
                "shape": "polygon", "mode": "keepout", "hardness": "soft",
                "points": poly, "margin_m": 0.0, "weight": 0.0
            })
    # 2) patch.spatial_masks
    if bev_patch and (bev_patch.get("spatial_masks") is not None):
        for m in bev_patch.get("spatial_masks") or []:
            polys.append(m)
    return polys

# =============== 主函数 ===============

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

    if bev_patch is None:
        print("[DEBUG] no patch provided (patch=None)")

    model = AckermannModel()
    nx, nu = model.nx, model.nu
    X = ca.MX.sym("X", nx, N + 1)
    U = ca.MX.sym("U", nu, N)

    g_list, lbg, ubg = [], [], []
    obj = 0

    # initial state constraint
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

    # terminal cost + optional hard box
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
        obj += 50.0 * ((X[0, -1] - goal[0])**2 + (X[1, -1] - goal[1])**2)

    # ---------- BEV masks & costs from task/patch ----------
    # 收集 polygon masks：task.masks(rect->poly) + patch.spatial_masks
    masks_poly = collect_polygon_masks(task, bev_patch)
    print(f"[DEBUG] collected polygon masks: {len(masks_poly)}")

    # task.costs（支持 barrier / soft）
    task_costs = list(task.get("costs") or [])
    if len(task_costs) > 0:
        print("[OCP-BEV] task-costs ENABLED")
    print(f"[DEBUG] task cost items: {len(task_costs)}")

    # 1) 先对 patch/rect→poly 的 keepout/keepin 加项
    for idx, m in enumerate(masks_poly):
        shape = str(m.get("shape", "polygon")).lower()
        if shape != "polygon":
            continue
        mode = str(m.get("mode", "keepout")).lower()
        hard = str(m.get("hardness", "soft")).lower()
        pts = m.get("points", [])
        base_margin = float(m.get("margin_m", 0.2))
        w = float(m.get("weight", 8.0))
        if not pts:
            continue
        cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)
        print(f"[DEBUG] add {mode}:{hard} @ aabb=({cx:.3f},{cy:.3f},{hx:.3f},{hy:.3f})")
        added_terms = 0
        for k in range(N + 1):
            xk = X[:, k]
            margin_k = base_margin + 0.25 * ca.fabs(xk[3])  # 速度膨胀
            sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)
            if mode == "keepout":
                if hard == "hard":
                    g_list.append(sd - margin_k)
                    lbg.append(0.0); ubg.append(ca.inf)
                else:
                    obj += soft_keepout(sd, margin_k, w=w)
            elif mode == "keepin":
                if hard == "hard":
                    g_list.append(-(sd + margin_k))
                    lbg.append(0.0); ubg.append(ca.inf)
                else:
                    obj += soft_keepin(sd, margin_k, w=w)
            added_terms += 1
        print(f"[DEBUG] mask {idx} added_terms: {added_terms}")

    # 2) 再把 task.costs 里的 barrier/soft 对应到 mask_id
    def find_mask_polygon(mask_id: str):
        for mm in masks_poly:
            if str(mm.get("id", "")) == str(mask_id):
                return mm
        return None

    for cidx, c in enumerate(task_costs):
        ctype = str(c.get("type", "")).lower()
        if ctype not in ("bev_mask_barrier", "bev_mask_soft"):
            print(f"[DEBUG] skip unsupported cost[{cidx}] type={ctype}")
            continue
        mask_id = c.get("mask_id", None)
        mref = find_mask_polygon(mask_id) if mask_id is not None else None
        if mref is None:
            print(f"[DEBUG] cost[{cidx}] cannot find mask_id={mask_id} -> SKIP")
            continue
        pts = mref.get("points", [])
        if not pts:
            print(f"[DEBUG] cost[{cidx}] mask_id={mask_id} has empty points -> SKIP")
            continue
        cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)
        weight = float(c.get("weight", 200.0))
        if ctype == "bev_mask_soft":
            base_margin = float(c.get("margin", 0.20))
            penalty = str(c.get("penalty", "huber"))
            for k in range(N + 1):
                xk = X[:, k]
                margin_k = base_margin + 0.25 * ca.fabs(xk[3])
                sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)
                obj += weight * smooth_hinge(margin_k - sd, rho=0.02)
            print(f"[DEBUG] add cost[{cidx}] soft mask={mask_id} weight={weight}")
        elif ctype == "bev_mask_barrier":
            barr = c.get("barrier", {}) or {}
            radius = float(barr.get("radius", 0.20))
            eps = float(barr.get("epsilon", 1e-3))
            for k in range(N + 1):
                xk = X[:, k]
                sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)
                # -log(sd - radius + eps) 型 barrier（sd <= r 会变大）
                obj += weight * (-ca.log(sd - radius + eps))
            print(f"[DEBUG] add cost[{cidx}] barrier mask={mask_id} weight={weight} r={radius} eps={eps}")

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
            lbg.append(0.0); ubg.append(ca.inf)

    if vmin_warmup > 0:
        warmN = max(1, N // 3)
        for k in range(warmN):
            g_list.append(X[3, k] - vmin_warmup)
            lbg.append(0.0); ubg.append(ca.inf)

    # controls bounds
    a_min, a_max = -0.3, +0.3
    d_min, d_max = -0.5, +0.5
    for k in range(N):
        g_list.append(U[:, k] - ca.DM([a_min, d_min]))
        lbg += [0.0, 0.0]; ubg += [ca.inf, ca.inf]
        g_list.append(ca.DM([a_max, d_max]) - U[:, k])
        lbg += [0.0, 0.0]; ubg += [ca.inf, ca.inf]

    # geometric speed cap
    v_max = float(task.get("vmax_geom", 1.0))
    for k in range(N):
        step = X[0:2, k+1] - X[0:2, k]
        move = ca.norm_2(step)
        g_list.append(v_max * dt - move)  # >= 0
        lbg.append(0.0); ubg.append(ca.inf)

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
        # 画 patch 的 polygon
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
        # 画 task.masks（rect->poly）
        for m in task.get("masks", []) or []:
            if str(m.get("type","rect")).lower()=="rect":
                poly = rect_to_poly(m.get("center",[0,0]), m.get("size",[1,1]), m.get("yaw",0.0))
                xs = [p[0] for p in poly] + [poly[0][0]]
                ys = [p[1] for p in poly] + [poly[0][1]]
                plt.plot(xs, ys)
        plt.axis("equal")
        plt.title("Initial guess")
        os.makedirs(os.path.dirname(debug_save_init), exist_ok=True)
        plt.savefig(debug_save_init, bbox_inches="tight")

    # IPOPT options（收紧可行性）
    opts = {
    "ipopt.print_level": 0,
    "print_time": 0,
    "ipopt.max_iter": 1200,
    "ipopt.tol": 1e-7,
    "ipopt.acceptable_tol": 1e-5,
    "ipopt.mu_strategy": "adaptive",
    "ipopt.linear_solver": "mumps",
    "ipopt.hessian_approximation": "limited-memory",
    "ipopt.nlp_scaling_method": "gradient-based",
    # 新增三条 ↓
    "ipopt.honor_original_bounds": "yes",
    "ipopt.bound_relax_factor": 1e-12,
    "ipopt.constr_viol_tol": 1e-8,
    }

    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
       # ====== 求解 ======
    sol = solver(lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg), x0=x0_guess)
    stats = solver.stats()

    z = sol["x"].full().ravel()
    Xsol = z[: (nx * (N + 1))].reshape((nx, N + 1))
    Usol = z[(nx * (N + 1)) :].reshape((nu, N))

    print(f"[OCP-BEV] xN=({float(Xsol[0,-1]):.3f},{float(Xsol[1,-1]):.3f})")
    print(f"[DEBUG] IPOPT return_status: {stats.get('return_status','?')}")

    # ====== 统一的可行性“自检” ======
    try:
        # g_fun(vars_) = g(x) 用于评估约束值
        g_fun = ca.Function("g_fun", [vars_], [g])
        g_val = g_fun(sol["x"]).full().ravel()

        # 根据 lbg/ubg 区分：等式约束 (lbg==ubg) 与 “>=0”/“<=0”不等式
        # 我们关心：等式的最大绝对误差；以及 “>=0” 约束的最小裕度（最小值）
        eq_max_abs = 0.0
        ineq_min_margin = float("+inf")

        # 注意：列表 lbg/ubg 里可能含 inf
        for gi, lb, ub in zip(g_val, lbg, ubg):
            lb = float(lb)
            ub = float(ub)
            if np.isfinite(lb) and np.isfinite(ub) and abs(ub - lb) < 1e-12:
                # 等式约束：g == lb == ub
                eq_max_abs = max(eq_max_abs, abs(gi - lb))
            else:
                # 不等式。常见建模是 g >= 0（即 lb=0, ub=+inf）
                # 我们统计 g 的最小值作为“最小裕度”（负值表示穿约束）
                if (abs(lb) < 1e-12) and (not np.isfinite(ub)):
                    ineq_min_margin = min(ineq_min_margin, float(gi))
                else:
                    # 其它形式也报一下最紧违约程度（到最近边界的距离）
                    # viol < 0 表示违约
                    to_lb = gi - lb if np.isfinite(lb) else float("+inf")
                    to_ub = ub - gi if np.isfinite(ub) else float("+inf")
                    viol = min(to_lb, to_ub)
                    ineq_min_margin = min(ineq_min_margin, float(viol))

        print(f"[CHECK] equality max |violation| = {eq_max_abs:.3e}")
        if ineq_min_margin == float("+inf"):
            print("[CHECK] no inequality constraints detected")
        else:
            print(f"[CHECK] inequality min margin  = {ineq_min_margin:.4f}  (negative => crossed)")

    except Exception as e:
        print("[WARN] feasibility check failed:", repr(e))

    # ====== 结果输出 ======
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

        # 画 patch masks
        if bev_patch:
            masks = bev_patch.get("spatial_masks") or []
            if bev_patch.get("risk_map"):
                masks += risk_to_masks(bev_patch["risk_map"])
            for m in masks:
                if str(m.get("shape", "polygon")).lower() != "polygon":
                    continue
                pts = m.get("points", [])
                if pts:
                    xs = [p[0] for p in pts] + [pts[0][0]]
                    ys = [p[1] for p in pts] + [pts[0][1]]
                    plt.plot(xs, ys)

        # 画 task rect->poly
        for m in (task.get("masks") or []):
            if str(m.get("type","rect")).lower() == "rect":
                poly = rect_to_poly(m.get("center",[0,0]), m.get("size",[1,1]), m.get("yaw",0.0))
                xs = [p[0] for p in poly] + [poly[0][0]]
                ys = [p[1] for p in poly] + [poly[0][1]]
                plt.plot(xs, ys)

        plt.axis("equal")
        plt.title("Trajectory with BEV masks")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")

    return Xsol, Usol

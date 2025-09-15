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
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    hx = 0.5 * (xmax - xmin) + margin
    hy = 0.5 * (ymax - ymin) + margin
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
    if not isinstance(risk_cfg, dict): return []
    if risk_cfg.get("type") != "grid": return []
    path = risk_cfg.get("values")
    if not path or (not os.path.exists(path)): return []
    res = float(risk_cfg.get("resolution_m", 0.1))
    ox, oy = (risk_cfg.get("origin_xy") or [0.0, 0.0])
    risk = np.load(path).astype("float32")
    H, W = risk.shape
    vis = np.zeros_like(risk, dtype=bool)
    masks = []
    for i in range(H):
        for j in range(W):
            if vis[i, j] or risk[i, j] < thr: continue
            stack = [(i, j)]; vis[i, j] = True; cells=[]
            while stack:
                y, x = stack.pop(); cells.append((y, x))
                for ny, nx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
                    if 0<=ny<H and 0<=nx<W and (not vis[ny,nx]) and risk[ny,nx]>=thr:
                        vis[ny,nx]=True; stack.append((ny,nx))
            if len(cells) < min_area: continue
            ys=[c[0] for c in cells]; xs=[c[1] for c in cells]
            xmin, xmax = min(xs)*res+ox, (max(xs)+1)*res+ox
            ymin, ymax = min(ys)*res+oy, (max(ys)+1)*res+oy
            masks.append({
                "id": f"ko_risk_{len(masks)}",
                "shape": "polygon", "mode": "keepout", "hardness": "soft",
                "points": [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]],
                "margin_m": 0.0, "weight": 3.0
            })
    return masks

# --- geometry helpers for initial guess ---
def rect_from_points(pts, extra_margin=0.0):
    cx, cy, hx, hy = aabb_from_polygon(pts, margin=0.0)
    return (cx, cy, hx + extra_margin, hy + extra_margin)

# ---- robust segment-rectangle intersection ----
def _point_in_aabb(px, py, cx, cy, hx, hy):
    return (cx-hx) <= px <= (cx+hx) and (cy-hy) <= py <= (cy+hy)

def _segments_intersect(p1, p2, q1, q2):
    def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    def on_seg(a,b,c): return (min(a[0],b[0])-1e-12 <= c[0] <= max(a[0],b[0])+1e-12 and
                               min(a[1],b[1])-1e-12 <= c[1] <= max(a[1],b[1])+1e-12)
    o1=orient(p1,p2,q1); o2=orient(p1,p2,q2); o3=orient(q1,q2,p1); o4=orient(q1,q2,p2)
    if (o1==0 and on_seg(p1,p2,q1)) or (o2==0 and on_seg(p1,p2,q2)) or \
       (o3==0 and on_seg(q1,q2,p1)) or (o4==0 and on_seg(q1,q2,p2)):
        return True
    return (o1>0)!=(o2>0) and (o3>0)!=(o4>0)

def seg_intersects_rect(p0, p1, rect):
    cx, cy, hx, hy = rect
    if (p0[0] < cx-hx and p1[0] < cx-hx) or (p0[0] > cx+hx and p1[0] > cx+hx) or \
       (p0[1] < cy-hy and p1[1] < cy-hy) or (p0[1] > cy+hy and p1[1] > cy+hy):
        pass
    if _point_in_aabb(p0[0],p0[1],cx,cy,hx,hy) or _point_in_aabb(p1[0],p1[1],cx,cy,hx,hy):
        return True
    r=[(cx-hx,cy-hy),(cx+hx,cy-hy),(cx+hx,cy+hy),(cx-hx,cy+hy)]
    for e0,e1 in [(r[0],r[1]),(r[1],r[2]),(r[2],r[3]),(r[3],r[0])]:
        if _segments_intersect(p0,p1,e0,e1): return True
    return False

def route_around_rects(start_xy, goal_xy, rects):
    pts=[tuple(start_xy), tuple(goal_xy)]
    for _ in range(24):
        collided=False; new_pts=[pts[0]]
        for i in range(len(pts)-1):
            a,b = pts[i], pts[i+1]; hit=None
            for R in rects:
                if seg_intersects_rect(a,b,R): hit=R; break
            if hit is None: new_pts.append(b); continue
            cx,cy,hx,hy=hit; eps=0.35
            candidates=[(cx-hx-eps, cy-hy-eps),(cx+hx+eps, cy-hy-eps),
                        (cx+hx+eps, cy+hy+eps),(cx-hx-eps, cy+hy+eps)]
            best=None; best_cost=1e18
            for c in candidates:
                if seg_intersects_rect(a,c,hit) or seg_intersects_rect(c,b,hit): continue
                cost=(abs(a[0]-c[0])+abs(a[1]-c[1])+abs(c[0]-b[0])+abs(c[1]-b[1]))
                if cost < best_cost: best=[a,c,b]; best_cost=cost
            if best is None:
                for mid in [(cx,cy+hy+eps),(cx,cy-hy-eps),(cx-hx-eps,cy),(cx+hx+eps,cy)]:
                    if not seg_intersects_rect(a,mid,hit) and not seg_intersects_rect(mid,b,hit):
                        best=[a,mid,b]; break
                if best is None:
                    corners=[(cx-hx-eps,cy-hy-eps),(cx+hx+eps,cy-hy-eps),
                             (cx+hx+eps,cy+hy+eps),(cx-hx-eps,cy+hy+eps)]
                    corners.sort(key=lambda p: (abs(a[0]-p[0])+abs(a[1]-p[1])+abs(b[0]-p[0])+abs(b[1]-p[1])))
                    best=[a,corners[0],b]
            new_pts += best[1:]; collided=True
        pts=new_pts
        if not collided: break
    simp=[pts[0]]
    for q in pts[1:]:
        if abs(q[0]-simp[-1][0]) + abs(q[1]-simp[-1][1]) > 1e-3: simp.append(q)
    return simp

def catmull_rom(points, samples_per_seg=12):
    if len(points)<2: return points
    P=[points[0]] + points + [points[-1]]
    out=[]
    for i in range(len(P)-3):
        p0,p1,p2,p3=P[i],P[i+1],P[i+2],P[i+3]
        for j in range(samples_per_seg):
            t=j/float(samples_per_seg); t2=t*t; t3=t2*t
            x=0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y=0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            out.append((x,y))
    out.append(points[-1]); return out

def resample_polyline(poly, N):
    if len(poly)==1: return [poly[0]]*(N+1)
    segs=[]; total=0.0
    for i in range(len(poly)-1):
        a,b=poly[i],poly[i+1]; L=((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
        segs.append((a,b,L)); total+=L
    if total<1e-6: return [poly[0]]*(N+1)
    targets=[total*k/float(N) for k in range(N+1)]
    out=[]; acc=0.0; idx=0; cur=segs[0][0]; out.append(cur)
    for s in targets[1:]:
        while idx<len(segs) and acc+segs[idx][2] < s:
            acc += segs[idx][2]; cur = segs[idx][1]; idx += 1
        if idx>=len(segs): out.append(segs[-1][1]); continue
        a,b,L=segs[idx]; t=(s-acc)/max(L,1e-9)
        x=a[0]+t*(b[0]-a[0]); y=a[1]+t*(b[1]-a[1]); out.append((x,y))
    return out

def heading_speed_curvature(points, dt, v_target=0.4, L=0.26):
    # 由几何曲率估计 delta_cmd
    xs, ys, ths, vs, deltas = [], [], [], [], []
    def wrap(a):
        while a>np.pi: a-=2*np.pi
        while a<-np.pi: a+=2*np.pi
        return a
    for i,p in enumerate(points):
        xs.append(p[0]); ys.append(p[1])
        if i==0: dx=points[1][0]-p[0]; dy=points[1][1]-p[1]
        else:    dx=p[0]-points[i-1][0]; dy=p[1]-points[i-1][1]
        th= np.arctan2(dy, dx+1e-12); ths.append(th)
    for i in range(len(points)):
        if i==0 or i==len(points)-1:
            kappa=0.0
        else:
            th0=ths[i-1]; th1=ths[i]; th2=ths[i+1] if i+1<len(ths) else th1
            dth = wrap(th2 - th0) * 0.5
            ds  = ((points[i+1][0]-points[i][0])**2 + (points[i+1][1]-points[i][1])**2)**0.5 if i+1<len(points) else \
                  ((points[i][0]-points[i-1][0])**2 + (points[i][1]-points[i-1][1])**2)**0.5
            kappa = dth / max(ds, 1e-4)
        delta_cmd = np.arctan(np.clip(L * kappa, -10.0, 10.0))
        deltas.append(delta_cmd)
    # 速度：按段长 / dt，夹在 [0.05, v_target]
    for i in range(len(points)):
        if i==0:
            dx=points[1][0]-points[0][0]; dy=points[1][1]-points[0][1]
        else:
            dx=points[i][0]-points[i-1][0]; dy=points[i][1]-points[i-1][1]
        dist=(dx*dx+dy*dy)**0.5
        vs.append(max(min(dist/dt, v_target), 0.05))
    return xs, ys, ths, vs, deltas

def build_initial_guess(start, goal, task, bev_patch, N, dt, v_target=0.4):
    rects=[]
    # task.masks (rect->poly)
    for m in (task.get("masks") or []):
        if str(m.get("type","rect")).lower()=="rect":
            cx,cy,hx,hy = aabb_from_polygon(
                [[m["center"][0]-0.5*m["size"][0], m["center"][1]-0.5*m["size"][1]],
                 [m["center"][0]+0.5*m["size"][0], m["center"][1]-0.5*m["size"][1]],
                 [m["center"][0]+0.5*m["size"][0], m["center"][1]+0.5*m["size"][1]],
                 [m["center"][0]-0.5*m["size"][0], m["center"][1]+0.5*m["size"][1]]], 0.0)
            rects.append((cx, cy, hx + 0.35, hy + 0.35))  # <== 补上！加点安全余量
    # patch.spatial_masks
    if bev_patch:
        masks=list(bev_patch.get("spatial_masks") or [])
        if bev_patch.get("risk_map"):
            masks += risk_to_masks(bev_patch["risk_map"], thr=0.35, min_area=6)
        for m in masks:
            if str(m.get("mode", "keepout")).lower()!="keepout": continue
            pts = m.get("points", [])
            if not pts: continue
            infl = float(m.get("margin_m", 0.2)) + 0.45
            rects.append(rect_from_points(pts, extra_margin=infl))
    start_xy=(start[0], start[1]); goal_xy=(goal[0], goal[1])
    coarse = route_around_rects(start_xy, goal_xy, rects)
    smooth = catmull_rom(coarse, samples_per_seg=12)
    pts    = resample_polyline(smooth, N)
    xs,ys,ths,vs,delta_cmds = heading_speed_curvature(pts, dt, v_target=v_target)

    nx, nu = 5, 2
    X_init = np.zeros((nx, N+1)); U_init = np.zeros((nu, N))
    for k in range(N+1):
        X_init[0,k]=xs[k]; X_init[1,k]=ys[k]; X_init[2,k]=ths[k]
        X_init[3,k]=vs[k]; X_init[4,k]=0.0
    for k in range(N):
        dv  = X_init[3,k+1]-X_init[3,k]
        U_init[0,k] = np.clip(dv/dt, -0.4, 0.6)                 # 加速度
        U_init[1,k] = np.clip(delta_cmds[k], -0.6, 0.6)         # 直接当作 delta_cmd

    # 用 Ackermann 动力学正推，保证可行
    model = AckermannModel()
    X_sim = np.zeros_like(X_init); X_sim[:,0]=np.array(start)
    for k in range(N):
        xk=ca.DM(X_sim[:,k]); uk=ca.DM(U_init[:,k])
        xnext=model.forward(xk,uk,dt).full().ravel()
        X_sim[:,k+1]=np.array(xnext,dtype=float)
    return X_sim, U_init

def rect_to_poly(center, size, yaw=0.0):
    cx, cy = float(center[0]), float(center[1])
    hx, hy = 0.5*float(size[0]), 0.5*float(size[1])
    return [[cx-hx, cy-hy], [cx+hx, cy-hy], [cx+hx, cy+hy], [cx-hx, cy+hy]]

def collect_polygon_masks(task: Dict[str, Any], bev_patch: Dict[str, Any]):
    polys=[]
    for m in task.get("masks", []) or []:
        if str(m.get("type","rect")).lower()=="rect":
            poly=rect_to_poly(m.get("center",[0,0]), m.get("size",[1,1]), m.get("yaw",0.0))
            polys.append({"id": m.get("id", f"task_rect_{len(polys)}"),
                          "shape":"polygon","mode":"keepout","hardness":"soft",
                          "points":poly,"margin_m":0.0,"weight":0.0})
    if bev_patch and (bev_patch.get("spatial_masks") is not None):
        polys += list(bev_patch.get("spatial_masks") or [])
    return polys

# =============== 主函数 ===============
def build_and_solve(task: Dict[str, Any], bev_patch: Dict[str, Any] = None,
                    out_csv="results/traj.csv", out_png="results/traj.png",
                    debug_save_init="results/init_guess.png"):

    start=_pad5(task.get("start",[0,0,0,0,0]))
    goal =_pad5(task.get("goal", [5,0,0,0,0]))
    N  = int(task.get("horizon", 60))
    dt = float(task.get("dt", 0.05))
    w_state = task.get("weights", {}).get("state", [2.5,2.5,0.2,0.02,0.02])
    w_ctrl  = task.get("weights", {}).get("control", [0.10,0.10])
    term_scale = float(task.get("terminal_scale", 3.0))
    term_mode  = str(task.get("terminal_box_mode", "soft")).lower()
    term_eps   = float(task.get("terminal_box_eps", 0.25))
    u_rate_w   = float(task.get("u_rate_weight", 0.5))
    enable_no_backtrack = bool(task.get("no_backtrack", False))
    vmin_warmup = float(task.get("vmin_warmup", 0.0))
    progress_min = float(task.get("progress_min_per_step", 0.0))  # 新增，可为 0

    if bev_patch is None:
        print("[DEBUG] no patch provided (patch=None)")

    model = AckermannModel()
    nx, nu = model.nx, model.nu
    X = ca.MX.sym("X", nx, N+1)
    U = ca.MX.sym("U", nu, N)

    g_list, lbg, ubg = [], [], []
    obj = 0

    # 初始状态等式
    g_list.append(X[:,0] - ca.DM(start)); lbg += [0.0]*nx; ubg += [0.0]*nx

    Q  = ca.diag(ca.DM(w_state))
    R  = ca.diag(ca.DM(w_ctrl))
    Qf = Q * term_scale

    print(f"[OCP-BEV] goal=({goal[0]:.3f},{goal[1]:.3f}), N={N}, dt={dt}")

    jerk_w = 0.5
    for k in range(N):
        xk=X[:,k]; uk=U[:,k]
        xnext=model.forward(xk,uk,dt)
        g_list.append(X[:,k+1] - xnext); lbg += [0.0]*nx; ubg += [0.0]*nx

        tau = (k+1)/float(N+1)
        x_ref=(1-tau)*start[0] + tau*goal[0]
        y_ref=(1-tau)*start[1] + tau*goal[1]
        ref=ca.DM([x_ref,y_ref,0.0,0.0,0.0])
        obj += ca.mtimes([(xk-ref).T, Q, (xk-ref)]) + ca.mtimes([uk.T, R, uk])

        if u_rate_w>0 and k>=1:
            du = U[:,k] - U[:,k-1]; obj += u_rate_w * ca.mtimes(du.T, du)
        if k>=1:
            ddelta = U[1,k] - U[1,k-1]; obj += jerk_w * (ddelta*ddelta)

    # 末端代价 + 可选硬盒
    xN = X[:,-1]
    obj += ca.mtimes([(xN - ca.DM(goal)).T, Qf, (xN - ca.DM(goal))]) + 2.0*(xN[3]**2) + 1.0*(xN[4]**2)
    if term_mode=="hard":
        for e in [(X[0,-1]-goal[0])-term_eps, -(X[0,-1]-goal[0])-term_eps,
                  (X[1,-1]-goal[1])-term_eps, -(X[1,-1]-goal[1])-term_eps]:
            g_list.append(e); lbg.append(-ca.inf); ubg.append(0.0)
    else:
        obj += 50.0 * ((X[0,-1]-goal[0])**2 + (X[1,-1]-goal[1])**2)

    # ---------- BEV masks ----------
    masks_poly = collect_polygon_masks(task, bev_patch)
    print(f"[DEBUG] collected polygon masks: {len(masks_poly)}")
    task_costs = list(task.get("costs") or [])
    print(f"[DEBUG] task cost items: {len(task_costs)}")

    for idx, m in enumerate(masks_poly):
        if str(m.get("shape","polygon")).lower()!="polygon": continue
        mode = str(m.get("mode","keepout")).lower()
        hard = str(m.get("hardness","soft")).lower()
        pts  = m.get("points", [])
        if not pts: continue
        cx,cy,hx,hy = aabb_from_polygon(pts, 0.0)
        base_margin = float(m.get("margin_m", 0.2))
        w = float(m.get("weight", 8.0))
        print(f"[DEBUG] add {mode}:{hard} @ aabb=({cx:.3f},{cy:.3f},{hx:.3f},{hy:.3f})")
        added=0
        for k in range(N+1):
            xk = X[:,k]
            margin_k = base_margin + 0.25*ca.fabs(xk[3])
            sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)
            if mode=="keepout":
                if hard=="hard":
                    g_list.append(sd - margin_k); lbg.append(0.0); ubg.append(ca.inf)
                else:
                    obj += soft_keepout(sd, margin_k, w=w)
            elif mode=="keepin":
                if hard=="hard":
                    g_list.append(-(sd + margin_k)); lbg.append(0.0); ubg.append(ca.inf)
                else:
                    obj += soft_keepin(sd, margin_k, w=w)
            added += 1
        print(f"[DEBUG] mask {idx} added_terms: {added}")

    # task.costs -> barrier / soft
    def find_mask_polygon(mask_id):
        for mm in masks_poly:
            if str(mm.get("id",""))==str(mask_id): return mm
        return None
    for cidx,c in enumerate(task_costs):
        ctype=str(c.get("type","")).lower()
        if ctype not in ("bev_mask_barrier","bev_mask_soft"):
            print(f"[DEBUG] skip unsupported cost[{cidx}] type={ctype}"); continue
        mid=c.get("mask_id",None); mref=find_mask_polygon(mid) if mid is not None else None
        if mref is None: print(f"[DEBUG] cost[{cidx}] cannot find mask_id={mid} -> SKIP"); continue
        pts = mref.get("points", [])
        if not pts:
            print(f"[DEBUG] cost[{cidx}] mask_id={mid} empty -> SKIP")
            continue
        cx,cy,hx,hy=aabb_from_polygon(pts,0.0); weight=float(c.get("weight",200.0))
        if ctype=="bev_mask_soft":
            base_margin=float(c.get("margin",0.20))
            for k in range(N+1):
                xk=X[:,k]; margin_k = base_margin + 0.25*ca.fabs(xk[3])
                sd=sdf_aabb(xk[0],xk[1],cx,cy,hx,hy)
                obj += weight * smooth_hinge(margin_k - sd, rho=0.02)
            print(f"[DEBUG] add cost[{cidx}] soft mask={mid} weight={weight}")
       
        else:  # ctype == "bev_mask_barrier"
            barr = (c.get("barrier", {}) or {})
            r    = float(barr.get("radius", 0.20))
            eps  = float(barr.get("epsilon", 1e-3))
            eps_safe = max(eps, 1e-3)  # 下界，避免 log(≤0)

            for k in range(N + 1):
                xk = X[:, k]
                sd = sdf_aabb(xk[0], xk[1], cx, cy, hx, hy)

        # 关键：把 sd 夹到 (r + eps_safe) 以上，完全消除 log 的负/零参
                sd_clamped = ca.fmax(sd, r + eps_safe)

        # 同伦：早期权重小、末期逐步加大，避免“顶墙”数值不稳
                tau = k / float(N)                 # 0 → 1
                w_k = weight * (0.1 + 0.9 * tau)   # 10% → 100%

        # 安全对数屏障
                obj += w_k * (-ca.log(sd_clamped - r + eps_safe))
            print(f"[DEBUG] add cost[{cidx}] barrier mask={mid} weight={weight} r={r} eps={eps}")

    # 可选：单调前进 + 最小前进分量
    if enable_no_backtrack or progress_min>0.0:
        gdir = ca.DM([goal[0]-start[0], goal[1]-start[1]])
        gdir = gdir / (1e-9 + ca.norm_2(gdir))
        eps_back = 0.0
        for k in range(N):
            step = X[0:2,k+1] - X[0:2,k]
            forward = ca.dot(step, gdir)
            # no_backtrack: forward >= 0
            if enable_no_backtrack:
                g_list.append(forward + eps_back); lbg.append(0.0); ubg.append(ca.inf)
            # progress_min_per_step: forward >= progress_min
            if progress_min>0.0:
                g_list.append(forward - progress_min); lbg.append(0.0); ubg.append(ca.inf)

    # v 最小热启动段
    if vmin_warmup>0:
        warmN = max(1, N//3)
        for k in range(warmN):
            g_list.append(X[3,k] - vmin_warmup); lbg.append(0.0); ubg.append(ca.inf)

    # 控制边界
    a_min,a_max = -0.3, +0.6
    d_min,d_max = -0.6, +0.6
    for k in range(N):
        g_list.append(U[:,k] - ca.DM([a_min, d_min])); lbg += [0.0,0.0]; ubg += [ca.inf,ca.inf]
        g_list.append(ca.DM([a_max, d_max]) - U[:,k]); lbg += [0.0,0.0]; ubg += [ca.inf,ca.inf]

    # 状态边界（安全盒）
    x_min,x_max = -10.0, 10.0
    y_min,y_max = -10.0, 10.0
    th_min,th_max = -np.pi, np.pi
    v_min,v_max_state = 0.0, 1.8
    d_min_s,d_max_s   = -0.8, 0.8
    for k in range(N+1):
        xk=X[:,k]
        for e in [xk[0]-x_min, x_max-xk[0], xk[1]-y_min, y_max-xk[1],
                  xk[2]-th_min, th_max-xk[2], xk[3]-v_min, v_max_state-xk[3],
                  xk[4]-d_min_s, d_max_s-xk[4]]:
            g_list.append(e); lbg.append(0.0); ubg.append(ca.inf)

    # 几何步长上限（速度帽）
    v_max_cfg = float(task.get("vmax_geom", 1.0))
    T_total = N*dt
    dist_go = max(1e-6, np.hypot(goal[0]-start[0], goal[1]-start[1]))
    v_needed = 1.2 * (dist_go / max(T_total, 1e-6))
    v_max = max(v_max_cfg, v_needed)
    for k in range(N):
        move = ca.norm_2(X[0:2,k+1] - X[0:2,k])
        g_list.append(v_max*dt - move); lbg.append(0.0); ubg.append(ca.inf)

    # 打包 NLP
    vars_ = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
    g = ca.vertcat(*g_list)
    nlp = {"x": vars_, "f": obj, "g": g}

    # 初值
    X_init, U_init = build_initial_guess(start, goal, task, bev_patch, N=N, dt=dt, v_target=0.40)
    x0_guess = np.concatenate([X_init.ravel(), U_init.ravel()])

    if debug_save_init:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(X_init[0,:], X_init[1,:], marker=".")
        if bev_patch:
            masks=bev_patch.get("spatial_masks") or []
            for m in masks:
                if str(m.get("shape","polygon")).lower()!="polygon": continue
                pts = m.get("points", [])
                if pts:
                    xs=[p[0] for p in pts]+[pts[0][0]]
                    ys=[p[1] for p in pts]+[pts[0][1]]
                    plt.plot(xs,ys)
        for m in (task.get("masks") or []):
            if str(m.get("type","rect")).lower()=="rect":
                poly=rect_to_poly(m.get("center",[0,0]), m.get("size",[1,1]), m.get("yaw",0.0))
                xs=[p[0] for p in poly]+[poly[0][0]]
                ys=[p[1] for p in poly]+[poly[0][1]]
                plt.plot(xs,ys)
        plt.axis("equal"); plt.title("Initial guess")
        os.makedirs(os.path.dirname(debug_save_init), exist_ok=True)
        plt.savefig(debug_save_init, bbox_inches="tight")

    # IPOPT 选项
    opts = {
        "ipopt.print_level": 0, "print_time": 0,
        "ipopt.max_iter": 1200,
        "ipopt.tol": 1e-5, "ipopt.acceptable_tol": 1e-3,
        "ipopt.acceptable_iter": 10,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.mu_oracle": "quality-function",
        "ipopt.linear_solver": "mumps",
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.nlp_scaling_method": "gradient-based",
        "ipopt.bound_push": 1e-6, "ipopt.bound_frac": 1e-6,
        "ipopt.constr_viol_tol": 1e-4,
        "ipopt.diverging_iterates_tol": 1e20,
        "ipopt.honor_original_bounds": "yes",
        "ipopt.warm_start_init_point": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # ====== 求解 ======
    sol = solver(lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg), x0=x0_guess)
    stats = solver.stats()

    z = sol["x"].full().ravel()
    Xsol = z[: (nx*(N+1))].reshape((nx, N+1))
    Usol = z[(nx*(N+1)) :].reshape((nu, N))

    print(f"[OCP-BEV] xN=({float(Xsol[0,-1]):.3f},{float(Xsol[1,-1]):.3f})")
    print(f"[DEBUG] IPOPT return_status: {stats.get('return_status','?')}")

    # ====== 自检 ======
    try:
        g_fun = ca.Function("g_fun", [vars_], [g])
        g_val = g_fun(sol["x"]).full().ravel()
        eq_max_abs=0.0; ineq_min_margin=float("+inf")
        for gi, lb, ub in zip(g_val, lbg, ubg):
            lb=float(lb); ub=float(ub)
            if np.isfinite(lb) and np.isfinite(ub) and abs(ub-lb)<1e-12:
                eq_max_abs = max(eq_max_abs, abs(gi-lb))
            else:
                if (abs(lb)<1e-12) and (not np.isfinite(ub)):
                    ineq_min_margin = min(ineq_min_margin, float(gi))
                else:
                    to_lb = gi - lb if np.isfinite(lb) else float("+inf")
                    to_ub = ub - gi if np.isfinite(ub) else float("+inf")
                    ineq_min_margin = min(ineq_min_margin, float(min(to_lb,to_ub)))
        print(f"[CHECK] equality max |violation| = {eq_max_abs:.3e}")
        if ineq_min_margin==float("+inf"):
            print("[CHECK] no inequality constraints detected")
        else:
            print(f"[CHECK] inequality min margin  = {ineq_min_margin:.4f}  (negative => crossed)")
    except Exception as e:
        print("[WARN] feasibility check failed:", repr(e))

    # ====== 输出 ======
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv
    with open(out_csv,"w",newline="",encoding="utf-8") as f:
        wr=csv.writer(f); wr.writerow(["k","x","y","theta","v","delta"])
        for k in range(N+1):
            wr.writerow([k] + [float(v) for v in Xsol[:,k]])

    if out_png:
        import matplotlib.pyplot as plt
        plt.figure(); plt.plot(Xsol[0,:], Xsol[1,:], marker=".")
        if bev_patch:
            masks=bev_patch.get("spatial_masks") or []
            if bev_patch.get("risk_map"): masks += risk_to_masks(bev_patch["risk_map"])
            for m in masks:
                if str(m.get("shape","polygon")).lower()!="polygon": continue
                pts = m.get("points", [])
                if pts:
                    xs=[p[0] for p in pts]+[pts[0][0]]
                    ys=[p[1] for p in pts]+[pts[0][1]]
                    plt.plot(xs,ys)
        for m in (task.get("masks") or []):
            if str(m.get("type","rect")).lower()=="rect":
                poly=rect_to_poly(m.get("center",[0,0]), m.get("size",[1,1]), m.get("yaw",0.0))
                xs=[p[0] for p in poly]+[poly[0][0]]
                ys=[p[1] for p in poly]+[poly[0][1]]
                plt.plot(xs,ys)
        plt.axis("equal"); plt.title("Trajectory with BEV masks")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, bbox_inches="tight")

    return Xsol, Usol

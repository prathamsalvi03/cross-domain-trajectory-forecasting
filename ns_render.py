#!/usr/bin/env python3
"""
Step 3 of the nuScenes demo: score and render the model's predictions.

  python3 ns_render.py <dataroot> ns_work/meta.json ns_work/predictions.txt \
          --out ns_videos --label "pedestrian-only model"

Prints a zero-shot results table (model vs. constant-velocity baseline) and
writes an MP4 per selected scene plus showcase.mp4:
  left  = front camera with trajectories projected onto the ground plane
  right = top-down view around the ego car (forward is up)

Legend: white = observed 4 s, green dashed = what really happened,
colored = model's 6 predicted futures (brighter = more likely),
blue = ego car's own future path, red ring = pedestrian predicted to come
within --near metres of the ego car's path.
"""
import argparse, json, math, os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Polygon, Circle
from PIL import Image

BG, FG = "#0e1117", "#e6edf3"
MODE_COLORS = ["#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#4dabf7", "#da77f2"]
GT_C, OBS_C, EGO_C, NEAR_C = "#37d67a", "#ffffff", "#3b82f6", "#ff3b30"


# ---------------------------------------------------------------- geometry
def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def global_to_ego(pts, fr):
    R = quat_to_R(fr["ego"]["rotation"])
    return (pts - np.array(fr["ego"]["translation"])) @ R  # == R^T (p - t)


def project(pts_xyz, fr, scale):
    """global (M,3) -> pixel (M,2), valid (M,)"""
    p_ego = global_to_ego(pts_xyz, fr)
    Rc = quat_to_R(fr["cam"]["rotation"])
    p_cam = (p_ego - np.array(fr["cam"]["translation"])) @ Rc
    K = np.array(fr["cam"]["intrinsic"])
    uvw = p_cam @ K.T
    depth = p_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3] * scale
    return uv, depth > 1.0


# ---------------------------------------------------------------- data
def load_predictions(path):
    preds = {}
    with open(path) as f:
        W, N, K, P = map(int, f.readline().split())
        for line in f:
            v = line.split()
            if not v:
                continue
            w, n = int(v[0]), int(v[1])
            vals = np.array(v[2:], dtype=float)
            probs = vals[:K]
            disp = vals[K:].reshape(K, P, 2)
            preds[(w, n)] = (probs, disp)
    return preds, K, P


def ego_future(meta, fr, P, dt):
    tr = meta["ego_tracks"][fr["scene"]]
    t = np.array(tr["t"]); xy = np.array(tr["xy"])
    if len(t) < 2:
        return None
    tf = fr["t"] + dt * np.arange(1, P + 1)
    tf = tf[tf <= t[-1] + 1e-6]
    if len(tf) == 0:
        return None
    return np.stack([np.interp(tf, t, xy[:, 0]), np.interp(tf, t, xy[:, 1])], 1)


def agent_entries(meta, preds, fr, dt, P):
    """Per pedestrian in this frame: obs, gt, predicted absolute modes."""
    out = []
    for n, ag in enumerate(fr["agents"]):
        key = (fr["window"], n)
        if key not in preds:
            continue
        probs, disp = preds[key]
        last = np.array(ag["obs"][-1])
        modes = last[None, None, :] + disp            # (K,P,2)
        out.append({"ag": ag, "last": last, "probs": probs, "modes": modes,
                    "expected": (probs[:, None, None] * modes).sum(0) / max(probs.sum(), 1e-9)})
    return out


# ---------------------------------------------------------------- metrics
def evaluate(meta, preds, dt, P):
    m_min_ade, m_min_fde, m_top_ade, m_top_fde, cv_ade, cv_fde = [], [], [], [], [], []
    for fr in meta["frames"]:
        if fr["window"] < 0:
            continue
        for e in agent_entries(meta, preds, fr, dt, P):
            gt = e["ag"]["gt"]
            if any(g is None for g in gt):
                continue
            gt = np.array(gt)
            err = np.linalg.norm(e["modes"] - gt[None], axis=-1)  # (K,P)
            ade_k, fde_k = err.mean(1), err[:, -1]
            m_min_ade.append(ade_k.min()); m_min_fde.append(fde_k.min())
            top = int(np.argmax(e["probs"]))
            m_top_ade.append(ade_k[top]); m_top_fde.append(fde_k[top])
            obs = e["ag"]["obs"]
            v = (np.array(obs[-1]) - np.array(obs[-2])) / dt if obs[-2] is not None else np.zeros(2)
            cv = e["last"] + v * dt * np.arange(1, P + 1)[:, None]
            ce = np.linalg.norm(cv - gt, axis=-1)
            cv_ade.append(ce.mean()); cv_fde.append(ce[-1])
    n = len(m_min_ade)
    if n == 0:
        print("No agents with a complete ground-truth future; cannot score.")
        return None
    r = {"n": n,
         "model_min_ade": float(np.mean(m_min_ade)), "model_min_fde": float(np.mean(m_min_fde)),
         "model_top1_ade": float(np.mean(m_top_ade)), "model_top1_fde": float(np.mean(m_top_fde)),
         "cv_ade": float(np.mean(cv_ade)), "cv_fde": float(np.mean(cv_fde))}
    print(f"\nZero-shot nuScenes pedestrians ({n} agent-windows with full {P*dt:.1f}s ground truth)")
    print(f"  {'method':32s} {'ADE (m)':>8s} {'FDE (m)':>8s}")
    print(f"  {'constant velocity':32s} {r['cv_ade']:8.3f} {r['cv_fde']:8.3f}")
    print(f"  {'model, most likely mode':32s} {r['model_top1_ade']:8.3f} {r['model_top1_fde']:8.3f}")
    print(f"  {'model, best of 6 (minADE_6)':32s} {r['model_min_ade']:8.3f} {r['model_min_fde']:8.3f}")
    print("  note: consecutive windows overlap (one per 0.5 s keyframe), so the same")
    print("  pedestrian is counted several times; best-of-6 is not comparable to a")
    print("  single-guess baseline -- compare constant velocity to 'most likely mode'.")
    return r


# ---------------------------------------------------------------- drawing
def segments(points, valid):
    seg, cur = [], []
    for p, ok in zip(points, valid):
        if ok:
            cur.append(p)
        else:
            if len(cur) >= 2:
                seg.append(np.array(cur))
            cur = []
    if len(cur) >= 2:
        seg.append(np.array(cur))
    return seg


def draw_cam(ax, fr, ents, ego_fut, dataroot, scale, near, img_cache):
    path = os.path.join(dataroot, fr["image"])
    if path not in img_cache:
        im = Image.open(path).convert("RGB")
        img_cache.clear()
        img_cache[path] = np.asarray(im.resize((int(im.width * scale), int(im.height * scale))))
    img = img_cache[path]
    H, Wd = img.shape[:2]
    ax.imshow(img)
    ax.set_xlim(0, Wd); ax.set_ylim(H, 0); ax.axis("off")

    def plot_xy(xy, z, **kw):
        pts = np.column_stack([xy, np.full(len(xy), z)])
        uv, ok = project(pts, fr, scale)
        ok &= np.all(np.abs(uv) < 4 * max(H, Wd), axis=1)
        for s in segments(uv, ok):
            ax.plot(s[:, 0], s[:, 1], **kw)
        return uv, ok

    ez = fr["ego"]["translation"][2]
    if ego_fut is not None:
        here = np.array(fr["ego"]["translation"][:2])
        plot_xy(np.vstack([here, ego_fut]), ez, color=EGO_C, lw=6, alpha=0.45, solid_capstyle="round")
    for e in ents:
        ag, z = e["ag"], e["ag"]["z"]
        obs = [p for p in ag["obs"] if p is not None]
        plot_xy(np.array(obs), z, color=OBS_C, lw=2.2)
        gt = [e["last"].tolist()] + ag["gt"]
        gt_ok = [g is not None for g in gt]
        gt_xy = np.array([g if g is not None else [0, 0] for g in gt])
        uv, ok = project(np.column_stack([gt_xy, np.full(len(gt_xy), z)]), fr, scale)
        for s in segments(uv, ok & np.array(gt_ok)):
            ax.plot(s[:, 0], s[:, 1], color=GT_C, lw=2.2, ls="--")
        pmax = max(e["probs"].max(), 1e-9)
        for k in np.argsort(e["probs"]):
            pth = np.vstack([e["last"], e["modes"][k]])
            plot_xy(pth, z, color=MODE_COLORS[k % 6], lw=1.2 + 2.0 * e["probs"][k] / pmax,
                    alpha=0.25 + 0.75 * e["probs"][k] / pmax)
        uv, ok = project(np.array([[*e["last"], z]]), fr, scale)
        if ok[0] and 0 <= uv[0, 0] < Wd and 0 <= uv[0, 1] < H:
            ax.plot(uv[0, 0], uv[0, 1], "o", color=OBS_C, ms=5, mec="black", mew=0.8)
            if e.get("near"):
                ax.add_patch(Circle(uv[0], 14, fill=False, ec=NEAR_C, lw=2.5))


def draw_bev(ax, fr, ents, ego_fut, near, fov_half):
    ax.set_facecolor("#161b22")
    ax.set_xlim(-30, 30); ax.set_ylim(-12, 48); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for v in range(-30, 31, 10):
        ax.axvline(v, color="#30363d", lw=0.5)
    for v in range(-10, 50, 10):
        ax.axhline(v, color="#30363d", lw=0.5)

    def to_bev(xy):  # global xy -> (right, forward) in metres
        pe = global_to_ego(np.column_stack([xy, np.zeros(len(xy))]), fr)
        return np.column_stack([-pe[:, 1], pe[:, 0]])

    r = 48
    ax.add_patch(Polygon([[0, 0], [r * math.tan(fov_half), r], [-r * math.tan(fov_half), r]],
                         closed=True, color="#ffffff", alpha=0.05))
    ax.add_patch(Rectangle((-1.0, -2.25), 2.0, 4.5, color=EGO_C))
    if ego_fut is not None:
        b = to_bev(np.vstack([fr["ego"]["translation"][:2], ego_fut]))
        ax.plot(b[:, 0], b[:, 1], color=EGO_C, lw=5, alpha=0.5)
    for e in ents:
        ag = e["ag"]
        obs = to_bev(np.array([p for p in ag["obs"] if p is not None]))
        ax.plot(obs[:, 0], obs[:, 1], color=OBS_C, lw=1.8)
        gt = [e["last"].tolist()] + ag["gt"]
        ok = [g is not None for g in gt]
        gxy = to_bev(np.array([g if g is not None else [0, 0] for g in gt]))
        for s in segments(gxy, ok):
            ax.plot(s[:, 0], s[:, 1], color=GT_C, lw=1.8, ls="--")
        pmax = max(e["probs"].max(), 1e-9)
        for k in np.argsort(e["probs"]):
            m = to_bev(np.vstack([e["last"], e["modes"][k]]))
            ax.plot(m[:, 0], m[:, 1], color=MODE_COLORS[k % 6], lw=0.8 + 1.6 * e["probs"][k] / pmax,
                    alpha=0.25 + 0.75 * e["probs"][k] / pmax)
        c = to_bev(e["last"][None])[0]
        ax.plot(c[0], c[1], "o", color=OBS_C, ms=4)
        if e.get("near"):
            ax.add_patch(Circle(c, 2.0, fill=False, ec=NEAR_C, lw=2))
    ax.set_title("top-down (ego frame, 10 m grid)", color=FG, fontsize=10)


def mark_near(ents, ego_fut, near):
    n_near = 0
    for e in ents:
        e["near"] = False
        if ego_fut is not None:
            L = min(len(ego_fut), len(e["expected"]))
            d = np.linalg.norm(e["expected"][:L] - ego_fut[:L], axis=-1)
            e["near"] = bool(len(d) and d.min() < near)
        n_near += e["near"]
    return n_near


def render(meta, preds, P, frames, dataroot, out_path, label, metrics, fps, scale, near):
    dt = meta["dt"]
    fr0 = frames[0]
    K = np.array(fr0["cam"]["intrinsic"])
    fov_half = math.atan(K[0, 2] / K[0, 0])
    fig = plt.figure(figsize=(16.8, 6.6), facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[16, 9], wspace=0.03,
                          left=0.01, right=0.99, top=0.86, bottom=0.08)
    ax_cam, ax_bev = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    sub = ""
    if metrics:
        sub = (f"zero-shot on nuScenes pedestrians  |  most-likely-mode ADE {metrics['model_top1_ade']:.2f} m"
               f" vs constant velocity {metrics['cv_ade']:.2f} m  |  best-of-6 ADE {metrics['model_min_ade']:.2f} m")
    head = fig.text(0.01, 0.955, "", color=FG, fontsize=14, fontweight="bold", va="center")
    fig.text(0.01, 0.905, sub, color="#9aa4b2", fontsize=10.5, va="center")
    fig.text(0.01, 0.035,
             "white: observed 4 s    green dashed: actual future    colored: 6 predicted futures (4 s)"
             "    blue: ego car path    red ring: predicted to cross near ego path",
             color="#9aa4b2", fontsize=9.5, va="center")
    cache = {}

    def update(i):
        fr = frames[i]
        ax_cam.clear(); ax_bev.clear()
        ents = agent_entries(meta, preds, fr, dt, P) if fr["window"] >= 0 else []
        ego_fut = ego_future(meta, fr, P, dt)
        n_near = mark_near(ents, ego_fut, near)
        draw_cam(ax_cam, fr, ents, ego_fut, dataroot, scale, near, cache)
        draw_bev(ax_bev, fr, ents, ego_fut, near, fov_half)
        head.set_text(f"Multi-agent pedestrian forecasting  |  C++ / LibTorch GAT + Transformer  |  {label}"
                      f"     [{fr['scene']}  t={fr['t'] - frames[0]['t']:.1f}s"
                      f"  peds={len(ents)}  near-ego={n_near}]")
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    if not shutil.which("ffmpeg"):
        try:  # pip install imageio-ffmpeg ships its own ffmpeg binary
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    if shutil.which("ffmpeg") or plt.rcParams["animation.ffmpeg_path"] != "ffmpeg":
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=6000), dpi=100,
                  savefig_kwargs={"facecolor": BG})
    else:
        out_path = out_path.rsplit(".", 1)[0] + ".gif"
        anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=70,
                  savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataroot")
    ap.add_argument("meta")
    ap.add_argument("predictions")
    ap.add_argument("--out", default="ns_videos")
    ap.add_argument("--label", default="trained on ETH/UCY")
    ap.add_argument("--scenes", nargs="*", help="scene names to render (default: busiest ones)")
    ap.add_argument("--top", type=int, default=3, help="how many busiest scenes in showcase.mp4")
    ap.add_argument("--fps", type=float, default=4.0, help="keyframes are 2 Hz, so 4 = 2x speed")
    ap.add_argument("--scale", type=float, default=0.5, help="camera image downscale for speed")
    ap.add_argument("--near", type=float, default=3.0, help="metres for the near-ego-path flag")
    ap.add_argument("--no_video", action="store_true")
    a = ap.parse_args()

    with open(a.meta) as f:
        meta = json.load(f)
    preds, K, P = load_predictions(a.predictions)
    metrics = evaluate(meta, preds, meta["dt"], P)
    if metrics:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    if a.no_video:
        return

    by_scene = {}
    for fr in meta["frames"]:
        by_scene.setdefault(fr["scene"], []).append(fr)
    busy = sorted(by_scene, key=lambda s: -sum(len(f["agents"]) for f in by_scene[s]))
    chosen = a.scenes or busy[:a.top]
    os.makedirs(a.out, exist_ok=True)
    for s in chosen:
        if s not in by_scene:
            print("unknown scene", s); continue
        render(meta, preds, P, by_scene[s], a.dataroot, os.path.join(a.out, f"{s}.mp4"),
               a.label, metrics, a.fps, a.scale, a.near)
    showcase = [f for s in chosen if s in by_scene for f in by_scene[s]]
    if len(chosen) > 1 and showcase:
        render(meta, preds, P, showcase, a.dataroot, os.path.join(a.out, "showcase.mp4"),
               a.label, metrics, a.fps, a.scale, a.near)
    print("\nper-scene pedestrian load:", ", ".join(
        f"{s}={sum(len(f['agents']) for f in by_scene[s])}" for s in busy[:8]))


if __name__ == "__main__":
    main()

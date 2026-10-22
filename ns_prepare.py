#!/usr/bin/env python3
"""
Step 1 of the nuScenes demo: turn nuScenes annotations into model input windows.

  python3 ns_prepare.py <dataroot> --version v1.0-mini --out ns_work

<dataroot> is the folder that contains v1.0-mini/ and samples/ (camera images).
Writes:
  ns_work/windows.txt   plain-text input for the C++ predict_windows tool
  ns_work/meta.json     everything the renderer needs (images, poses, tracks)

How the windows are built
  * nuScenes annotates at 2 Hz. Tracks are linearly interpolated onto a 0.4 s
    grid so the step size matches the ETH/UCY data the model was trained on.
    10 observed steps = 4 s of history, 10 predicted steps = 4 s of future.
  * One window per camera keyframe, sampled relative to that frame's own
    timestamp (last observed step == the image), so overlays line up exactly
    and a video of consecutive keyframes shows predictions updating live.
  * Pedestrians only. Each window is shifted so its agents are centred near 0,
    like ETH/UCY scene coordinates. Up to n_pad agents, closest to ego first.
No nuscenes-devkit needed; only the JSON tables and the camera images.
"""
import argparse, collections, json, math, os
import numpy as np

PED_OK = {"adult", "child", "construction_worker", "police_officer"}


def load(d, name):
    with open(os.path.join(d, name + ".json")) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataroot")
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--out", default="ns_work")
    ap.add_argument("--camera", default="CAM_FRONT")
    ap.add_argument("--obs", type=int, default=10)
    ap.add_argument("--pred", type=int, default=10)
    ap.add_argument("--dt", type=float, default=0.4)
    ap.add_argument("--n_pad", type=int, default=30)
    ap.add_argument("--max_gap", type=float, default=1.6,
                    help="seconds; longer annotation gaps are treated as missing, not interpolated")
    ap.add_argument("--scenes", nargs="*", help="optional scene names, e.g. scene-0061")
    a = ap.parse_args()

    tdir = os.path.join(a.dataroot, a.version)
    T = {n: load(tdir, n) for n in ["category", "instance", "scene", "sample", "sample_annotation",
                                     "sample_data", "calibrated_sensor", "ego_pose", "sensor"]}
    cat = {c["token"]: c["name"] for c in T["category"]}
    inst_cat = {i["token"]: cat[i["category_token"]] for i in T["instance"]}
    sensor = {s["token"]: s for s in T["sensor"]}
    calib = {c["token"]: c for c in T["calibrated_sensor"]}
    ego = {e["token"]: e for e in T["ego_pose"]}
    samples = {s["token"]: s for s in T["sample"]}

    cam_sd = {}
    for sd in T["sample_data"]:
        if not sd.get("is_key_frame"):
            continue
        ch = sensor[calib[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
        if ch == a.camera:
            cam_sd[sd["sample_token"]] = sd
    ann_by_sample = collections.defaultdict(list)
    for an in T["sample_annotation"]:
        ann_by_sample[an["sample_token"]].append(an)

    scenes = T["scene"]
    if a.scenes:
        scenes = [s for s in scenes if s["name"] in set(a.scenes)]

    windows, frames, ego_tracks = [], [], {}
    for sc in scenes:
        toks, tok = [], sc["first_sample_token"]
        while tok:
            toks.append(tok)
            tok = samples[tok]["next"]
        et, exy = [], []
        for t in toks:
            sd = cam_sd.get(t)
            if sd:
                et.append(sd["timestamp"] / 1e6)
                exy.append(ego[sd["ego_pose_token"]]["translation"][:2])
        ego_tracks[sc["name"]] = {"t": et, "xy": exy}

        raw = collections.defaultdict(list)
        for t in toks:
            ts = samples[t]["timestamp"] / 1e6
            for an in ann_by_sample[t]:
                name = inst_cat[an["instance_token"]]
                if name.startswith("human.pedestrian.") and name.split(".")[-1] in PED_OK:
                    x, y, z = an["translation"]
                    raw[an["instance_token"]].append((ts, x, y, z - an["size"][2] / 2.0))
        tracks = {}
        for inst, rows in raw.items():
            if len(rows) >= 2:
                r = np.array(sorted(rows))
                tracks[inst] = {"r": r, "z": float(np.median(r[:, 3]))}

        def sample_track(tr, times):
            """interpolated xy at times + validity (inside track, not inside a long gap)"""
            r = tr["r"]
            xy = np.stack([np.interp(times, r[:, 0], r[:, 1]), np.interp(times, r[:, 0], r[:, 2])], 1)
            valid = (times >= r[0, 0] - 1e-3) & (times <= r[-1, 0] + 1e-3)
            for g in np.where(np.diff(r[:, 0]) > a.max_gap)[0]:
                valid &= ~((times > r[g, 0] + 1e-3) & (times < r[g + 1, 0] - 1e-3))
            return xy, valid

        for tok in toks:
            sd = cam_sd.get(tok)
            if not sd:
                continue
            tk = sd["timestamp"] / 1e6
            e, c = ego[sd["ego_pose_token"]], calib[sd["calibrated_sensor_token"]]
            frame = {"scene": sc["name"], "t": tk, "image": sd["filename"],
                     "ego": {"translation": e["translation"], "rotation": e["rotation"]},
                     "cam": {"translation": c["translation"], "rotation": c["rotation"],
                             "intrinsic": c["camera_intrinsic"]},
                     "window": -1, "agents": []}
            # window grid anchored on this camera frame: last observed step == now
            t_obs = tk - a.dt * np.arange(a.obs - 1, -1, -1)
            t_fut = tk + a.dt * np.arange(1, a.pred + 1)
            exy0 = np.array(e["translation"][:2])
            cand = []
            for i, tr in tracks.items():
                oxy, ov = sample_track(tr, t_obs)
                if ov[-1] and ov.sum() >= 2:
                    cand.append((np.linalg.norm(oxy[-1] - exy0), i, oxy, ov))
            cand.sort(key=lambda x: x[0])
            cand = cand[:a.n_pad]
            if cand:
                center = np.mean([cd[2][-1] for cd in cand], axis=0)
                pos = np.zeros((a.obs, a.n_pad, 2))
                m = np.zeros((a.obs, a.n_pad))
                for n, (_, i, oxy, ov) in enumerate(cand):
                    pos[:, n] = (oxy - center) * ov[:, None]
                    m[:, n] = ov
                    fxy, fv = sample_track(tracks[i], t_fut)
                    frame["agents"].append({
                        "id": i[:8], "z": tracks[i]["z"],
                        "obs": [oxy[k].tolist() if ov[k] else None for k in range(a.obs)],
                        "gt": [fxy[k].tolist() if fv[k] else None for k in range(a.pred)]})
                frame["window"] = len(windows)
                frame["center"] = center.tolist()
                windows.append(np.concatenate([pos, m[..., None]], -1))
            frames.append(frame)

    os.makedirs(a.out, exist_ok=True)
    W = len(windows)
    with open(os.path.join(a.out, "windows.txt"), "w") as f:
        f.write(f"{W} {a.n_pad} {a.obs}\n")
        if W:
            np.savetxt(f, np.stack(windows).reshape(-1, 3), fmt="%.4f")
    meta = {"obs": a.obs, "pred": a.pred, "dt": a.dt, "n_pad": a.n_pad, "camera": a.camera,
            "frames": frames, "ego_tracks": ego_tracks}
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f)

    n_ag = [len(fr["agents"]) for fr in frames if fr["window"] >= 0]
    print(f"scenes={len(scenes)}  keyframes={len(frames)}  windows with pedestrians={W}")
    if n_ag:
        print(f"pedestrians per window: mean={np.mean(n_ag):.1f} max={max(n_ag)}")
    print("wrote", os.path.join(a.out, "windows.txt"), "and meta.json")


if __name__ == "__main__":
    main()

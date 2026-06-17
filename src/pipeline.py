"""端到端流程：追踪两浮子 -> 标定 -> 平滑/峰值 -> CSV/曲线/标注视频/报告。"""
import csv
import os

import cv2
import numpy as np

from .analyzer import detect_peaks, smooth
from .calibration import Calibration
from .float_tracker import FloatTracker
from . import visualize as viz


def _open(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video}")
    return cap


def run(cfg, render_video=True):
    video = cfg["video"]
    out_dir = cfg.get("output_dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    stride = int(cfg.get("stride", 1))
    names = [f["name"] for f in cfg["floats"]]

    cap = _open(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cal = Calibration(cfg["calibration"])

    # 模板帧
    tf = int(cfg.get("tracker", {}).get("template_frame", 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, tf)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("读取模板帧失败")
    tracker = FloatTracker(cfg, cv2.cvtColor(first, cv2.COLOR_BGR2GRAY))

    # ---------- Pass 1: 逐帧追踪 ----------
    print(f"[1/3] 追踪浮子 ({nfr} 帧, fps={fps:.1f}) ...")
    rec = {nm: {"t": [], "frame": [], "cx": [], "cy": [], "conf": []} for nm in names}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            res = tracker.update(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
            for nm in names:
                cx, cy, conf = res[nm]
                rec[nm]["t"].append(idx / fps)
                rec[nm]["frame"].append(idx)
                rec[nm]["cx"].append(cx)
                rec[nm]["cy"].append(cy)
                rec[nm]["conf"].append(conf)
        idx += 1
    cap.release()

    # ---------- 标定 + 平滑 + 峰值 ----------
    print("[2/3] 标定(浮子锚定)、平滑、峰值检测 ...")
    an = cfg.get("analysis", {})
    cal_cfg = cfg["calibration"]
    rest_n = max(1, int(round(fps * cal_cfg.get("rest_seconds", 2.0))))
    scale_factor = float(cal_cfg.get("scale_factor", 1.0))
    rest_h_map = {f["name"]: float(f.get("rest_height_m", 2.0)) for f in cfg["floats"]}

    series = {}
    float_cal = {}
    for nm in names:
        t = np.array(rec[nm]["t"])
        cx = np.array(rec[nm]["cx"])
        cy = np.array(rec[nm]["cy"])
        # 浮子自身静止基准(视差偏移已被吸收): 开头静止段中值
        x_med = float(np.median(cx))
        rest_y = float(np.median(cy[:rest_n]))
        ppm = cal.px_per_m(x_med) * scale_factor      # 该列像素/米
        rest_h = rest_h_map[nm]
        # 锚定: 静止=rest_h, 上移(y减小)为升高
        h_raw = rest_h + (rest_y - cy) / ppm
        float_cal[nm] = {"rest_y": rest_y, "ppm": ppm,
                         "rest_h": rest_h, "x": x_med}
        h = smooth(h_raw, fps,
                   an.get("median_seconds", 0.05),
                   an.get("smooth_seconds", 0.15))
        stats = detect_peaks(t, h,
                             prominence=an.get("min_prominence_m", 0.015),
                             min_distance_s=an.get("min_peak_distance_s", 0.3),
                             fps=fps)
        series[nm] = {"t": t, "h_raw": h_raw, "h": h, "stats": stats,
                      "cy": cy, "cx": cx, "conf": np.array(rec[nm]["conf"]),
                      "frame": np.array(rec[nm]["frame"])}

    _write_csv(series, names, os.path.join(out_dir, "float_heights.csv"))
    viz.plot_curves(
        {nm: series[nm] for nm in names},
        os.path.join(out_dir, "float_curves.png"))
    _write_report(series, names, cal, cfg,
                  os.path.join(out_dir, "peak_report.md"))

    # ---------- Pass 2: 标注视频 ----------
    if render_video:
        print("[3/3] 渲染标注视频 ...")
        _render(video, series, names, float_cal, cfg,
                os.path.join(out_dir, "annotated.mp4"), fps, W, H, stride)
    else:
        print("[3/3] 跳过标注视频 (--no-video)")

    print("\n完成。产出位于:", out_dir)
    for nm in names:
        gm = series[nm]["stats"]["global_max"]
        print(f"  - {nm} 浮子峰值: {gm['h']:.3f} m  @ {gm['t']:.2f}s "
              f"(帧 {int(gm['t'] * fps)})")
    return series


def _write_csv(series, names, path):
    n = len(series[names[0]]["t"])
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        head = ["frame", "time_s"]
        for nm in names:
            head += [f"{nm}_h_m", f"{nm}_h_raw_m", f"{nm}_conf"]
        w.writerow(head)
        for i in range(n):
            row = [int(series[names[0]]["frame"][i]),
                   round(float(series[names[0]]["t"][i]), 4)]
            for nm in names:
                s = series[nm]
                row += [round(float(s["h"][i]), 4),
                        round(float(s["h_raw"][i]), 4),
                        round(float(s["conf"][i]), 3)]
            w.writerow(row)


def _write_report(series, names, cal, cfg, path):
    lines = ["# 浮子高度峰值识别报告\n",
             f"- 视频: `{cfg['video']}`",
             f"- 罐直径: {cfg['tank']['diameter_m']} m",
             f"- 标定锚点: " + ", ".join(
                 f"{a['name']}(x={a['x']})" for a in cfg["calibration"]["anchors"]),
             ""]
    lines.append("## 各浮子峰值汇总\n")
    lines.append("| 浮子 | 峰值高度(m) | 出现时刻(s) | 帧号 | 均值(m) | 振幅(cm) | 峰个数 | 追踪置信均值 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for nm in names:
        s = series[nm]["stats"]
        gm = s["global_max"]
        fr = int(series[nm]["frame"][gm["idx"]])
        conf = float(np.mean(series[nm]["conf"]))
        lines.append(f"| {nm} | **{gm['h']:.3f}** | {gm['t']:.2f} | {fr} | "
                     f"{s['mean']:.3f} | {s['amplitude']*100:.1f} | "
                     f"{s['n_peaks']} | {conf:.2f} |")
    lines.append("")
    for nm in names:
        s = series[nm]["stats"]
        lines.append(f"## {nm} 浮子 - 主要峰值(前10)\n")
        lines.append("| 时刻(s) | 高度(m) |")
        lines.append("|---|---|")
        top = sorted(s["peaks"], key=lambda p: -p["h"])[:10]
        for p in top:
            lines.append(f"| {p['t']:.2f} | {p['h']:.3f} |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _render(video, series, names, float_cal, cfg, path, fps, W, H, stride):
    cap = _open(video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps / max(1, stride), (W, H))
    vcfg = cfg.get("visual", {})
    box_color = tuple(vcfg.get("box_color", [0, 0, 255]))
    draw_ref = vcfg.get("draw_ref_lines", True)
    ref_heights = vcfg.get("ref_heights", [2.0, 2.5])
    # frame -> sample index 映射
    fmap = {int(series[names[0]]["frame"][i]): i
            for i in range(len(series[names[0]]["frame"]))}
    boxes = {f["name"]: f["roi"] for f in cfg["floats"]}
    idx = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in fmap:
            i = fmap[idx]
            tracks = {}
            for nm in names:
                s = series[nm]
                cx, cy = float(s["cx"][i]), float(s["cy"][i])
                hm = float(s["h"][i])
                bw, bh = boxes[nm][2], boxes[nm][3]
                bx, by = int(cx - bw / 2), int(cy - bh / 2)
                tracks[nm] = (cx, cy, hm, (bx, by, bw, bh))
            viz.draw_overlay(fr, tracks, float_cal, names, ref_heights,
                             draw_ref=draw_ref, box_color=box_color)
            cv2.putText(fr, f"t={idx/fps:5.2f}s  frame={idx}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)
            vw.write(fr)
        idx += 1
    cap.release()
    vw.release()

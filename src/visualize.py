"""可视化：高度时程曲线图 + 标注视频。"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def _setup_cjk_font():
    """尝试加载中文字体, 避免 matplotlib 缺字警告。"""
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for p in candidates:
        try:
            font_manager.fontManager.addfont(p)
            name = font_manager.FontProperties(fname=p).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return True
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return False


_HAS_CJK = _setup_cjk_font()
COLORS = {"left": "tab:blue", "right": "tab:red"}


def plot_curves(series, out_path):
    """series: {name: dict(t, h_raw, h, stats)} -> 保存多子图曲线。"""
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (name, d) in zip(axes, series.items()):
        t, h, hr, st = d["t"], d["h"], d["h_raw"], d["stats"]
        col = COLORS.get(name, "tab:green")
        ax.plot(t, hr, color=col, alpha=0.25, lw=0.6, label="原始")
        ax.plot(t, h, color=col, lw=1.4, label="平滑")
        # 所有峰
        for pk in st["peaks"]:
            ax.plot(pk["t"], pk["h"], ".", color=col, ms=5)
        # 全局峰值
        gm = st["global_max"]
        ax.plot(gm["t"], gm["h"], "k^", ms=11)
        ax.annotate(f"峰值 {gm['h']:.3f} m @ {gm['t']:.2f}s",
                    (gm["t"], gm["h"]), textcoords="offset points",
                    xytext=(8, 6), fontsize=9, fontweight="bold")
        ax.set_title(f"{name} 浮子高度时程    "
                     f"峰值 {st['max']:.3f}m / 均值 {st['mean']:.3f}m / "
                     f"振幅 {st['amplitude'] * 100:.1f}cm / 峰数 {st['n_peaks']}")
        ax.set_ylabel("高度 (m)")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("时间 (s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def draw_overlay(frame, tracks, float_cal, names, ref_heights=(2.0, 2.5),
                 draw_ref=True, box_color=(0, 0, 255)):
    """在一帧上画浮子框、当前高度、刻度参考线。

    tracks: {name:(cx,cy,h_m,box)}
    float_cal: {name:{rest_y, ppm, rest_h, x}}  浮子自身锚定标定
    参考线按浮子自身基准绘制(静止位置=rest_h), 反映斜拍下浮子的真实尺度。
    """
    if draw_ref:
        for nm in names:
            fc = float_cal.get(nm)
            f = tracks.get(nm)
            if not fc or not f:
                continue
            cx = int(f[0])
            for hh in ref_heights:
                # 浮子坐标系: y = rest_y - (hh - rest_h)*ppm
                yy = int(fc["rest_y"] - (hh - fc["rest_h"]) * fc["ppm"])
                cv2.line(frame, (cx - 75, yy), (cx + 75, yy), (0, 200, 255), 1)
                cv2.putText(frame, f"{hh:.1f}m", (cx + 78, yy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    for nm in names:
        f = tracks.get(nm)
        if not f:
            continue
        cx, cy, hm, (bx, by, bw, bh) = f
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0, 255, 0), -1)
        cv2.putText(frame, f"{nm}: {hm:.3f} m", (bx, by - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
    return frame

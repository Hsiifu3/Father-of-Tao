"""时间序列平滑与峰值检测。"""
import numpy as np
from scipy.signal import find_peaks, medfilt


def smooth(y, fps, median_seconds=0.05, smooth_seconds=0.15):
    """中值去离群 + 反射边界的移动平均低通。"""
    y = np.asarray(y, float)
    # 1) 中值滤波去单点离群(飞溅/误匹配)
    k = max(3, int(round(fps * median_seconds)))
    if k % 2 == 0:
        k += 1
    if k < len(y):
        y = medfilt(y, k)
    # 2) 移动平均, 反射边界避免端点伪影
    win = max(1, int(round(fps * smooth_seconds)))
    if win > 1:
        pad = win // 2
        yp = np.pad(y, pad, mode="reflect")
        ker = np.ones(win) / win
        y = np.convolve(yp, ker, "valid")[: len(np.asarray(y))]
    return y


def detect_peaks(t, h, prominence=0.015, min_distance_s=0.3, fps=60.0):
    """返回峰值列表与全局统计。h 为米制高度时序。"""
    h = np.asarray(h, float)
    dist = max(1, int(round(fps * min_distance_s)))
    idx, props = find_peaks(h, prominence=prominence, distance=dist)
    gmax = int(np.argmax(h))
    peaks = [{"t": float(t[i]), "h": float(h[i]), "idx": int(i)} for i in idx]
    return {
        "peaks": peaks,
        "global_max": {"t": float(t[gmax]), "h": float(h[gmax]), "idx": gmax},
        "min": float(np.min(h)),
        "max": float(np.max(h)),
        "mean": float(np.mean(h)),
        "amplitude": float(np.max(h) - np.min(h)),
        "n_peaks": len(peaks),
    }

"""像素 <-> 物理高度标定。

每个锚点(量测杆)处有两条已知高度的红色刻度线，对其做线性拟合
y = slope*h + intercept。不同 x 处的 slope/intercept 由各锚点
沿 x 线性插值，以处理斜俯视带来的透视尺度变化。

由像素 y 反算高度：h = (y - intercept) / slope。
"""
import numpy as np


class Calibration:
    def __init__(self, cfg):
        anchors = []
        for a in cfg["anchors"]:
            hs = np.array([m["height_m"] for m in a["marks"]], float)
            ys = np.array([m["y"] for m in a["marks"]], float)
            slope, inter = np.polyfit(hs, ys, 1)   # y = slope*h + inter
            anchors.append((float(a["x"]), float(slope), float(inter)))
        anchors.sort(key=lambda t: t[0])
        self.xs = np.array([a[0] for a in anchors])
        self.slopes = np.array([a[1] for a in anchors])
        self.inters = np.array([a[2] for a in anchors])

    def height_m(self, x, y):
        """像素列 x、像素行 y -> 物理高度(米)。"""
        slope = np.interp(x, self.xs, self.slopes)
        inter = np.interp(x, self.xs, self.inters)
        return float((y - inter) / slope)

    def px_per_m(self, x):
        """该列处的像素/米(透视尺度)。"""
        return float(abs(np.interp(x, self.xs, self.slopes)))

    def y_at_height(self, x, h):
        """给定列 x 与高度 h(米)，返回像素行 y(用于画参考线)。"""
        slope = np.interp(x, self.xs, self.slopes)
        inter = np.interp(x, self.xs, self.inters)
        return float(slope * h + inter)

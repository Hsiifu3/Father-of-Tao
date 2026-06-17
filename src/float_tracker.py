"""浮子追踪：模板匹配 + 亚像素 + 竖直/水平搜索 + 低置信度保持。

浮子主要沿导向杆竖直运动(伴小幅水平)，故在以上一帧位置为中心的
搜索带内做归一化互相关匹配，定位浮子，并对相关峰做抛物线亚像素细化。
匹配置信度低于阈值时保持上一帧位置，避免飞溅/反光导致跳变。
"""
import cv2
import numpy as np


class FloatTracker:
    def __init__(self, cfg, first_gray):
        tk = cfg.get("tracker", {})
        self.search_v = int(tk.get("search_v", 70))
        self.search_h = int(tk.get("search_h", 18))
        self.min_conf = float(tk.get("min_conf", 0.35))
        self.floats = {}
        for fl in cfg["floats"]:
            x, y, w, h = fl["roi"]
            self.floats[fl["name"]] = {
                "w": w, "h": h,
                "tpl": first_gray[y:y + h, x:x + w].copy(),
                "px": float(x), "py": float(y),     # 当前左上角
                "x0": float(x),                       # 初始 x(用于标定列)
            }

    @staticmethod
    def _subpix_y(res, loc):
        """对相关峰在竖直方向做抛物线插值，得到亚像素行偏移。"""
        x0, y0 = loc
        if 0 < y0 < res.shape[0] - 1:
            a, b, c = res[y0 - 1, x0], res[y0, x0], res[y0 + 1, x0]
            d = a - 2 * b + c
            if abs(d) > 1e-6:
                return y0 + 0.5 * (a - c) / d
        return float(y0)

    def update(self, gray):
        """对一帧灰度图更新所有浮子位置。返回 {name: (cx, cy, conf)}。"""
        H, W = gray.shape
        out = {}
        for name, f in self.floats.items():
            w, h = f["w"], f["h"]
            ys = max(0, int(f["py"]) - self.search_v)
            ye = min(H, int(f["py"]) + h + self.search_v)
            xs = max(0, int(f["px"]) - self.search_h)
            xe = min(W, int(f["px"]) + w + self.search_h)
            band = gray[ys:ye, xs:xe]
            if band.shape[0] < h or band.shape[1] < w:
                out[name] = (f["px"] + w / 2, f["py"] + h / 2, 0.0)
                continue
            res = cv2.matchTemplate(band, f["tpl"], cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx >= self.min_conf:
                f["py"] = ys + self._subpix_y(res, loc)
                f["px"] = xs + loc[0]
            # 低置信度: 保持上一帧 px/py
            out[name] = (f["px"] + w / 2, f["py"] + h / 2, float(mx))
        return out

    def col_x(self, name):
        """该浮子用于高度标定的列坐标(取中心)。"""
        f = self.floats[name]
        return f["x0"] + f["w"] / 2

"""交互式工具：在某帧上框选浮子 ROI / 标定刻度线像素坐标。

用法:
  python tools/pick_roi.py                 # 默认用 config.yaml 的视频, 首帧
  python tools/pick_roi.py --frame 1200

操作:
  - 鼠标拖拽框选浮子矩形, 回车/空格确认, 可连续框选多个
  - 选完按 ESC 退出, 终端打印可直接粘贴进 config.yaml 的 roi
  - 左键单击: 打印该点像素坐标(用于标定红线/锚点)
"""
import argparse
import sys

import cv2
import yaml

clicks = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"点击坐标: x={x}, y={y}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--video", default=None)
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    video = args.video or cfg["video"]

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("读取帧失败"); sys.exit(1)

    print("== 框选浮子: 拖拽矩形, 回车确认, ESC 结束 ==")
    rois = cv2.selectROIs("pick (Enter=确认, ESC=结束)", frame,
                          showCrosshair=True)
    for i, (x, y, w, h) in enumerate(rois):
        print(f"float #{i}: roi: [{x}, {y}, {w}, {h}]")

    print("\n== 单击标定点(红线): 左键打印坐标, 任意键退出 ==")
    cv2.setMouseCallback("pick (Enter=确认, ESC=结束)", on_mouse)
    cv2.imshow("pick (Enter=确认, ESC=结束)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

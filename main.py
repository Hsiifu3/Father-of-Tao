"""命令行入口。

用法:
  python main.py                      # 用 config.yaml 跑全部产出
  python main.py --no-video           # 跳过标注视频(更快)
  python main.py --video other.mp4    # 指定视频
  python main.py --config my.yaml
"""
import argparse

import yaml

from src.pipeline import run


def main():
    ap = argparse.ArgumentParser(description="储液罐浮子高度时程识别")
    ap.add_argument("--config", default="config.yaml", help="配置文件")
    ap.add_argument("--video", default=None, help="覆盖配置里的视频路径")
    ap.add_argument("--no-video", action="store_true", help="不渲染标注视频")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.video:
        cfg["video"] = args.video

    run(cfg, render_video=not args.no_video)


if __name__ == "__main__":
    main()

"""Capture one aligned RGB-D frame + intrinsics from an Intel RealSense (needs `pip install pyrealsense2`).

Writes <out>/color.png, <out>/depth.png (16-bit, mm) and <out>/intrinsics.json, ready for `snappose match`.
NOTE: written against the pyrealsense2 API but not tested on hardware yet.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures/shot")
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=30, help="frames to discard (auto-exposure)")
    a = ap.parse_args()

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.depth, a.width, a.height, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, a.width, a.height, rs.format.bgr8, 30)
    prof = pipe.start(cfg)
    try:
        depth_scale_mm = prof.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        align = rs.align(rs.stream.color)
        for _ in range(a.warmup):
            pipe.wait_for_frames()
        frames = align.process(pipe.wait_for_frames())
        d, c = frames.get_depth_frame(), frames.get_color_frame()
        intr = c.profile.as_video_stream_profile().intrinsics
        depth_mm = np.asanyarray(d.get_data()).astype(np.float32) * depth_scale_mm
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "color.png"), np.asanyarray(c.get_data()))
        cv2.imwrite(str(out / "depth.png"), np.clip(depth_mm, 0, 65535).astype(np.uint16))
        (out / "intrinsics.json").write_text(json.dumps({"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}))
        print(f"saved to {out}")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()

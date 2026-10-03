from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .api import PoseMatcher
from .config import PROFILES
from .geometry import pose_error
from .synth import DEFAULT_K


def _parse_set(items: list[str]) -> dict:
    out: dict = {}
    for it in items or []:
        key, _, val = it.partition("=")
        try:
            import yaml
            v = yaml.safe_load(val)
        except Exception:
            v = val
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def _load_K(path: str) -> np.ndarray:
    d = json.loads(Path(path).read_text())
    if "K" in d:
        return np.array(d["K"], float).reshape(3, 3)
    return np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], float)


def _load_pose(path: str) -> np.ndarray:
    d = json.loads(Path(path).read_text())
    if "T_cam_obj" in d:
        return np.array(d["T_cam_obj"], float)
    T = np.eye(4)
    T[:3, 3] = d["translation_mm"]
    if "quaternion_xyzw" in d:
        T[:3, :3] = Rotation.from_quat(d["quaternion_xyzw"]).as_matrix()
    else:
        T[:3, :3] = Rotation.from_rotvec(d["rotvec"]).as_matrix()
    return T


def _matcher(args) -> PoseMatcher:
    m = PoseMatcher(config=args.config, store=args.store, profile=args.profile)
    ov = _parse_set(getattr(args, "set", []))
    if ov or getattr(args, "budget", None) is not None:
        m.cfg = m.cfg.with_overrides(ov, time_budget_ms=getattr(args, "budget", None))
    return m


def cmd_onboard(args) -> int:
    m = PoseMatcher(store=args.store)
    model = m.onboard(args.object, args.cad, scale=args.scale, symmetry_path=args.symmetry, force=args.force)
    print(f"onboarded {args.object}: {len(model.points)} points, extents {model.extents.round(1)} mm, "
          f"watertight={model.watertight}, symmetry={'none' if model.symmetry.is_trivial else 'yes'}")
    return 0


def cmd_match(args) -> int:
    m = _matcher(args)
    depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    if depth is None or depth.ndim != 2:
        print("depth must be a single-channel (16-bit PNG, mm) image", file=sys.stderr)
        return 2
    depth = depth.astype(np.float32) * args.depth_scale
    res = m.match(args.object, depth, _load_K(args.intrinsics), _load_pose(args.prior))
    out = res.to_dict()
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    if args.overlay and args.rgb:
        from .viz import overlay
        rgb = cv2.cvtColor(cv2.imread(args.rgb), cv2.COLOR_BGR2RGB)
        img = overlay(rgb, m.model(args.object), res.T_cam_obj, _load_K(args.intrinsics))
        cv2.imwrite(args.overlay, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return 0 if res.status == "OK" else 1


def cmd_demo(args) -> int:
    from .onboarding import build_model
    from .synth import make_bracket, perturb, random_pose, render_rgbd
    from .viz import overlay
    rng = np.random.default_rng(args.seed)
    mesh = make_bracket()
    model = build_model("bracket", mesh)
    gt = random_pose(rng)
    rgb, depth = render_rgbd(mesh, gt, background_z=gt[2, 3] + 60, seed=args.seed)
    prior = perturb(gt, rng, args.prior_t, args.prior_r)
    m = _matcher(args)
    m.set_model(model)
    res = m.match("bracket", depth, DEFAULT_K, prior)
    et, er = pose_error(res.T_cam_obj, gt)
    print(json.dumps(res.to_dict()["timing_ms"]))
    print(f"status={res.status} confidence={res.confidence:.2f} "
          f"prior error: {pose_error(prior, gt)[0]:.1f} mm / {pose_error(prior, gt)[1]:.1f} deg  ->  "
          f"result error: {et:.2f} mm / {er:.2f} deg  ({res.timing_ms['total']:.0f} ms, profile {res.profile})")
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        both = np.hstack([overlay(rgb, model, prior, DEFAULT_K, (255, 80, 0)), overlay(rgb, model, res.T_cam_obj, DEFAULT_K)])
        cv2.imwrite(str(out / "demo_overlay.png"), cv2.cvtColor(both, cv2.COLOR_RGB2BGR))
        print(f"overlay (left: prior, right: result) -> {out / 'demo_overlay.png'}")
    return 0


def cmd_bench(args) -> int:
    from . import bench
    res = bench.run(n=args.n, profiles=args.profiles, seed=args.seed)
    print(bench.format_table(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


def cmd_bop(args) -> int:
    from . import eval_bop
    res = eval_bop.run(args.dataset, profile=args.profile or "balanced", per_object=args.per_object,
                       prior_t=args.prior_t, prior_r=args.prior_r, obj_ids=args.obj, seed=args.seed,
                       overrides=_parse_set(args.set), min_visib=args.min_visib, split=args.split, use_image=not args.no_image)
    print(eval_bop.format_report(res))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="snappose", description="Single-shot 6D pose matching from CAD + RGB-D")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-c", "--config", help="YAML config")
        p.add_argument("-p", "--profile", choices=sorted(PROFILES), help="fast | balanced | precise")
        p.add_argument("--budget", type=float, help="time budget in ms (overrides profile)")
        p.add_argument("-s", "--set", action="append", default=[], metavar="sec.key=value",
                       help="override a single parameter, e.g. s3.refine_iters=3 (repeatable)")
        p.add_argument("--store", default="objects", help="directory for onboarded objects")

    p = sub.add_parser("onboard", help="learn a new object from CAD (STL/PLY/OBJ)")
    p.add_argument("--object", required=True)
    p.add_argument("--cad", required=True)
    p.add_argument("--scale", type=float, default=1.0, help="CAD unit -> mm (1000 for metres)")
    p.add_argument("--symmetry", help="symmetries.json")
    p.add_argument("--store", default="objects")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_onboard)

    p = sub.add_parser("match", help="match one RGB-D frame against an onboarded object (prior mode)")
    common(p)
    p.add_argument("--object", required=True)
    p.add_argument("--depth", required=True, help="16-bit PNG registered to RGB")
    p.add_argument("--depth-scale", type=float, default=1.0, help="PNG units -> mm")
    p.add_argument("--rgb", help="colour image (only for --overlay)")
    p.add_argument("--intrinsics", required=True, help='JSON {"fx","fy","cx","cy"} or {"K": [...]}')
    p.add_argument("--prior", required=True, help='JSON {"T_cam_obj": 4x4} or {"translation_mm", "quaternion_xyzw"|"rotvec"}')
    p.add_argument("--out", help="write result JSON")
    p.add_argument("--overlay", help="write overlay PNG (needs --rgb)")
    p.set_defaults(fn=cmd_match)

    p = sub.add_parser("demo", help="synthetic end-to-end run, no camera or CAD needed")
    common(p)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--prior-t", type=float, default=6.0, help="prior translation error per axis, mm")
    p.add_argument("--prior-r", type=float, default=3.0, help="prior rotation error per axis, deg")
    p.add_argument("--out", default="out", help="directory for the overlay image")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("bench", help="synthetic benchmark over profiles (success rate, error, time)")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profiles", nargs="+", choices=sorted(PROFILES))
    p.add_argument("--json", help="write full results as JSON")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("bop", help="evaluate on a BOP-format dataset (LM, TUD-L, ITODD, ...) with GT-perturbed priors")
    common(p)
    p.add_argument("dataset", help="directory with models/ and test/")
    p.add_argument("--per-object", type=int, default=30)
    p.add_argument("--obj", type=int, nargs="+", help="restrict to object ids")
    p.add_argument("--prior-t", type=float, default=6.0)
    p.add_argument("--prior-r", type=float, default=3.0)
    p.add_argument("--split", default="test", help="test | val (ITODD ground truth is only public for val)")
    p.add_argument("--no-image", action="store_true", help="depth only (ignore rgb/gray images)")
    p.add_argument("--min-visib", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    p.set_defaults(fn=cmd_bop)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

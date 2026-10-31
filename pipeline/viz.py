"""Visual checks for every pipeline step. Each function saves one PNG and returns its path.

Colour code throughout: prediction = red (left side darker), ground truth = green,
wand = blue / orange.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .skeleton import SKELETON_EDGES  # noqa: E402

PRED, GT, WAND, WAND_GT = "#d62728", "#2ca02c", "#1f77b4", "#ff7f0e"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def _skeleton_2d(ax, kp, color, lw=1.2, s=8, label=None):
    for a, b in SKELETON_EDGES:
        if np.isfinite(kp[[a, b]]).all():
            ax.plot(kp[[a, b], 0], kp[[a, b], 1], color=color, lw=lw, alpha=0.8)
    ax.scatter(kp[:, 0], kp[:, 1], s=s, c=color, label=label, zorder=3)


def _skeleton_3d(ax, P, color, label=None, s=10):
    for a, b in SKELETON_EDGES:
        if np.isfinite(P[[a, b]]).all():
            ax.plot(P[[a, b], 0], P[[a, b], 1], P[[a, b], 2], color=color, lw=1.2, alpha=0.8)
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=s, c=color, label=label, depthshade=False)


def _equal_3d(ax, pts):
    pts = pts[np.isfinite(pts).all(axis=1)]
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def _cam_view(P):
    """Camera coords (x right, y down, z forward) -> plot axes (x, z, -y) so up is up."""
    return P[..., [0, 2, 1]] * np.array([1, 1, -1])


def _wcs_view(P):
    """WCS (y up) -> plot axes (x, z, y)."""
    return P[..., [0, 2, 1]]


def plot_inputs(images, bboxes, adj_boxes, cams, path):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 6))
    for ax, img, b, a, cam in zip(axes, images, bboxes, adj_boxes, cams):
        ax.imshow(img)
        ax.add_patch(plt.Rectangle(b[:2], b[2], b[3], fill=False, ec="yellow", lw=1.5, label="JSON bbox"))
        ax.add_patch(plt.Rectangle(a[:2], a[2], a[3], fill=False, ec="cyan", lw=1.5, ls="--", label="model crop (3:4)"))
        ax.set_title(f"{cam}  {img.shape[1]}x{img.shape[0]}")
        ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Step 1 - input images and person crops")
    return _save(fig, path)


def plot_crops_heatmaps(crops, heatmaps, kp_crop, cams, path):
    """crops (3, H, W, 3) uint8; heatmaps (3, J, h, w); kp_crop (3, J, 2) in crop pixels."""
    fig, axes = plt.subplots(2, len(crops), figsize=(3.6 * len(crops), 9))
    for c, cam in enumerate(cams):
        H, W = crops[c].shape[:2]
        axes[0, c].imshow(crops[c])
        _skeleton_2d(axes[0, c], kp_crop[c], PRED, s=6)
        axes[0, c].set_title(f"{cam}: input {W}x{H} + keypoints", fontsize=10)
        axes[1, c].imshow(crops[c])
        axes[1, c].imshow(heatmaps[c].max(axis=0), cmap="jet", alpha=0.55, extent=(0, W, H, 0))
        axes[1, c].set_title(f"{cam}: max of 59 heatmaps ({heatmaps.shape[-1]}x{heatmaps.shape[-2]})", fontsize=10)
        for ax in axes[:, c]:
            ax.axis("off")
    fig.suptitle("Step 2 - network input, heatmaps and decoded keypoints")
    return _save(fig, path)


def plot_pose_2d(images, pred, gt, conf, cams, path, adj_boxes=None):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 6.5))
    for c, (ax, img, cam) in enumerate(zip(axes, images, cams)):
        ax.imshow(img)
        if gt is not None and np.isfinite(gt[c]).any():
            _skeleton_2d(ax, gt[c], GT, label="Vicon GT (projected)")
        _skeleton_2d(ax, pred[c], PRED, label="prediction")
        err = np.nanmean(np.linalg.norm(pred[c] - gt[c], axis=-1)) if gt is not None else np.nan
        ax.set_title(f"{cam}: mean 2D error {err:.1f} px, conf {conf[c].mean():.2f}")
        if adj_boxes is not None:
            x, y, w, h = adj_boxes[c]
            ax.set_xlim(x - 0.1 * w, x + 1.1 * w)
            ax.set_ylim(y + 1.05 * h, y - 0.05 * h)
        ax.axis("off")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Step 3 - 2D keypoints in full-image coordinates")
    return _save(fig, path)


def plot_triangulation(images, pred_3d, gt_3d, reproj, obs, cams, path, units="mm", reproj_err=None):
    """3D in main-camera coordinates (left) and reprojection of the 3D points into each view."""
    fig = plt.figure(figsize=(5 + 4 * len(images), 6.5))
    ax = fig.add_subplot(1, len(images) + 1, 1, projection="3d")
    pts = [_cam_view(pred_3d)]
    if gt_3d is not None and np.isfinite(gt_3d).any():
        _skeleton_3d(ax, _cam_view(gt_3d), GT, "Vicon GT")
        pts.append(_cam_view(gt_3d))
    _skeleton_3d(ax, _cam_view(pred_3d), PRED, "triangulated prediction")
    _equal_3d(ax, np.vstack(pts))
    ax.locator_params(nbins=4)
    ax.set_xlabel(f"x ({units})"); ax.set_ylabel(f"z depth ({units})"); ax.set_zlabel(f"-y ({units})")
    ax.view_init(elev=10, azim=-80)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("main-camera coordinates")
    for c, (img, cam) in enumerate(zip(images, cams)):
        a = fig.add_subplot(1, len(images) + 1, c + 2)
        a.imshow(img)
        a.scatter(obs[c][:, 0], obs[c][:, 1], s=14, facecolors="none", edgecolors=PRED, label="2D prediction")
        _skeleton_2d(a, reproj[c], "#9467bd", s=4, label="3D reprojected")
        valid = np.isfinite(obs[c]).all(1)
        x0, y0 = obs[c][valid].min(0); x1, y1 = obs[c][valid].max(0); pad = 0.15 * (y1 - y0)
        a.set_xlim(x0 - pad, x1 + pad); a.set_ylim(y1 + pad, y0 - pad)
        title = cam if reproj_err is None else f"{cam}: reprojection {reproj_err[c]:.1f} px"
        a.set_title(title); a.axis("off")
        if c == 0:
            a.legend(fontsize=8, loc="lower left")
    fig.suptitle("Step 4 - DLT triangulation (main camera frame) and reprojection check")
    return _save(fig, path)


def plot_wand(images, wand_xy, wand_reproj, wand_cam, cams, path, wand_cam_json=None, units="mm"):
    fig = plt.figure(figsize=(5 + 4 * len(images), 5))
    for c, (img, cam) in enumerate(zip(images, cams)):
        a = fig.add_subplot(1, len(images) + 1, c + 1)
        a.imshow(img)
        a.scatter(wand_xy[c][:, 0], wand_xy[c][:, 1], s=40, facecolors="none", edgecolors=WAND, lw=1.5, label="wand 2D (JSON)")
        a.scatter(wand_reproj[c][:, 0], wand_reproj[c][:, 1], s=10, c=WAND_GT, label="3D reprojected")
        for k, (u, v) in enumerate(wand_xy[c]):
            a.annotate(str(k), (u, v), xytext=(4, -4), textcoords="offset points", color="white", fontsize=8)
        x0, y0 = wand_xy[c].min(0); x1, y1 = wand_xy[c].max(0); pad = 60
        a.set_xlim(x0 - pad, x1 + pad); a.set_ylim(y1 + pad, y0 - pad)
        a.set_title(f"{cam} wand (zoom)"); a.axis("off")
        if c == 0:
            a.legend(fontsize=8, loc="upper left")
    ax = fig.add_subplot(1, len(images) + 1, len(images) + 1, projection="3d")
    W = _cam_view(wand_cam)
    ax.plot(*W[[0, 1, 2]].T, color=WAND); ax.plot(*W[[1, 3, 4]].T, color=WAND)
    ax.scatter(*W.T, c=WAND, s=25, label="triangulated")
    if wand_cam_json is not None:
        ax.scatter(*_cam_view(wand_cam_json).T, c=WAND_GT, marker="x", s=40, label="JSON wand_xyz")
    for k, p in enumerate(W):
        ax.text(*p, str(k), fontsize=8)
    _equal_3d(ax, W); ax.legend(fontsize=8)
    ax.set_title(f"wand in main-camera frame ({units})")
    fig.suptitle("Step 5 - wand triangulation")
    return _save(fig, path)


def plot_world(pred_wcs, gt_wcs, wand_wcs_pred, wand_wcs_ref, path, mpjpe=None):
    """3D perspective plus front / side / top orthographic projections of the world frame."""
    fig = plt.figure(figsize=(18, 5.5))
    has_gt = gt_wcs is not None and np.isfinite(gt_wcs).any()
    allp = np.vstack([pred_wcs, wand_wcs_ref] + ([gt_wcs] if has_gt else []))
    allp = allp[np.isfinite(allp).all(1)]

    ax = fig.add_subplot(1, 4, 1, projection="3d")
    lo, hi = allp.min(0), allp.max(0)
    gx, gz = np.meshgrid(np.linspace(lo[0], hi[0], 6), np.linspace(lo[2], hi[2], 6))
    ax.plot_wireframe(gx, gz, np.zeros_like(gx), color="0.8", lw=0.5)        # floor, y = 0
    for axis, col in zip(np.eye(3) * 300, ("r", "g", "b")):                  # WCS axes at the origin
        a = _wcs_view(axis)
        ax.plot([0, a[0]], [0, a[1]], [0, a[2]], color=col, lw=2)
    if has_gt:
        _skeleton_3d(ax, _wcs_view(gt_wcs), GT, "Vicon GT", s=6)
    _skeleton_3d(ax, _wcs_view(pred_wcs), PRED, "prediction", s=6)
    ax.scatter(*_wcs_view(wand_wcs_ref).T, c=WAND_GT, marker="x", s=30, label="wand (WCS file)")
    ax.scatter(*_wcs_view(wand_wcs_pred).T, c=WAND, s=12, label="wand (triangulated)")
    _equal_3d(ax, _wcs_view(allp))
    ax.locator_params(nbins=4)
    ax.view_init(elev=15, azim=-60)
    ax.set_xlabel("x"); ax.set_ylabel("z"); ax.set_zlabel("y (up)"); ax.set_title("perspective")
    ax.legend(fontsize=7, loc="upper left")

    for i, (h, v, name) in enumerate([(0, 1, "front: x / y"), (2, 1, "side: z / y"), (0, 2, "top: x / z")]):
        a2 = fig.add_subplot(1, 4, i + 2)
        a2.axhline(0, color="0.8", lw=0.8); a2.axvline(0, color="0.8", lw=0.8)
        if has_gt:
            _skeleton_2d(a2, gt_wcs[:, [h, v]], GT, s=5)
        _skeleton_2d(a2, pred_wcs[:, [h, v]], PRED, s=5)
        a2.scatter(wand_wcs_ref[:, h], wand_wcs_ref[:, v], c=WAND_GT, marker="x", s=30)
        a2.scatter(wand_wcs_pred[:, h], wand_wcs_pred[:, v], c=WAND, s=10)
        a2.set_aspect("equal"); a2.set_title(name)
        a2.set_xlabel("xyz"[h] + " (mm)"); a2.set_ylabel("xyz"[v] + " (mm)")
    title = "Step 6 - prediction in the Vicon world frame (mm); axes: x red, y green (up), z blue"
    if mpjpe is not None:
        title += f"   |   MPJPE vs GT: {mpjpe:.1f} mm"
    fig.suptitle(title)
    return _save(fig, path)


STAGE_COLOURS = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]


def plot_timing(rows, stages, path, title="Time per step"):
    """Mean time per per-frame step (bars with p95 whiskers, coloured by stage) and the stage shares."""
    frame_rows = [r for r in rows if r["stage"]]
    stage_names = list(stages)
    colour = {s: STAGE_COLOURS[i % len(STAGE_COLOURS)] for i, s in enumerate(stage_names)}
    total = sum(r["mean_ms"] for r in frame_rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 0.55 * len(frame_rows) + 2.5),
                                 gridspec_kw={"width_ratios": [2.2, 1]})
    names = [r["step"].replace("frame: ", "") for r in frame_rows]
    means = np.array([r["mean_ms"] for r in frame_rows])
    p95 = np.array([r["p95_ms"] for r in frame_rows])
    a1.barh(names, means, xerr=[np.zeros_like(means), np.clip(p95 - means, 0, None)],
            color=[colour[r["stage"]] for r in frame_rows], capsize=3)
    a1.set_xscale("symlog", linthresh=1)
    for i, r in enumerate(frame_rows):
        a1.text(max(r["p95_ms"], r["mean_ms"]) * 1.15 + 0.05, i,
                f"{r['mean_ms']:.2f} ms  ({r['pct_of_frame']:.1f}%)", va="center", fontsize=9)
    a1.invert_yaxis()
    a1.set_xlabel("mean ms per frame set (log scale), whisker = p95")
    a1.set_xlim(0, means.max() * 8)
    a1.set_title(f"per step - total {total:.1f} ms per frame set ({1000 / total:.1f} / s)")

    stage_ms = [sum(r["mean_ms"] for r in frame_rows if r["stage"] == s) for s in stage_names]
    wedges, _ = a2.pie(stage_ms, colors=[colour[s] for s in stage_names], startangle=90, counterclock=False,
                       wedgeprops={"width": 0.38, "edgecolor": "white"})
    a2.text(0, 0, f"{total:.1f} ms\nper frame set", ha="center", va="center", fontsize=10)
    a2.legend(wedges, [f"{s}: {ms:.1f} ms ({100 * ms / total:.1f}%)" for s, ms in zip(stage_names, stage_ms)],
              loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=9, frameon=False)
    a2.set_title("share per stage")
    fig.suptitle(title)
    return _save(fig, path)


def plot_summary(per_frame, timing_rows, path):
    """per_frame: list of dicts with 'index', 'err2d', 'mpjpe_cam', 'mpjpe_wcs', 'reproj'."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    idx = [r["index"] for r in per_frame]
    axes[0].plot(idx, [np.mean(r["err2d"]) for r in per_frame], label="2D error (px, mean of 3 views)")
    axes[0].plot(idx, [np.mean(r["reproj"]) for r in per_frame], label="reprojection error (px)")
    axes[0].set_xlabel("frame index"); axes[0].legend(); axes[0].set_title("2D")
    axes[1].plot(idx, [r["mpjpe_cam"] for r in per_frame], label="camera frame")
    axes[1].plot(idx, [r["mpjpe_wcs"] for r in per_frame], label="world frame")
    axes[1].set_xlabel("frame index"); axes[1].set_ylabel("MPJPE (mm)"); axes[1].legend(); axes[1].set_title("3D vs Vicon GT")
    steps = [r for r in timing_rows if r["step"].startswith("frame:")]
    axes[2].barh([r["step"][7:] for r in steps], [r["mean_ms"] for r in steps], color="#4c72b0")
    for i, r in enumerate(steps):
        axes[2].text(r["mean_ms"], i, f" {r['mean_ms']:.1f}", va="center", fontsize=8)
    axes[2].invert_yaxis(); axes[2].set_xlabel("mean ms per frame set"); axes[2].set_title("time per step")
    fig.suptitle("Run summary")
    return _save(fig, path)

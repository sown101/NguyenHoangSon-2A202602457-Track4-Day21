"""Topic A: sweep yaw/lateral drift, fixed GT point cohorts, plots and failure.

Code do Codex hỗ trợ viết cho bài cá nhân; dùng loader/perturb của đề bài.
Run from repo root: python -m src.calibration_benchmark --help
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import cv2
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame
from starter.projection import cam_to_image, perturb_extrinsic, velo_to_cam

DATASETS = {"kitti": "data/kitti_mini", "nuscenes": "data/nuscenes_mini_subset"}


def project(points, calib, shape):
    """Keep original point indices: invalid/outside points have NaN pixels."""
    cam = velo_to_cam(points[:, :3], calib)
    pixels, depth, valid = cam_to_image(cam, calib.P2, shape)
    uv = np.full((len(points), 2), np.nan)
    uv[valid] = pixels
    return cam, uv, valid


def points_in_box(cam, obj):
    """KITTI bottom center; inverse rotation of row vectors is @ R."""
    h, w, length = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (cam - obj.location) @ rotation
    return (np.isfinite(local).all(axis=1) & (np.abs(local[:, 0]) <= length / 2)
            & (local[:, 1] >= -h) & (local[:, 1] <= 0)
            & (np.abs(local[:, 2]) <= w / 2))


def inside_box(uv, bbox):
    x1, y1, x2, y2 = bbox
    return (np.isfinite(uv).all(axis=1) & (uv[:, 0] >= x1) & (uv[:, 0] <= x2)
            & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))


def cohorts(cam, valid, labels, min_points):
    result = []
    for index, obj in enumerate(labels):
        if not np.isfinite(obj.dimensions).all() or (obj.dimensions <= 0).any():
            continue
        indices = np.flatnonzero(points_in_box(cam, obj) & valid)
        if len(indices) >= min_points:
            result.append((index, obj, indices))
    return result


def configs():
    return ([('yaw', x, 'deg') for x in [-3., -2., -1., -.5, 0., .5, 1., 2., 3.]]
            + [('lateral', x, 'm') for x in [-.1, -.05, -.02, 0., .02, .05, .1]])


def changed_calib(calib, dataset, family, level):
    if family == 'yaw':
        return perturb_extrinsic(calib, yaw_deg=level)
    # Physical lateral direction: KITTI y-left; nuScenes x-right.
    translation = (0., level, 0.) if dataset == 'kitti' else (level, 0., 0.)
    return perturb_extrinsic(calib, t_xyz_m=translation)


def savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def draw_overlay(ax, fr, cam, uv, valid, title, seed, distance=None):
    ax.imshow(cv2.cvtColor(fr['image'], cv2.COLOR_BGR2RGB))
    selected = np.flatnonzero(valid)
    if distance is not None:
        low, high = distance
        selected = selected[(cam[selected, 2] >= low) & (cam[selected, 2] < high)]
    count = len(selected)
    if count > 6000:
        selected = np.random.default_rng(seed).choice(selected, 6000, replace=False)
    ax.scatter(uv[selected, 0], uv[selected, 1], c=cam[selected, 2], s=2,
               cmap='turbo_r', vmin=0, vmax=60, linewidths=0)
    for obj in fr['labels']:
        x1, y1, x2, y2 = obj.bbox
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec='lime', lw=.7))
    ax.set_title(f'{title} | {count:,} projected points', fontsize=10)
    ax.set_xlim(0, fr['image'].shape[1])
    ax.set_ylim(fr['image'].shape[0], 0)
    ax.axis('off')


def make_demos(out, seed):
    scenes = [('000019', 0, 10, 'Near: 0-10 m'), ('000011', 10, 30, 'Middle: 10-30 m'),
              ('000009', 30, np.inf, 'Far: >=30 m')]
    counts = []
    grid, axes = plt.subplots(3, 1, figsize=(12, 11))
    for ax, (frame, low, high, title) in zip(axes, scenes):
        fr = load_frame(DATASETS['kitti'], frame)
        cam, uv, valid = project(fr['points'], fr['calib'], fr['image'].shape)
        draw_overlay(ax, fr, cam, uv, valid, f'KITTI {frame} - {title}', seed, (low, high))
        fig, single = plt.subplots(figsize=(12, 4))
        draw_overlay(single, fr, cam, uv, valid, f'KITTI {frame} - {title}', seed, (low, high))
        savefig(fig, out / 'figures' / f'demo_{frame}.png')
        counts.append({'frame_id': frame, 'depth_min_m': low,
                       'depth_max_m': high if np.isfinite(high) else '',
                       'n_visible': int((valid & (cam[:, 2] >= low) & (cam[:, 2] < high)).sum())})
    grid.suptitle('Baseline projection at three camera-depth ranges\nSource: KITTI Vision Benchmark Suite', fontsize=13)
    grid.tight_layout()
    savefig(grid, out / 'figures' / 'demo_three_ranges.png')
    pd.DataFrame(counts).to_csv(out / 'demo_ranges.csv', index=False)
    fr = load_frame(DATASETS['nuscenes'], 'scene-0103_010')
    cam, uv, valid = project(fr['points'], fr['calib'], fr['image'].shape)
    fig, ax = plt.subplots(figsize=(12, 7))
    draw_overlay(ax, fr, cam, uv, valid, 'nuScenes scene-0103_010 (ego motion compensated)', seed)
    fig.text(.5, .02, 'Source: nuScenes (Motional)', ha='center')
    savefig(fig, out / 'figures' / 'demo_nuscenes.png')


def make_plots(summary, ranges, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for dataset, group in summary.groupby('dataset', sort=True):
        yaw = group[group.family == 'yaw'].sort_values('level')
        lateral = group[group.family == 'lateral'].sort_values('level')
        axes[0, 0].plot(yaw.level, yaw.box_pct, 'o-', label=dataset)
        axes[0, 1].plot(yaw.level, yaw.fov_pct, 'o-', label=dataset)
        axes[1, 0].plot(lateral.level * 100, lateral.box_pct, 'o-', label=dataset)
        axes[1, 1].plot(yaw.level, yaw.mean_shift_px, 'o-', label=dataset)
    settings = [('Yaw drift (deg)', 'Object point hits in own 2D box (%)'),
                ('Yaw drift (deg)', 'All finite LiDAR points inside camera FOV (%)'),
                ('Lateral translation (cm)', 'Object point hits in own 2D box (%)'),
                ('Yaw drift (deg)', 'Mean pixel shift on common visible points (px)')]
    for ax, (x, y) in zip(axes.flat, settings):
        ax.set_xlabel(x)
        ax.set_ylabel(y, fontsize=9)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Calibration drift: 20 KITTI frames + 80 nuScenes frames\nFixed baseline object cohorts; out-of-FOV object points count as misses')
    fig.tight_layout()
    savefig(fig, out / 'figures' / 'calibration_sweep.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, dataset in zip(axes, ['kitti', 'nuscenes']):
        for band, group in ranges[(ranges.dataset == dataset) & (ranges.family == 'yaw')].groupby('range_band', sort=True):
            group = group.sort_values('level')
            ax.plot(group.level, group.box_pct, 'o-', label=band)
        ax.set_title(dataset)
        ax.set_xlabel('Yaw drift (deg)')
        ax.set_ylabel('Point hits in own 2D box (%)')
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Object cohorts grouped by baseline camera depth (m)')
    fig.tight_layout()
    savefig(fig, out / 'figures' / 'range_sweep.png')


def make_failure(objects, frames, out, seed):
    base = frames[(frames.dataset == 'kitti') & (frames.family == 'yaw') & (frames.level == 0)]
    drift = frames[(frames.dataset == 'kitti') & (frames.family == 'yaw') & (frames.level == 1)]
    delta = drift.merge(base[['frame_id', 'fov_pct']], on='frame_id', suffixes=('', '_base'))
    delta['fov_delta_pp'] = delta.fov_pct - delta.fov_pct_base
    candidates = objects[(objects.dataset == 'kitti') & (objects.family == 'yaw')
                         & (objects.level == 1) & (objects.n_points >= 30)].copy()
    candidates['loss_pp'] = candidates.baseline_box_pct - candidates.box_pct
    candidates = candidates.merge(delta[['frame_id', 'fov_delta_pp']], on='frame_id')
    subtle = candidates[candidates.fov_delta_pp.abs() < .5]
    selected = (subtle if len(subtle) else candidates).sort_values(
        ['loss_pp', 'frame_id', 'object_index'], ascending=[False, True, True]).iloc[0]
    fr = load_frame(DATASETS['kitti'], selected.frame_id)
    cam0, uv0, valid0 = project(fr['points'], fr['calib'], fr['image'].shape)
    obj = fr['labels'][int(selected.object_index)]
    idx = np.flatnonzero(points_in_box(cam0, obj) & valid0)
    _, uv1, valid1 = project(fr['points'], changed_calib(fr['calib'], 'kitti', 'yaw', 1.), fr['image'].shape)
    x1, y1, x2, y2 = obj.bbox
    pad = max(50., (x2-x1) * .6)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, uv, valid, title, color in zip(axes, [uv0, uv1], [valid0, valid1],
        [f'Baseline: {selected.baseline_box_pct:.1f}% in box', f'Yaw +1 deg: {selected.box_pct:.1f}% in box'], ['cyan', 'red']):
        ax.imshow(cv2.cvtColor(fr['image'], cv2.COLOR_BGR2RGB))
        shown = idx[valid[idx]]
        if len(shown) > 1500:
            shown = np.random.default_rng(seed).choice(shown, 1500, replace=False)
        ax.scatter(uv[shown, 0], uv[shown, 1], c=color, s=5, linewidths=0)
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, ec='lime', lw=2))
        ax.set_xlim(max(0, x1-pad), min(fr['image'].shape[1], x2+pad))
        ax.set_ylim(min(fr['image'].shape[0], y2+pad*.4), max(0, y1-pad*.4))
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'KITTI {selected.frame_id} / {selected["class"]} / depth {selected.depth_m:.1f} m / {len(idx)} fixed points')
    fig.text(.5, .03, f'Whole-frame FOV changes only {selected.fov_delta_pp:+.3f} percentage points.\nSource: KITTI Vision Benchmark Suite', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92))
    savefig(fig, out / 'figures' / 'fail_01_yaw_1deg_fov_blind_spot.png')
    info = {key: selected[key].item() if hasattr(selected[key], 'item') else selected[key]
            for key in ['frame_id', 'object_index', 'class', 'depth_m', 'n_points',
                        'baseline_box_pct', 'box_pct', 'loss_pp', 'fov_delta_pp']}
    (out / 'failure_case.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    return info


def main():
    # Windows may default to cp1252, which cannot print Vietnamese CLI help.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', type=Path, default=Path('results'))
    ap.add_argument('--datasets', nargs='+', choices=list(DATASETS), default=list(DATASETS))
    ap.add_argument('--min-object-points', type=int, default=10)
    ap.add_argument('--seed', type=int, default=457, help='Visualization subsampling only; metrics use every point')
    ap.add_argument('--latency-repeats', type=int, default=30)
    args = ap.parse_args()
    if args.min_object_points < 1 or args.latency_repeats < 20:
        ap.error('min-object-points >=1 and latency-repeats >=20 required')
    out = args.out_dir
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    frame_rows, object_rows, timings = [], [], []
    input_frames = {}
    for dataset in args.datasets:
        root = DATASETS[dataset]
        ids = list_frames(root)
        input_frames[dataset] = ids
        for frame_id in ids:
            fr = load_frame(root, frame_id)
            points, shape, calib = fr['points'], fr['image'].shape, fr['calib']
            cam0, uv0, valid0 = project(points, calib, shape)
            finite_count = int(np.isfinite(points[:, :3]).all(axis=1).sum())
            groups = cohorts(cam0, valid0, fr['labels'], args.min_object_points)
            for family, level, unit in configs():
                changed = changed_calib(calib, dataset, family, level)
                _, uv, valid = project(points, changed, shape)
                common = valid & valid0
                shift = np.linalg.norm(uv[common] - uv0[common], axis=1)
                total_hits, total_points = 0, 0
                for index, obj, idx in groups:
                    hits = int(inside_box(uv[idx], obj.bbox).sum())
                    baseline_hits = int(inside_box(uv0[idx], obj.bbox).sum())
                    depth = float(np.median(cam0[idx, 2]))
                    band = 'near [0,10)' if depth < 10 else 'middle [10,30)' if depth < 30 else 'far [30,inf)'
                    object_rows.append(dict(dataset=dataset, frame_id=frame_id, family=family, level=level,
                        unit=unit, object_index=index, **{'class': obj.type}, depth_m=depth, range_band=band,
                        occluded=obj.occluded, truncated=obj.truncated, n_points=len(idx), n_hits=hits,
                        box_pct=100*hits/len(idx), baseline_box_pct=100*baseline_hits/len(idx)))
                    total_hits += hits
                    total_points += len(idx)
                frame_rows.append(dict(dataset=dataset, frame_id=frame_id, family=family, level=level, unit=unit,
                    n_finite=finite_count, n_visible=int(valid.sum()), fov_pct=100*valid.sum()/finite_count,
                    n_objects=len(groups), object_points=total_points, object_hits=total_hits,
                    box_pct=100*total_hits/total_points if total_points else np.nan,
                    common_points=int(common.sum()), mean_shift_px=float(shift.mean()) if len(shift) else np.nan,
                    p95_shift_px=float(np.percentile(shift, 95)) if len(shift) else np.nan))
            if frame_id == ids[0]:
                # Warm-up excluded; measure projection only, not loading/rendering/GT membership.
                project(points, calib, shape)
                for repeat in range(args.latency_repeats):
                    start = time.perf_counter_ns()
                    project(points, calib, shape)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    timings.append(dict(dataset=dataset, frame_id=frame_id, repeat=repeat+1,
                                        n_points=len(points), elapsed_ms=elapsed))
            print(f'{dataset} {frame_id}: {len(points):,} points, {len(groups)} eligible objects', flush=True)
    frames, objects = pd.DataFrame(frame_rows), pd.DataFrame(object_rows)
    frames.to_csv(out / 'calibration_frames.csv', index=False, float_format='%.9f')
    objects.to_csv(out / 'calibration_objects.csv', index=False, float_format='%.9f')
    summary_rows = []
    for keys, group in frames.groupby(['dataset', 'family', 'level', 'unit'], sort=True):
        weighted_shift = float((group.mean_shift_px.fillna(0)*group.common_points).sum()/group.common_points.sum())
        summary_rows.append(dict(zip(['dataset', 'family', 'level', 'unit'], keys), n_frames=len(group),
            n_finite=int(group.n_finite.sum()), n_visible=int(group.n_visible.sum()),
            fov_pct=100*group.n_visible.sum()/group.n_finite.sum(),
            object_points=int(group.object_points.sum()), object_hits=int(group.object_hits.sum()),
            box_pct=100*group.object_hits.sum()/group.object_points.sum() if group.object_points.sum() else np.nan,
            mean_shift_px=weighted_shift))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / 'calibration_summary.csv', index=False, float_format='%.9f')
    ranges = objects.groupby(['dataset', 'family', 'level', 'range_band'], as_index=False).agg(
        n_objects=('object_index', 'size'), n_points=('n_points', 'sum'), n_hits=('n_hits', 'sum'))
    ranges['box_pct'] = 100*ranges.n_hits/ranges.n_points
    ranges.to_csv(out / 'calibration_ranges.csv', index=False, float_format='%.9f')
    timing_df = pd.DataFrame(timings)
    timing_df.to_csv(out / 'projection_latency.csv', index=False, float_format='%.6f')
    timing_summary = timing_df.groupby('dataset', as_index=False).agg(
        repeats=('elapsed_ms', 'size'), p50_ms=('elapsed_ms', 'median'),
        p95_ms=('elapsed_ms', lambda s: s.quantile(.95)))
    timing_summary.to_csv(out / 'projection_latency_summary.csv', index=False, float_format='%.6f')
    make_plots(summary, ranges, out)
    make_demos(out, args.seed)
    failure = make_failure(objects, frames, out, args.seed) if 'kitti' in args.datasets else None
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
        cpu=platform.processor(), numpy=np.__version__, opencv=cv2.__version__,
        matplotlib=matplotlib.__version__, pandas=pd.__version__, gpu_used=False,
        seed=args.seed, min_object_points=args.min_object_points, latency_repeats=args.latency_repeats,
        frames=input_frames, configurations=[dict(family=f, level=v, unit=u) for f,v,u in configs()],
        metric_definition='Fixed baseline GT-3D-box + camera-FOV cohorts; micro average of point-object pairs. '
        'Outside-image points are misses; overlapping boxes can count one point more than once.',
        failure=failure)
    (out / 'experiment_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Done: {len(frames)} frame/config rows, {len(objects)} object/config rows -> {out}', flush=True)


if __name__ == '__main__':
    main()

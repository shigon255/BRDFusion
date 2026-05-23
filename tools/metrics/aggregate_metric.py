import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_OUTPUT_ROOT = Path('outputs')
DEFAULT_WAYMO_STATIC_IDS = [3, 19, 245]
DEFAULT_WAYMO_DYNAMIC_IDS = [114, 172, 703]
DEFAULT_PATH_IDS = [1, 2, 3, 4, 5, 6]
DEFAULT_SCENE_IDX = 'qwantani_moon_noon_puresky_4k'
DEFAULT_RELIGHT_SCENES = [
    'qwantani_moon_noon_puresky_4k_rot90',
    'citrus_orchard_4k',
    'the_sky_is_on_fire_4k',
]
DEFAULT_SEEDS = [1000, 1001, 1002, 1003, 1004]
ABLATION_LABELS = [
    'baseline',
    'nodiffusion',
    'nogenrefine',
    'nomultistage',
    'nopbr',
    'nogenrender',
]


def warn_missing(path: Path) -> None:
    print(f"[missing] {path}", file=sys.stderr)


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        warn_missing(path)
        return None
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def strict_mean(values: List[float]) -> float:
    return math.nan if any(math.isnan(v) for v in values) else (sum(values) / len(values) if values else math.nan)


def nanmean(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return math.nan if not valid else sum(valid) / len(valid)


def nanstd(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return math.nan
    avg = sum(valid) / len(valid)
    return math.sqrt(sum((v - avg) ** 2 for v in valid) / len(valid))


def fmt(value: float) -> str:
    return 'NA' if math.isnan(value) else f'{value:.6f}'


def weighted_named(train: Dict[str, float], test: Dict[str, float], key: str) -> float:
    total = train['count'] + test['count']
    if total == 0:
        return math.nan
    return (train[key] * train['count'] + test[key] * test['count']) / total


def test_image_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {'psnr': math.nan, 'ssim': math.nan, 'lpips': math.nan}
    test = data['metrics']['test']
    return {'psnr': test['psnr'], 'ssim': test['ssim'], 'lpips': test['lpips']}


def test_intrinsic_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {
            'normal_mae': math.nan,
            'albedo_si_psnr': math.nan,
            'roughness_rmse': math.nan,
            'metallic_rmse': math.nan,
        }
    metrics = data['metrics']
    out = {
        'normal_mae': metrics['test_normal']['mae_deg'],
        'albedo_si_psnr': metrics['test_albedo']['si_psnr'],
        'roughness_rmse': metrics['test_roughness']['rmse'],
        'metallic_rmse': math.nan,
    }
    if 'test_metallic' in metrics:
        out['metallic_rmse'] = metrics['test_metallic']['rmse']
    return out


def test_relight_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {'psnr': math.nan, 'ssim': math.nan, 'lpips': math.nan}
    test = data['metrics']['test_relight']
    return {'psnr': test['psnr'], 'ssim': test['ssim'], 'lpips': test['lpips']}


def weighted_image_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {'psnr': math.nan, 'ssim': math.nan, 'lpips': math.nan}
    return {
        'psnr': weighted_named(data['metrics']['train'], data['metrics']['test'], 'psnr'),
        'ssim': weighted_named(data['metrics']['train'], data['metrics']['test'], 'ssim'),
        'lpips': weighted_named(data['metrics']['train'], data['metrics']['test'], 'lpips'),
    }


def weighted_intrinsic_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {
            'normal_mae': math.nan,
            'albedo_si_psnr': math.nan,
            'roughness_rmse': math.nan,
            'metallic_rmse': math.nan,
        }
    metrics = data['metrics']
    out = {
        'normal_mae': weighted_named(metrics['train_normal'], metrics['test_normal'], 'mae_deg'),
        'albedo_si_psnr': weighted_named(metrics['train_albedo'], metrics['test_albedo'], 'si_psnr'),
        'roughness_rmse': weighted_named(metrics['train_roughness'], metrics['test_roughness'], 'rmse'),
        'metallic_rmse': math.nan,
    }
    if 'train_metallic' in metrics and 'test_metallic' in metrics:
        out['metallic_rmse'] = weighted_named(metrics['train_metallic'], metrics['test_metallic'], 'rmse')
    return out


def weighted_relight_metrics(data: Optional[Dict]) -> Dict[str, float]:
    if data is None:
        return {'psnr': math.nan, 'ssim': math.nan, 'lpips': math.nan}
    return {
        'psnr': weighted_named(data['metrics']['train_relight'], data['metrics']['test_relight'], 'psnr'),
        'ssim': weighted_named(data['metrics']['train_relight'], data['metrics']['test_relight'], 'ssim'),
        'lpips': weighted_named(data['metrics']['train_relight'], data['metrics']['test_relight'], 'lpips'),
    }


def prune_all_nan(metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:
    return {k: v for k, v in metrics.items() if any(not math.isnan(x) for x in v)}


def print_table(title: str, rows: List[str], metrics: Dict[str, List[float]]) -> None:
    metrics = prune_all_nan(metrics)
    if not metrics:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n(no available metrics)")
        return
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    headers = list(metrics.keys())
    print('id\t' + '\t'.join(headers))
    for i, row in enumerate(rows):
        vals = [fmt(metrics[h][i]) for h in headers]
        print(f"{row}\t" + '\t'.join(vals))
    avg_vals = [fmt(strict_mean(metrics[h])) for h in headers]
    print('avg\t' + '\t'.join(avg_vals))


def print_seed_table(title: str, rows: List[str], values_by_row: List[List[Dict[str, float]]]) -> None:
    discovered_keys = {key for seed_values in values_by_row for values in seed_values for key in values.keys()}
    preferred_order = ['psnr', 'ssim', 'lpips', 'normal_mae', 'albedo_si_psnr', 'roughness_rmse', 'metallic_rmse']
    metric_keys = [key for key in preferred_order if key in discovered_keys]
    metric_keys.extend(sorted(discovered_keys - set(metric_keys)))
    metric_keys = [
        key for key in metric_keys
        if any(not math.isnan(values.get(key, math.nan)) for seed_values in values_by_row for values in seed_values)
    ]
    if not metric_keys:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n(no available metrics)")
        return
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")
    print('id\t' + '\t'.join([f'{key}_avg\t{key}_std' for key in metric_keys]))
    row_means = {key: [] for key in metric_keys}
    row_stds = {key: [] for key in metric_keys}
    for row, seed_values in zip(rows, values_by_row):
        out = []
        for key in metric_keys:
            values = [entry.get(key, math.nan) for entry in seed_values]
            mean_value = nanmean(values)
            std_value = nanstd(values)
            row_means[key].append(mean_value)
            row_stds[key].append(std_value)
            out.extend([fmt(mean_value), fmt(std_value)])
        print(f"{row}\t" + '\t'.join(out))
    avg_out = []
    for key in metric_keys:
        avg_out.extend([fmt(nanmean(row_means[key])), fmt(nanmean(row_stds[key]))])
    print('avg\t' + '\t'.join(avg_out))


def append_metric_row(container: Dict[str, List[float]], values: Dict[str, float]) -> None:
    for key, value in values.items():
        container.setdefault(key, []).append(value)


def waymo_output_dir(root: Path, scene_id: int) -> Path:
    return root / f'waymo_1cam_{scene_id}' / 'drivestudio'


def self_output_dir(root: Path, path_id: int, scene_idx: str) -> Path:
    return root / f'self_1cam_path{path_id}_fixed_tree_gamma_full_{scene_idx}' / 'drivestudio'


def strict_reduce_metric_list(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = set().union(*(m.keys() for m in metric_list))
    out = {}
    for key in keys:
        values = [m.get(key, math.nan) for m in metric_list]
        out[key] = strict_mean(values)
    return out


def nan_reduce_metric_list(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    keys = set().union(*(m.keys() for m in metric_list))
    return {key: nanmean([m.get(key, math.nan) for m in metric_list]) for key in keys}


def seeded_refinement_dir(base: Path, external: bool, seed: int) -> Path:
    suffix = '_external' if external else ''
    return base / f'pbr_refinement{suffix}_seed_{seed}'


def seeded_relight_dir(base: Path, scene: str, external: bool, seed: int) -> Path:
    suffix = '_external' if external else ''
    return base / f'pbr_relight_{scene}_refinement{suffix}_seed_{seed}'


def seed_values_for_path(
    base: Path,
    seeds: List[int],
    relight_scenes: List[str],
    external: bool,
    metric_kind: str,
) -> List[Dict[str, float]]:
    out = []
    image_metric_fn = weighted_image_metrics if external else test_image_metrics
    intrinsic_metric_fn = weighted_intrinsic_metrics if external else test_intrinsic_metrics
    relight_metric_fn = weighted_relight_metrics if external else test_relight_metrics
    for seed in seeds:
        refine_root = seeded_refinement_dir(base, external, seed)
        if metric_kind == 'image':
            out.append(image_metric_fn(load_json(refine_root / 'refined_render_w0.5_sky' / 'image_metrics.json')))
        elif metric_kind == 'intrinsic':
            out.append(intrinsic_metric_fn(load_json(refine_root / 'raw_render' / 'intrinsic_metrics.json')))
        elif metric_kind in {'relight', 'ground_relight'}:
            filename = 'image_metrics_ground.json' if metric_kind == 'ground_relight' else 'image_metrics.json'
            scene_values = []
            for scene in relight_scenes:
                scene_values.append(
                    relight_metric_fn(
                        load_json(seeded_relight_dir(base, scene, external, seed) / 'refined_render_w0.5_sky' / filename)
                    )
                )
            out.append(nan_reduce_metric_list(scene_values))
        else:
            raise ValueError(f'Unsupported metric kind: {metric_kind}')
    return out


def print_multiseed_self_tables(args: argparse.Namespace) -> None:
    rows = [f'path{pid}' for pid in args.path_ids]
    groups = [('Self External', True)]
    if args.include_main:
        groups.insert(0, ('Self', False))

    for label, external in groups:
        for metric_kind, title_suffix in [
            ('image', 'NVS'),
            ('intrinsic', 'Intrinsic'),
            ('relight', 'Relight: Average Across Scenes'),
            ('ground_relight', 'Ground Relight: Average Across Scenes'),
        ]:
            values_by_row = []
            for pid in args.path_ids:
                base = self_output_dir(args.root, pid, args.scene_idx)
                values_by_row.append(seed_values_for_path(base, args.seeds, args.relight_scenes, external, metric_kind))
            print_seed_table(f'{label} Multiseed {title_suffix}', rows, values_by_row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate current experiment metrics from outputs/.')
    parser.add_argument('--root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--scene-idx', type=str, default=DEFAULT_SCENE_IDX)
    parser.add_argument('--waymo-static-ids', type=int, nargs='+', default=DEFAULT_WAYMO_STATIC_IDS)
    parser.add_argument('--waymo-dynamic-ids', type=int, nargs='+', default=DEFAULT_WAYMO_DYNAMIC_IDS)
    parser.add_argument('--path-ids', type=int, nargs='+', default=DEFAULT_PATH_IDS)
    parser.add_argument('--relight-scenes', type=str, nargs='+', default=DEFAULT_RELIGHT_SCENES)
    parser.add_argument('--multiseed', action='store_true', help='Aggregate seed-dependent self outputs and report avg/std across seeds.')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--include-main', action='store_true', help='Also print original/main-path self metrics in addition to external-path tables.')
    args = parser.parse_args()

    if args.multiseed:
        print_multiseed_self_tables(args)
        return

    static_rows = [f'waymo_{wid}' for wid in args.waymo_static_ids]
    static_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    for wid in args.waymo_static_ids:
        data = load_json(waymo_output_dir(args.root, wid) / 'pbr_refinement' / 'refined_render_w0.5_sky' / 'image_metrics.json')
        append_metric_row(static_metrics, test_image_metrics(data))
    print_table('Waymo Static NVS', static_rows, static_metrics)

    dynamic_rows = [f'waymo_{wid}' for wid in args.waymo_dynamic_ids]
    dynamic_metrics = {'psnr': [], 'ssim': [], 'lpips': []}
    for wid in args.waymo_dynamic_ids:
        data = load_json(waymo_output_dir(args.root, wid) / 'pbr_refinement' / 'refined_render_w0.5_sky' / 'image_metrics.json')
        append_metric_row(dynamic_metrics, test_image_metrics(data))
    print_table('Waymo Dynamic NVS', dynamic_rows, dynamic_metrics)

    path_rows = [f'path{pid}' for pid in args.path_ids]

    self_image = {'psnr': [], 'ssim': [], 'lpips': []}
    self_intrinsic = {'normal_mae': [], 'albedo_si_psnr': [], 'roughness_rmse': [], 'metallic_rmse': []}
    self_relight_by_scene = {scene: {'psnr': [], 'ssim': [], 'lpips': []} for scene in args.relight_scenes}
    self_ground_relight_by_scene = {scene: {'psnr': [], 'ssim': [], 'lpips': []} for scene in args.relight_scenes}

    self_external_image = {'psnr': [], 'ssim': [], 'lpips': []}
    self_external_intrinsic = {'normal_mae': [], 'albedo_si_psnr': [], 'roughness_rmse': [], 'metallic_rmse': []}
    self_external_relight_by_scene = {scene: {'psnr': [], 'ssim': [], 'lpips': []} for scene in args.relight_scenes}
    self_external_ground_relight_by_scene = {scene: {'psnr': [], 'ssim': [], 'lpips': []} for scene in args.relight_scenes}

    for pid in args.path_ids:
        base = self_output_dir(args.root, pid, args.scene_idx)

        append_metric_row(self_image, test_image_metrics(load_json(base / 'pbr_refinement' / 'refined_render_w0.5_sky' / 'image_metrics.json')))
        append_metric_row(self_intrinsic, test_intrinsic_metrics(load_json(base / 'pbr_refinement' / 'raw_render' / 'intrinsic_metrics.json')))
        for scene in args.relight_scenes:
            append_metric_row(self_relight_by_scene[scene], test_relight_metrics(load_json(base / f'pbr_relight_{scene}_refinement' / 'refined_render_w0.5_sky' / 'image_metrics.json')))
            append_metric_row(self_ground_relight_by_scene[scene], test_relight_metrics(load_json(base / f'pbr_relight_{scene}_refinement' / 'refined_render_w0.5_sky' / 'image_metrics_ground.json')))

        append_metric_row(self_external_image, weighted_image_metrics(load_json(base / 'pbr_refinement_external' / 'refined_render_w0.5_sky' / 'image_metrics.json')))
        append_metric_row(self_external_intrinsic, weighted_intrinsic_metrics(load_json(base / 'pbr_refinement_external' / 'raw_render' / 'intrinsic_metrics.json')))
        for scene in args.relight_scenes:
            append_metric_row(self_external_relight_by_scene[scene], weighted_relight_metrics(load_json(base / f'pbr_relight_{scene}_refinement_external' / 'refined_render_w0.5_sky' / 'image_metrics.json')))
            append_metric_row(self_external_ground_relight_by_scene[scene], weighted_relight_metrics(load_json(base / f'pbr_relight_{scene}_refinement_external' / 'refined_render_w0.5_sky' / 'image_metrics_ground.json')))


    if args.include_main:
        print_table('Self NVS', path_rows, self_image)
        print_table('Self Intrinsic', path_rows, self_intrinsic)
        for scene in args.relight_scenes:
            print_table(f'Self Relight: {scene}', path_rows, self_relight_by_scene[scene])
        self_relight_avg = {'psnr': [], 'ssim': [], 'lpips': []}
        for i in range(len(args.path_ids)):
            for key in self_relight_avg:
                self_relight_avg[key].append(strict_mean([self_relight_by_scene[scene][key][i] for scene in args.relight_scenes]))
        print_table('Self Relight: Average Across Scenes', path_rows, self_relight_avg)
        for scene in args.relight_scenes:
            print_table(f'Self Ground Relight: {scene}', path_rows, self_ground_relight_by_scene[scene])
        self_ground_relight_avg = {'psnr': [], 'ssim': [], 'lpips': []}
        for i in range(len(args.path_ids)):
            for key in self_ground_relight_avg:
                self_ground_relight_avg[key].append(strict_mean([self_ground_relight_by_scene[scene][key][i] for scene in args.relight_scenes]))
        print_table('Self Ground Relight: Average Across Scenes', path_rows, self_ground_relight_avg)

    print_table('Self External NVS', path_rows, self_external_image)
    print_table('Self External Intrinsic', path_rows, self_external_intrinsic)
    for scene in args.relight_scenes:
        print_table(f'Self External Relight: {scene}', path_rows, self_external_relight_by_scene[scene])
    self_ext_relight_avg = {'psnr': [], 'ssim': [], 'lpips': []}
    for i in range(len(args.path_ids)):
        for key in self_ext_relight_avg:
            self_ext_relight_avg[key].append(strict_mean([self_external_relight_by_scene[scene][key][i] for scene in args.relight_scenes]))
    print_table('Self External Relight: Average Across Scenes', path_rows, self_ext_relight_avg)
    for scene in args.relight_scenes:
        print_table(f'Self External Ground Relight: {scene}', path_rows, self_external_ground_relight_by_scene[scene])
    self_ext_ground_relight_avg = {'psnr': [], 'ssim': [], 'lpips': []}
    for i in range(len(args.path_ids)):
        for key in self_ext_ground_relight_avg:
            self_ext_ground_relight_avg[key].append(strict_mean([self_external_ground_relight_by_scene[scene][key][i] for scene in args.relight_scenes]))
    print_table('Self External Ground Relight: Average Across Scenes', path_rows, self_ext_ground_relight_avg)

if __name__ == '__main__':
    main()

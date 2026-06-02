import argparse
import json
import os
import re
import shlex
import subprocess
import time
from typing import Dict, Iterable, List

try:
    from tools.env_config import (
        apply_legacy_env_overrides,
        conda_python_command,
        conda_run_command,
        load_env_config,
        operation_env as configured_operation_env,
    )
except ModuleNotFoundError:
    from env_config import (  # type: ignore
        apply_legacy_env_overrides,
        conda_python_command,
        conda_run_command,
        load_env_config,
        operation_env as configured_operation_env,
    )


TRAIN_STAGES = [
    "train_stage1a",
    "gen_refine_intrinsics",
    "train_stage1b",
    "train_stage2_light",
    "train_stage3_finetune",
]

DEFAULT_STAGES = ["train", "render", "gen_render", "compute_metrics"]

STAGE_GROUPS = {
    "train": TRAIN_STAGES,
    "render": ["render"],
    "gen_render": ["gen_render"],
    "compute_metrics": ["compute_metrics"],
}

STAGE_ALIASES = {
    "render_eval": "render",
}

PRESET_CONFIGS = {
    ("self", 1): "configs/pipeline/self_1cam_repro.yaml",
    ("self", 3): "configs/pipeline/self_3cam_repro.yaml",
    ("waymo", 1): "configs/pipeline/waymo_1cam_repro.yaml",
    ("waymo", 3): "configs/pipeline/waymo_3cam_repro.yaml",
}

DATASET_CONFIGS = {
    ("self", 1): "self/brdfusion_1cam",
    ("self", 3): "self/brdfusion_3cams",
    ("waymo", 1): "waymo/brdfusion_1cam",
    ("waymo", 3): "waymo/brdfusion_3cams",
}

CAM_IDS = {
    1: "0",
    3: "0 1 2",
}

RENDER_TARGET_NAMES = ["recon", "relight", "shifted_recon", "shifted_relight"]


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if (value[0], value[-1]) in [("'", "'"), ('"', '"')]:
        return value[1:-1]
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _yaml_content_lines(path: str):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))
    return lines


def _load_simple_yaml(path: str) -> Dict[str, object]:
    """Parse the simple YAML subset used by configs/pipeline/*.yaml."""
    lines = _yaml_content_lines(path)
    root = {}
    stack = [(-1, root)]
    for idx, (indent, content) in enumerate(lines):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Invalid indentation near line: {content}")
        parent = stack[-1][1]
        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"List item without list parent near line: {content}")
            parent.append(_parse_scalar(content[2:].strip()))
            continue
        if ":" not in content:
            raise ValueError(f"Expected key: value near line: {content}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            if not isinstance(parent, dict):
                raise ValueError(f"Mapping item without dict parent near line: {content}")
            parent[key] = _parse_scalar(value)
            continue
        next_container = {}
        for next_indent, next_content in lines[idx + 1 :]:
            if next_indent <= indent:
                break
            next_container = [] if next_content.startswith("- ") else {}
            break
        if not isinstance(parent, dict):
            raise ValueError(f"Nested mapping without dict parent near line: {content}")
        parent[key] = next_container
        stack.append((indent, next_container))
    return root


def _load_pipeline_config(path: str) -> Dict[str, object]:
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ModuleNotFoundError:
        OmegaConf = None
    if OmegaConf is not None:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Pipeline config must be a mapping: {path}")
    return loaded


def _quote(value) -> str:
    return shlex.quote(str(value))


def _env_prefix(env: Dict[str, object]) -> str:
    parts = []
    for key, value in env.items():
        if value is None:
            continue
        parts.append(f"{key}={_quote(value)}")
    return " ".join(parts)


def _join(parts: Iterable[str]) -> str:
    return " ".join(p for p in parts if p)


def _opts(value: object) -> str:
    if value is None:
        return ""
    if hasattr(value, "items"):
        return _join(f"{key}={_quote(val)}" for key, val in value.items())
    if isinstance(value, (str, bytes)):
        return str(value)
    try:
        return _join(str(item) for item in value)
    except TypeError:
        return str(value)


def _abs(repo_root: str, path: str | None) -> str | None:
    if path is None:
        return None
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(repo_root, path))


def _safe_tag(value: object, fallback: str = "envmap") -> str:
    tag = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()
    return tag or fallback


def _relight_tag_from_envmap_path(path: object) -> str:
    stem = os.path.splitext(os.path.basename(str(path)))[0]
    return f"relight_{_safe_tag(stem)}"


def _num_cams(cam_ids: object) -> int:
    return len(str(cam_ids).split())


def _strength_tag(value: object) -> str:
    first = str(value).split()[0]
    try:
        text = f"{float(first):.6f}".rstrip("0").rstrip(".")
    except ValueError:
        text = first
    return text if text else "0"


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _cam_tokens(cam_ids: object) -> List[str]:
    return [item for item in str(cam_ids).split() if item] or ["0"]


def _metric_tile_cam_order(cam_ids: object) -> str:
    tokens = _cam_tokens(cam_ids)
    if tokens == ["0", "1", "2"]:
        return "1 0 2"
    return ""


def _derive_light_prior_path(repo_root: str, dataset_kind: str, data_root: str | None, scene_idx: object, explicit: object = None) -> str | None:
    light_prior_path = _abs(repo_root, explicit)
    if light_prior_path is not None or data_root is None or scene_idx is None:
        return light_prior_path
    if dataset_kind == "self":
        return os.path.join(str(data_root), str(scene_idx), "dlenvmap", f"{scene_idx}_envmap_median.exr")
    if dataset_kind == "waymo":
        scene_padded = f"{int(scene_idx):03d}"
        return os.path.join(str(data_root), scene_padded, "dlenvmap", f"{scene_padded}_envmap_median.exr")
    return None


def _scene_root(dataset_kind: str, data_root: str | None, scene_idx: object) -> str | None:
    if data_root is None or scene_idx is None:
        return None
    if dataset_kind == "self":
        return os.path.join(str(data_root), str(scene_idx))
    if dataset_kind == "waymo":
        scene_padded = f"{int(scene_idx):03d}"
        return os.path.join(str(data_root), scene_padded)
    return None


def _prior_dir_opts(scene_root: str | None, postfix: object = "") -> str:
    if scene_root is None:
        return ""
    postfix = "" if postfix is None else str(postfix)
    return _join([
        f"data.pixel_source.prior_dirs.albedo={_quote(os.path.join(scene_root, 'diffusion_renderer_albedo' + postfix))}",
        f"data.pixel_source.prior_dirs.roughness={_quote(os.path.join(scene_root, 'diffusion_renderer_roughness' + postfix))}",
        f"data.pixel_source.prior_dirs.metallic={_quote(os.path.join(scene_root, 'diffusion_renderer_metallic' + postfix))}",
        f"data.pixel_source.prior_dirs.normal={_quote(os.path.join(scene_root, 'diffusion_renderer_normal' + postfix))}",
        f"data.pixel_source.prior_dirs.mono_depth={_quote(os.path.join(scene_root, 'diffusion_renderer_depth' + postfix))}",
    ])


def _infer_dataset_kind(raw: object) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    if text.startswith("self"):
        return "self"
    if text.startswith("waymo"):
        return "waymo"
    return None


def _infer_cams(cfg: Dict[str, object]) -> int | None:
    if cfg.get("cams", None) is not None:
        return int(cfg["cams"])
    cam_ids = cfg.get("cam_ids", None)
    if cam_ids is not None:
        return _num_cams(cam_ids)
    dataset = str(cfg.get("dataset", ""))
    if "1cam" in dataset or "1cams" in dataset:
        return 1
    if "3cam" in dataset or "3cams" in dataset:
        return 3
    return None


def _resolve_preset_path(args) -> str:
    repo_root = os.getcwd()
    if args.pipeline_config:
        return os.path.abspath(args.pipeline_config)
    if args.dataset is None or args.cams is None:
        raise ValueError("Either --pipeline_config or both --dataset and --cams are required.")
    key = (args.dataset, int(args.cams))
    if key not in PRESET_CONFIGS:
        raise ValueError(f"No preset config for dataset={args.dataset}, cams={args.cams}.")
    return os.path.abspath(os.path.join(repo_root, PRESET_CONFIGS[key]))


def _default_project(dataset_kind: str, cams: int, scene_idx: object, path_id: object | None) -> str:
    if dataset_kind == "self":
        if path_id is None:
            raise ValueError("Self dataset requires --path_id when deriving project name.")
        return f"self_{cams}cam_path{path_id}_{scene_idx}"
    return f"waymo_{cams}cam_{scene_idx}"


def _set_render_targets_from_cli(cfg: Dict[str, object], render_targets: List[str], view_source: str | None) -> None:
    targets = list(render_targets or [])
    if not targets and view_source == "external":
        targets = ["shifted_recon"]
    if not targets:
        return
    if "all" in targets:
        targets = list(RENDER_TARGET_NAMES)
    invalid = [target for target in targets if target not in RENDER_TARGET_NAMES]
    if invalid:
        raise ValueError(f"Invalid render target(s): {invalid}. Valid targets: {RENDER_TARGET_NAMES}")
    cfg["render_targets"] = {
        name: {"enabled": name in targets}
        for name in RENDER_TARGET_NAMES
    }


def _materialize_config(args) -> tuple[Dict[str, object], Dict[str, object]]:
    preset_path = _resolve_preset_path(args)
    cfg = _load_pipeline_config(preset_path)
    static_preset = {
        "path": preset_path,
        "config": json.loads(json.dumps(cfg)),
    }

    use_dynamic_cli = bool(
        args.dataset is not None
        or args.cams is not None
        or args.path_id is not None
        or args.scene_idx is not None
        or args.data_root is not None
        or args.project is not None
        or args.output_root is not None
        or args.start_timestep is not None
        or args.num_frames is not None
        or args.end_timestep is not None
        or not args.pipeline_config
    )

    if use_dynamic_cli:
        dataset_kind = args.dataset or _infer_dataset_kind(cfg.get("dataset", None))
        cams = int(args.cams or _infer_cams(cfg) or 0)
        if dataset_kind not in {"self", "waymo"}:
            raise ValueError("Dynamic CLI requires --dataset self or --dataset waymo.")
        if cams not in {1, 3}:
            raise ValueError("Dynamic CLI requires --cams 1 or --cams 3.")

        start = int(args.start_timestep if args.start_timestep is not None else cfg.get("start_timestep", 0))
        if args.end_timestep is not None:
            end = int(args.end_timestep)
            num_frames = end - start + 1
        else:
            num_frames = int(args.num_frames if args.num_frames is not None else cfg.get("num_timesteps", 51))
            end = start + num_frames - 1
        if num_frames <= 0:
            raise ValueError("num_frames/end_timestep must produce at least one frame.")
        test_stride = int(args.test_image_stride if args.test_image_stride is not None else cfg.get("test_image_stride", 10))

        path_id = args.path_id if args.path_id is not None else cfg.get("path_id", None)
        external_data_root = None
        external_scene_idx = None
        relight_scene_idx = args.relight_scene_idx or cfg.get("relight_scene_idx", None)
        external_relight_scene_idx = args.external_relight_scene_idx or cfg.get("external_relight_scene_idx", None)
        relight_tag = args.relight_tag or cfg.get("relight_tag", None)

        if dataset_kind == "self":
            if path_id is None:
                raise ValueError("Self dataset requires --path_id for dynamic derivation.")
            scene_idx = args.scene_idx or cfg.get("scene_idx", "qwantani_moon_noon_puresky_4k")
            path_name = f"path{path_id}_fixed_tree_gamma_full"
            data_root = args.data_root or os.path.join("data", "self", path_name)
            if relight_scene_idx is None:
                relight_scene_idx = f"{scene_idx}_rot90"
            external_data_root = args.external_data_root or os.path.join(
                "data", "self", f"path{path_id}-3-calib_fixed_tree_gamma_full"
            )
            external_scene_idx = args.external_scene_idx or scene_idx
            if external_relight_scene_idx is None:
                external_relight_scene_idx = relight_scene_idx
        else:
            if args.scene_idx is None and cfg.get("scene_idx", None) is None:
                raise ValueError("Waymo dynamic derivation requires --scene_idx.")
            scene_idx = args.scene_idx if args.scene_idx is not None else cfg.get("scene_idx")
            data_root = args.data_root or os.path.join("data", "waymo", "processed", "training")

        repo_root = os.path.abspath(cfg.get("repo_root", os.getcwd()))
        data_root_abs = _abs(repo_root, data_root)
        light_prior_path = args.light_prior_path or cfg.get("light_prior_path", None)
        if light_prior_path is None:
            light_prior_path = _derive_light_prior_path(repo_root, dataset_kind, data_root_abs, scene_idx, None)
        relight_envmap_path = args.relight_envmap_path or cfg.get("relight_envmap_path", None)
        if relight_envmap_path is None and dataset_kind == "self" and relight_scene_idx is not None:
            relight_envmap_path = os.path.join(str(data_root), str(relight_scene_idx), f"{relight_scene_idx}.exr")
        if relight_tag is None:
            if relight_envmap_path is not None:
                relight_tag = _relight_tag_from_envmap_path(relight_envmap_path)
            elif relight_scene_idx is not None:
                relight_tag = f"relight_{_safe_tag(relight_scene_idx)}"

        cfg.update({
            "dataset": DATASET_CONFIGS[(dataset_kind, cams)],
            "dataset_kind": dataset_kind,
            "cams": cams,
            "cam_ids": CAM_IDS[cams],
            "data_root": data_root,
            "scene_idx": scene_idx,
            "start_timestep": start,
            "end_timestep": end,
            "num_timesteps": num_frames,
            "test_image_stride": test_stride,
            "project": args.project or _default_project(dataset_kind, cams, scene_idx, path_id),
            "light_prior_path": light_prior_path,
        })
        if args.start_timestep is not None:
            cfg["render_start_timestep"] = start
        if args.num_frames is not None or args.end_timestep is not None:
            cfg["render_num_timesteps"] = num_frames
        if args.output_root is not None:
            cfg["output_root"] = args.output_root
        if path_id is not None:
            cfg["path_id"] = path_id
        if relight_scene_idx is not None:
            cfg["relight_scene_idx"] = relight_scene_idx
        if relight_tag is not None:
            cfg["relight_tag"] = relight_tag
        if relight_envmap_path is not None:
            cfg["relight_envmap_path"] = relight_envmap_path
        if external_data_root is not None:
            cfg["external_data_root"] = external_data_root
        if external_scene_idx is not None:
            cfg["external_scene_idx"] = external_scene_idx
        if external_relight_scene_idx is not None:
            cfg["external_relight_scene_idx"] = external_relight_scene_idx

        dynamic_context = {
            "mode": "derived",
            "dataset": dataset_kind,
            "cams": cams,
            "path_id": path_id,
            "scene_idx": scene_idx,
            "data_root": data_root,
            "cam_ids": CAM_IDS[cams],
            "start_timestep": start,
            "end_timestep": end,
            "num_timesteps": num_frames,
            "test_image_stride": test_stride,
            "project": cfg["project"],
            "relight_scene_idx": relight_scene_idx,
            "relight_envmap_path": relight_envmap_path,
            "external_data_root": external_data_root,
            "external_scene_idx": external_scene_idx,
            "external_relight_scene_idx": external_relight_scene_idx,
            "light_prior_path": light_prior_path,
        }
    else:
        if cfg.get("dataset", None) is None or cfg.get("data_root", None) is None or cfg.get("scene_idx", None) is None:
            raise ValueError(
                "This pipeline config is a static preset. Provide --dataset, --cams, and the scene selector "
                "(--path_id for self or --scene_idx for waymo), or use a legacy config that contains dataset/data_root/scene_idx."
            )
        dynamic_context = {
            "mode": "legacy_config",
            "dataset": _infer_dataset_kind(cfg.get("dataset", None)),
            "cams": _infer_cams(cfg),
            "scene_idx": cfg.get("scene_idx", None),
            "data_root": cfg.get("data_root", None),
            "project": cfg.get("project", None),
        }

    env_config = load_env_config(args.env_config, repo_root=os.path.abspath(cfg.get("repo_root", os.getcwd())))
    cfg["env_config"] = apply_legacy_env_overrides(env_config, cfg.get("env", {}))

    _set_render_targets_from_cli(cfg, args.render_target, args.view_source)
    _apply_stage_env_overrides(cfg, args.stage_env)
    return cfg, {"static_preset": static_preset, "dynamic_context": dynamic_context}


def _enabled_render_targets(
    cfg: Dict[str, object],
    repo_root: str,
    dataset_kind: str,
    data_root: str | None,
    scene_idx: object,
    stage3_dir: str,
    gen_render_tag: str,
) -> List[Dict[str, object]]:
    raw_target_cfg = cfg.get("render_targets", {}) or {}
    if not hasattr(raw_target_cfg, "get"):
        raise ValueError("render_targets must be a mapping.")

    default_relight_scene_idx = cfg.get("relight_scene_idx", None)
    if default_relight_scene_idx is None and dataset_kind == "self" and scene_idx is not None:
        default_relight_scene_idx = f"{scene_idx}_rot90"
    default_relight_tag = cfg.get("relight_tag", None)

    specs = [
        ("recon", True, "primary", False),
        ("relight", False, "primary", True),
        ("shifted_recon", False, "external", False),
        ("shifted_relight", False, "external", True),
    ]

    targets: List[Dict[str, object]] = []
    for name, default_enabled, default_source, default_relight in specs:
        target_cfg = raw_target_cfg.get(name, {}) or {}
        if not hasattr(target_cfg, "get"):
            raise ValueError(f"render_targets.{name} must be a mapping.")
        enabled = _as_bool(target_cfg.get("enabled", default_enabled), default_enabled)
        if not enabled:
            continue

        source = str(target_cfg.get("source", default_source))
        if source not in {"primary", "external"}:
            raise ValueError(f"render_targets.{name}.source must be 'primary' or 'external', got {source!r}")
        if source == "external" and dataset_kind != "self":
            raise ValueError(f"render target '{name}' uses external shifted path, which is only supported for self datasets.")

        relight = _as_bool(target_cfg.get("relight", default_relight), default_relight)
        relight_scene_idx = target_cfg.get("relight_scene_idx", default_relight_scene_idx)
        relight_envmap_path = _abs(repo_root, target_cfg.get("relight_envmap_path", cfg.get("relight_envmap_path", None)))
        if relight and relight_envmap_path is None and dataset_kind == "self" and data_root is not None and relight_scene_idx is not None:
            relight_envmap_path = os.path.join(str(data_root), str(relight_scene_idx), f"{relight_scene_idx}.exr")
        if relight and relight_envmap_path is None:
            raise ValueError(
                f"render target '{name}' is relighting but no relight_envmap_path is configured. "
                "Set render_targets.<name>.relight_envmap_path or top-level relight_envmap_path."
            )

        configured_relight_tag = target_cfg.get("relight_tag", default_relight_tag)
        if configured_relight_tag is not None:
            relight_tag = str(configured_relight_tag)
        elif relight and relight_envmap_path is not None:
            relight_tag = _relight_tag_from_envmap_path(relight_envmap_path)
        elif relight_scene_idx is not None:
            relight_tag = f"relight_{_safe_tag(relight_scene_idx)}"
        else:
            relight_tag = "relight"

        relight_label = relight_tag[len("relight_"):] if relight_tag.startswith("relight_") else relight_tag
        relight_suffix = "" if relight_label == "relight" else f"_{relight_label}"
        if name == "recon":
            target_label = "original_path"
        elif name == "relight":
            target_label = f"original_path_relight{relight_suffix}"
        elif name == "shifted_recon":
            target_label = "shifted_path"
        else:
            target_label = f"shifted_path_relight{relight_suffix}"
        output_dir = str(target_cfg.get("output_dir", target_label))
        if "postfix" in target_cfg:
            postfix = str(target_cfg["postfix"])
        elif name == "recon":
            postfix = "_eval"
        elif name == "relight":
            postfix = f"_{relight_tag}"
        elif name == "shifted_recon":
            postfix = "_eval_external"
        else:
            postfix = f"_{relight_tag}_external"
        run_root = os.path.dirname(stage3_dir)
        render_root = os.path.join(run_root, "render")
        output_root = _abs(repo_root, target_cfg.get("output_root", os.path.join(render_root, output_dir)))
        if output_root is None:
            raise ValueError(f"render target '{name}' did not resolve to an output_root.")

        source_env: Dict[str, object] = {}
        source_opts: List[str] = []
        metric_flags: List[str] = []
        metric_opts: List[str] = []
        external_data_root = None
        external_scene_idx = None
        external_relight_scene_idx = None
        if source == "external":
            external_data_root = _abs(repo_root, target_cfg.get("external_data_root", cfg.get("external_data_root", None)))
            external_scene_idx = target_cfg.get("external_scene_idx", cfg.get("external_scene_idx", scene_idx))
            external_relight_scene_idx = target_cfg.get(
                "external_relight_scene_idx",
                cfg.get("external_relight_scene_idx", relight_scene_idx),
            )
            if external_data_root is None or external_scene_idx is None:
                raise ValueError(
                    f"render target '{name}' uses source=external but external_data_root/external_scene_idx are incomplete."
                )
            source_env = {
                "DATASET_SOURCE": "external",
                "EXTERNAL_DATA_ROOT": external_data_root,
                "EXTERNAL_SCENE_IDX": external_scene_idx,
                "EXTERNAL_RELIGHT_SCENE_IDX": external_relight_scene_idx if relight else None,
            }
            source_opts = [
                "data.pixel_source.external_source.enable=true",
                f"data.pixel_source.external_source.data_root={_quote(external_data_root)}",
                f"data.pixel_source.external_source.scene_idx={_quote(external_scene_idx)}",
                "data.pixel_source.external_source.active_source=external",
                "data.pixel_source.load_sky_mask=false",
                "data.pixel_source.load_priors=false",
                "data.pixel_source.load_gt_depth=false",
                "data.pixel_source.load_materials=false",
            ]
            metric_flags.extend(["--dataset_source", "external"])
            metric_opts.append("data.pixel_source.load_gt_sky_mask=true")

        if dataset_kind == "self" and not relight:
            metric_opts.append("data.pixel_source.load_gt_intrinsic=true")

        if relight and dataset_kind == "self":
            metric_opts.extend([
                "data.pixel_source.load_relighted_rgb=true",
                f"data.pixel_source.relighted_scene_idx={_quote(relight_scene_idx)}",
                "data.pixel_source.load_materials=false",
            ])
            if source == "external" and external_relight_scene_idx is not None:
                metric_opts.append(
                    f"data.pixel_source.external_source.relighted_scene_idx={_quote(external_relight_scene_idx)}"
                )

        raw_root = os.path.join(output_root, "raw_render")
        gen_root = os.path.join(output_root, f"refined_render_w{gen_render_tag}_sky")
        targets.append({
            "name": name,
            "source": source,
            "relight": relight,
            "relight_scene_idx": relight_scene_idx,
            "relight_envmap_path": relight_envmap_path if relight else None,
            "relight_tag": relight_tag,
            "output_root": output_root,
            "target_label": target_label,
            "raw_root": raw_root,
            "raw_metric_root": os.path.join(raw_root, "videos"),
            "metric_dir": os.path.join(run_root, "metrics", target_label),
            "gen_root": gen_root,
            "gen_video_root": os.path.join(gen_root, "videos"),
            "postfix": postfix,
            "video_root": os.path.join(output_root, "videos"),
            "source_env": source_env,
            "source_opts": source_opts,
            "metric_flags": metric_flags,
            "metric_opts": metric_opts,
        })
    return targets


def _default_prior_num_timesteps(cfg, end: int) -> int:
    explicit = cfg.get("num_timesteps", None)
    if explicit is not None:
        return int(explicit)
    if int(end) >= 0:
        return int(end) + 1
    return 51


def _conda_exe_for_root(conda_root: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(conda_root)), "bin", "conda")


def _conda_run(env_cfg, conda_root: str, env_name: str) -> str:
    conda_exe = env_cfg.get("conda_exe", _conda_exe_for_root(conda_root))
    sanitize = bool(env_cfg.get("sanitize_conda_run", True))
    unset_vars = env_cfg.get(
        "sanitize_unset_vars",
        [
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_LIBRARY_PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "CPATH",
            "LIBRARY_PATH",
            "CMAKE_PREFIX_PATH",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_SHLVL",
            "CONDA_PROMPT_MODIFIER",
        ],
    )
    compiler_cfg = env_cfg.get("compiler", {}) or {}
    if not hasattr(compiler_cfg, "get"):
        compiler_cfg = {}
    cc = env_cfg.get("cc", compiler_cfg.get("cc", "gcc-11"))
    cxx = env_cfg.get("cxx", compiler_cfg.get("cxx", "g++-11"))
    env_prefix = ["env"]
    if sanitize:
        env_prefix.extend(f"-u {_quote(name)}" for name in unset_vars)
    if cc:
        env_prefix.append(f"CC={_quote(cc)}")
    if cxx:
        env_prefix.append(f"CXX={_quote(cxx)}")
    return _join([
        _join(env_prefix),
        _quote(conda_exe),
        "run -v --no-capture-output",
        f"-n {_quote(env_name)}",
    ])


def _conda_python(env_cfg, conda_root: str, env_name: str, pythonpath: str) -> str:
    return _join([
        _conda_run(env_cfg, conda_root, env_name),
        "env",
        f"PYTHONPATH={_quote(pythonpath)}",
        "python",
    ])


def _stage_env(env_cfg, stage: str, role: str, default: str) -> str:
    overrides = env_cfg.get("stage_overrides", env_cfg.get("stages", {}))
    if overrides is None:
        return default
    candidates = [stage]
    if stage == "render":
        candidates.append("render_eval")
    for candidate in candidates:
        value = overrides.get(candidate, None)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            return str(value)
        if hasattr(value, "get"):
            return str(value.get(role, default))
    return default


def _operation_env_name(cfg, operation: str, stage: str | None = None, role: str | None = None) -> str:
    env_config = cfg.get("env_config", None) or load_env_config(repo_root=cfg.get("repo_root", os.getcwd()))
    env_name = configured_operation_env(env_config, operation)
    env = cfg.get("env", {}) or {}
    overrides = env.get("stage_overrides", env.get("stages", {})) if hasattr(env, "get") else {}
    if not stage or not hasattr(overrides, "get"):
        return env_name
    candidates = [stage]
    if stage == "render":
        candidates.append("render_eval")
    for candidate in candidates:
        value = overrides.get(candidate, None)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            return str(value)
        if hasattr(value, "get"):
            if value.get(operation, None) is not None:
                return str(value.get(operation))
            if role and value.get(role, None) is not None:
                return str(value.get(role))
            if value.get("env", None) is not None:
                return str(value.get("env"))
    return env_name


def _single_operation_env_config(cfg, operation: str, stage: str | None = None, role: str | None = None) -> Dict[str, object]:
    base = cfg.get("env_config", None) or load_env_config(repo_root=cfg.get("repo_root", os.getcwd()))
    return {
        "conda_root": base.get("conda_root"),
        "operations": {operation: _operation_env_name(cfg, operation, stage=stage, role=role)},
    }


def _operation_run(cfg, operation: str, stage: str | None = None, role: str | None = None) -> str:
    return conda_run_command(_single_operation_env_config(cfg, operation, stage=stage, role=role), operation)


def _operation_python(cfg, operation: str, pythonpath: str, stage: str | None = None, role: str | None = None) -> str:
    return conda_python_command(_single_operation_env_config(cfg, operation, stage=stage, role=role), operation, pythonpath)


def _apply_stage_env_overrides(cfg, overrides: Iterable[str]) -> None:
    if not overrides:
        return
    env = cfg.setdefault("env", {})
    stage_overrides = env.setdefault("stage_overrides", {})
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --stage_env '{item}'. Use stage=env or stage.role=env.")
        lhs, env_name = item.split("=", 1)
        if not lhs or not env_name:
            raise ValueError(f"Invalid --stage_env '{item}'. Use stage=env or stage.role=env.")
        if "." in lhs:
            stage, role = lhs.split(".", 1)
            current = stage_overrides.get(stage, None)
            if current is None or isinstance(current, (str, bytes)):
                stage_overrides[stage] = {}
            stage_overrides[stage][role] = env_name
        else:
            stage_overrides[lhs] = env_name


def _default_stage_commands(cfg) -> Dict[str, str]:
    repo_root = os.path.abspath(cfg.get("repo_root", os.getcwd()))
    output_root = _abs(repo_root, cfg.get("output_root", "work_dirs"))
    project = cfg.get("project", "brdfusion")
    dataset = cfg.get("dataset", "self/brdfusion_1cam")
    dataset_kind = str(dataset).split("/")[0]
    config_file = _abs(repo_root, cfg.get("config_file", "configs/omnire.yaml"))
    scene_idx = cfg.get("scene_idx", None)
    data_root = _abs(repo_root, cfg.get("data_root", None))
    start = cfg.get("start_timestep", 0)
    end = cfg.get("end_timestep", 50)
    test_stride = cfg.get("test_image_stride", 10)
    cam_ids = cfg.get("cam_ids", "0")
    one_cam = _num_cams(cam_ids) == 1
    train_stage1a_python = _operation_python(cfg, "train_stage1a", repo_root, stage="train_stage1a", role="main")
    gen_refine_render_run = _operation_run(cfg, "render", stage="gen_refine_intrinsics", role="main")
    gen_refine_run = _operation_run(cfg, "gen_refine", stage="gen_refine_intrinsics", role="diffusion_renderer")
    gen_refine_python = _operation_python(cfg, "gen_refine", repo_root, stage="gen_refine_intrinsics", role="diffusion_renderer")
    train_stage1b_python = _operation_python(cfg, "train_stage1b", repo_root, stage="train_stage1b", role="main")
    train_stage2_python = _operation_python(cfg, "train_stage2_light", repo_root, stage="train_stage2_light", role="main")
    train_stage3_python = _operation_python(cfg, "train_stage3_finetune", repo_root, stage="train_stage3_finetune", role="main")
    render_run = _operation_run(cfg, "render", stage="render", role="main")
    gen_render_run = _operation_run(cfg, "gen_render", stage="gen_render", role="diffusion_renderer")
    gen_render_python = _operation_python(cfg, "gen_render", repo_root, stage="gen_render", role="diffusion_renderer")
    compute_metrics_python = _operation_python(cfg, "metric_compute", repo_root, stage="compute_metrics", role="main")
    dr_prior_run = _operation_run(cfg, "dr_prior_generation", stage="inverse_prior_initial", role="diffusion_renderer")
    dl_prior_run = _operation_run(cfg, "diffusionlight_prior_generation", stage="light_prior", role="diffusion_light")
    dl_merge_run = _operation_run(cfg, "diffusionlight_prior_merge", stage="merge_light_prior", role="main")

    common_opts = [
        f"dataset={_quote(dataset)}",
        f"data.start_timestep={_quote(start)}",
        f"data.end_timestep={_quote(end)}",
        f"data.pixel_source.test_image_stride={_quote(test_stride)}",
    ]
    if scene_idx is not None:
        common_opts.append(f"data.scene_idx={_quote(scene_idx)}")
    if data_root is not None:
        common_opts.append(f"data.data_root={_quote(data_root)}")
    opts = " ".join(common_opts)

    light_prior_path = _derive_light_prior_path(repo_root, dataset_kind, data_root, scene_idx, cfg.get("light_prior_path", None))
    light_opts = opts
    if light_prior_path is not None:
        light_opts = _join([opts, f"model.Sky.params.prior_path={_quote(light_prior_path)}"])
    scene_root = _scene_root(dataset_kind, data_root, scene_idx)
    initial_prior_opts = _prior_dir_opts(scene_root, cfg.get("init_dr_postfix", ""))

    cosmos_root = _abs(
        repo_root,
        cfg.get(
            "diffusion_renderer_root",
            os.path.join(repo_root, "third_party", "cosmos1-diffusion-renderer"),
        ),
    )
    dl_root = _abs(
        repo_root,
        cfg.get(
            "diffusion_light_root",
            os.path.join(repo_root, "third_party", "DiffusionLight-Turbo"),
        ),
    )
    num_timesteps = _default_prior_num_timesteps(cfg, end)
    render_start_timestep = cfg.get("render_start_timestep", cfg.get("start_timestep", None))
    render_num_timesteps = cfg.get("render_num_timesteps", cfg.get("num_timesteps", None))
    pipeline_cfg = cfg.get("pipeline", {})
    check_priors = int(pipeline_cfg.get("check_priors", 0))

    stage1a_iters = int(cfg.get("stage1a_iters", cfg.get("stage1_iters", 30000)))
    # Stage 1-2 follows the original scripts: it is a fresh Stage-1 run using
    # refined priors, so downstream cumulative counters start from its own
    # checkpoint step rather than stage1a + stage1b.
    stage1b_iters = int(cfg.get("stage1b_total_iters", cfg.get("stage1b_iters", cfg.get("stage1_iters", 30000))))
    stage2_delta = int(cfg.get("stage2_iters", 5000))
    stage3_delta = int(cfg.get("stage3_iters", 20000))
    stage2_iters = int(cfg.get("stage2_total_iters", stage1b_iters + stage2_delta))
    stage3_iters = int(cfg.get("stage3_total_iters", stage2_iters + stage3_delta))
    stage1b_resume_from_stage1a = bool(cfg.get("stage1b_resume_from_stage1a", False))

    scaleinv_w = cfg.get("scaleinv_w", 0.1)
    scaleinv_w_final = cfg.get("scaleinv_w_final", None)
    diffuse_light_white_w = cfg.get("diffuse_light_white_w", None)
    stage3_stop_reset_alpha_iters = cfg.get("stage3_stop_reset_alpha_iters", None)
    stage3_stop_split_iters = cfg.get("stage3_stop_split_iters", None)

    stage1a_dir = os.path.join(output_root, project, "stage1a")
    stage1b_dir = os.path.join(output_root, project, "stage1b")
    stage2_dir = os.path.join(output_root, project, "stage2_light")
    stage3_dir = os.path.join(output_root, project, "stage3_finetune")
    stage1a_ckpt = os.path.join(stage1a_dir, "checkpoint_final.pth")
    stage1b_ckpt = os.path.join(stage1b_dir, "checkpoint_final.pth")
    stage2_ckpt = os.path.join(stage2_dir, "checkpoint_final.pth")
    stage3_ckpt = os.path.join(stage3_dir, "checkpoint_final.pth")

    gen_refine_strengths = cfg.get("gen_refine_strengths", "0.3")
    gen_render_strengths = cfg.get("gen_render_strengths", "0.5")
    gen_refine_tag = _strength_tag(gen_refine_strengths)
    gen_render_tag = _strength_tag(gen_render_strengths)
    sdedit_script = "run_sdedit_pipeline_1cam.sh" if one_cam else "run_sdedit_pipeline.sh"

    stage1a_render_postfix = "_stage1a_raw_intrinsic"
    stage1a_render_root = os.path.join(output_root, project, "stage1a_render", "original_path")
    stage1a_video_root = os.path.join(stage1a_render_root, "videos")
    raw_intrinsic_root = os.path.join(stage1a_render_root, "raw_intrinsic")
    intrinsic_refine_root = os.path.join(stage1a_dir, "intrinsic_refinement")
    refined_intrinsic_root = os.path.join(
        intrinsic_refine_root,
        f"refined_intrinsic_w{gen_refine_tag}",
        "images",
    )
    refined_prior_opts = _join([
        f"data.pixel_source.prior_dirs.normal={_quote(os.path.join(refined_intrinsic_root, 'normal'))}",
        f"data.pixel_source.prior_dirs.mono_depth={_quote(os.path.join(refined_intrinsic_root, 'normalized_depth'))}",
        f"data.pixel_source.prior_dirs.albedo={_quote(os.path.join(refined_intrinsic_root, 'albedo'))}",
        f"data.pixel_source.prior_dirs.roughness={_quote(os.path.join(refined_intrinsic_root, 'roughness'))}",
        f"data.pixel_source.prior_dirs.metallic={_quote(os.path.join(refined_intrinsic_root, 'metallic'))}",
    ])

    stage1a_extra_opts = _opts(cfg.get("stage1a_extra_opts", cfg.get("stage1_extra_opts", None)))
    stage1b_extra_opts = _opts(cfg.get("stage1b_extra_opts", cfg.get("stage1_extra_opts", None)))
    stage2_computed_opts = [
        f"trainer.losses.env_scaleinv.w={_quote(scaleinv_w)}",
        f"trainer.losses.env_scaleinv.decay_start_step={stage1b_iters}",
        f"trainer.losses.env_scaleinv.decay_end_step={stage2_iters}",
    ]
    if scaleinv_w_final is not None:
        stage2_computed_opts.insert(1, f"trainer.losses.env_scaleinv.w_final={_quote(scaleinv_w_final)}")
    if diffuse_light_white_w is not None:
        stage2_computed_opts.append(f"trainer.losses.diffuse_light_white.w={_quote(diffuse_light_white_w)}")
    stage3_computed_opts = []
    if stage3_stop_reset_alpha_iters is not None:
        stage3_computed_opts.append(
            f"trainer.gaussian_ctrl_general_cfg.stop_reset_alpha_at={stage2_iters + int(stage3_stop_reset_alpha_iters)}"
        )
    if stage3_stop_split_iters is not None:
        stage3_computed_opts.append(
            f"trainer.gaussian_ctrl_general_cfg.stop_split_at={stage2_iters + int(stage3_stop_split_iters)}"
        )

    stage2_extra_opts = _join([_join(stage2_computed_opts), _opts(cfg.get("stage2_extra_opts", None))])
    stage3_extra_opts = _join([_opts(cfg.get("stage3_extra_opts", None)), _join(stage3_computed_opts)])
    render_extra_opts = _opts(cfg.get("render_extra_opts", None))
    metric_extra_opts = _opts(cfg.get("metric_extra_opts", None))
    train_common_flags = []
    if _as_bool(cfg.get("enable_wandb", True), True):
        train_common_flags.append("--enable_wandb")
    vis_frame_mode = cfg.get("vis_frame_mode", "middle")
    if vis_frame_mode not in (None, ""):
        train_common_flags.extend(["--vis_frame_mode", _quote(vis_frame_mode)])
    train_common_flags_str = _join(train_common_flags)
    render_loader_opts = "data.pixel_source.load_images_only=true"
    metric_loader_opts = "data.pixel_source.load_priors=false data.pixel_source.load_materials=false"

    render_targets = _enabled_render_targets(
        cfg,
        repo_root,
        dataset_kind,
        data_root,
        scene_idx,
        stage3_dir,
        gen_render_tag,
    )

    def _chain(commands: List[str]) -> str:
        return " && ".join(command for command in commands if command)

    def _target_source_env(target: Dict[str, object]) -> Dict[str, object]:
        source_env = target.get("source_env", {}) or {}
        return source_env if hasattr(source_env, "items") else {}

    def _render_target_command(target: Dict[str, object]) -> str:
        base_extra = _join([
            "render.render_test=false render.render_novel=null",
            render_loader_opts,
            render_extra_opts,
            _join(target.get("source_opts", [])),
        ])
        env_vars = {
            "CKPT": stage3_ckpt,
            "POSTFIX": target["postfix"],
            "START": render_start_timestep,
            "NUM_FRAMES": render_num_timesteps,
            "VIDEO_OUTPUT_DIR": target["video_root"],
            "CALIBRATE_ENVMAP": 1 if _as_bool(cfg.get("calibrate_envmap", True), True) else 0,
            "EXTRA_OPTS": base_extra,
            **_target_source_env(target),
        }
        if target["relight"]:
            env_vars.update({
                "NEW_ENVMAP": target["relight_envmap_path"],
                "NEW_ENVMAP_RES": cfg.get("relight_envmap_res", 1024),
            })
            return _join([_env_prefix(env_vars), render_run, "bash scripts/applications/render.sh"])
        env_vars["RENDER_VIDEO_POSTFIX"] = "raw_render"
        return _join([_env_prefix(env_vars), render_run, "bash scripts/render/render_checkpoint.sh"])

    def _prepare_camera_envmaps_for_target(target: Dict[str, object], envmap_path: str) -> str:
        if one_cam:
            return ""
        return _join([
            "&&",
            gen_render_python,
            "tools/prepare_camera_envmaps.py",
            f"--config_file {_quote(config_file)}",
            f"--dataset {_quote(dataset)}",
            f"--input {_quote(envmap_path)}",
            f"--output_root {_quote(target['raw_root'])}",
            f"--cam_ids {cam_ids}",
            "--time_idx 0",
            opts,
            render_loader_opts,
            _join(target.get("source_opts", [])),
        ])

    def _gen_render_target_command(target: Dict[str, object]) -> str:
        envmap_path = os.path.join(str(target["video_root"]), "envmap.hdr")
        sky_video = os.path.join(str(target["raw_root"]), "rgb_sky.mp4")
        opacity_video = os.path.join(str(target["raw_root"]), "opacity.mp4")
        envmap_arg = f"--envmap_path {_quote(envmap_path)}" if one_cam else ""
        return _join([
            gen_render_python,
            "tools/prepare_sdedit_inputs.py",
            "--mode forward",
            "--prefer raw_render",
            f"--video_root {_quote(target['video_root'])}",
            f"--output_root {_quote(target['raw_root'])}",
            f"--cam_ids {cam_ids}",
            envmap_arg,
            _prepare_camera_envmaps_for_target(target, envmap_path),
            "&&",
            "(",
            f"cd {_quote(cosmos_root)} &&",
            _env_prefix({
                "FORWARD_INPUT_ROOT": target["raw_root"],
                "FORWARD_OUTPUT_ROOT": target["output_root"],
                "EXPECTED_FRAMES": num_timesteps,
                "STRENGTHS": gen_render_strengths,
            }),
            gen_render_run,
            f"bash {sdedit_script} forward",
            "&&",
            _env_prefix({
                "SKY_OUTPUT_ROOT": target["output_root"],
                "SKY_VIDEO": sky_video,
                "OPACITY_VIDEO": opacity_video,
                "STRENGTHS": gen_render_strengths,
            }),
            gen_render_run,
            f"bash {sdedit_script} sky",
            ")",
        ])

    def _metric_base_args() -> str:
        return _join([
            f"--config_file {_quote(config_file)}",
            f"--start_timestep {_quote(start)}",
            f"--end_timestep {_quote(end)}",
            f"--test_image_stride {_quote(test_stride)}",
            f"--dataset {_quote(dataset)}",
            f"--cam_ids {cam_ids}",
        ])

    def _target_metric_opts(target: Dict[str, object]) -> str:
        return _join([opts, metric_loader_opts, _join(target.get("source_opts", [])), _join(target.get("metric_opts", [])), metric_extra_opts])

    def _compute_metric_command(target: Dict[str, object], video_root: str, output_root: str, prefix: str) -> str:
        output_json = os.path.join(output_root, f"{prefix}_image_metrics.json")
        compute_intrinsic = dataset_kind == "self" and not target["relight"] and prefix == "gen"
        if target["relight"]:
            if dataset_kind != "self":
                return ""
            relight_flags = "--ground_only_relight" if _as_bool(cfg.get("relight_metric_ground_only", True), True) else ""
            return _join([
                compute_metrics_python,
                "tools/compute_video_metrics.py",
                _metric_base_args(),
                f"--relight_video_root {_quote(video_root)}",
                "--relight_video_name pbr_rgb.mp4",
                f"--relight_output_json {_quote(output_json)}",
                relight_flags,
                _join(target.get("metric_flags", [])),
                _target_metric_opts(target),
            ])
        return _join([
            compute_metrics_python,
            "tools/compute_video_metrics.py",
            _metric_base_args(),
            f"--video_root {_quote(video_root)}",
            "--video_name pbr_rgb.mp4",
            f"--image_output_json {_quote(output_json)}",
            f"--intrinsic_video_root {_quote(video_root)}" if compute_intrinsic else "",
            f"--intrinsic_output_json {_quote(os.path.join(output_root, f'{prefix}_intrinsic_metrics.json'))}" if compute_intrinsic else "",
            _join(target.get("metric_flags", [])),
            _target_metric_opts(target),
        ])

    def _metric_target_commands(target: Dict[str, object]) -> List[str]:
        tile_cam_order = _metric_tile_cam_order(cam_ids)
        prepare_raw = _join([
            compute_metrics_python,
            "tools/prepare_metric_videos.py",
            f"--input_root {_quote(target['raw_root'])}",
            f"--output_root {_quote(target['raw_metric_root'])}",
            f"--cam_ids {cam_ids}",
            f"--tile_cam_order {tile_cam_order}" if tile_cam_order else "",
        ])
        commands = [prepare_raw]
        raw_metric = _compute_metric_command(target, str(target["raw_metric_root"]), str(target["metric_dir"]), "raw")
        gen_metric = _compute_metric_command(target, str(target["gen_video_root"]), str(target["metric_dir"]), "gen")
        if raw_metric:
            commands.append(raw_metric)
        if gen_metric:
            commands.append(gen_metric)
        return commands

    return {
        "check_data": _join([
            _env_prefix({
                "DATASET": dataset.split("/")[0],
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "CAM_IDS": cam_ids,
                "CHECK_PRIORS": check_priors,
            }),
            "scripts/data/check_dataset_layout.sh",
        ]),
        "inverse_prior_initial": _join([
            _env_prefix({
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "CAM_IDS": cam_ids,
                "NUM_TIMESTEPS": num_timesteps,
                "DR_ROOT": cosmos_root,
            }),
            dr_prior_run,
            f"bash scripts/priors/run_dr_{dataset_kind}.sh",
        ]),
        "light_prior": _join([
            _env_prefix({
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "NUM_TIMESTEPS": num_timesteps,
                "DL_ROOT": dl_root,
                "CAM_IDS": cam_ids,
                "INTERVAL": cfg.get("diffusion_light_interval", 2),
            }),
            dl_prior_run,
            f"bash scripts/priors/run_dl_{dataset_kind}.sh",
        ]),
        "merge_light_prior": _join([
            _env_prefix({
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "NUM_TIMESTEPS": num_timesteps,
                "DL_ROOT": dl_root,
                "CAM_IDS": cam_ids,
                "INTERVAL": cfg.get("diffusion_light_interval", 2),
            }),
            dl_merge_run,
            f"bash scripts/priors/merge_dl_{dataset_kind}.sh",
        ]),
        "train_stage1a": _join([
            train_stage1a_python,
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage1_raster.yaml",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage1a",
            train_common_flags_str,
            opts,
            initial_prior_opts,
            stage1a_extra_opts,
            f"trainer.optim.num_iters={stage1a_iters}",
        ]),
        "gen_refine_intrinsics": _join([
            _env_prefix({
                "CKPT": stage1a_ckpt,
                "POSTFIX": stage1a_render_postfix,
                "START": render_start_timestep,
                "NUM_FRAMES": render_num_timesteps,
                "VIDEO_OUTPUT_DIR": stage1a_video_root,
                "RENDER_VIDEO_POSTFIX": "raw_intrinsic",
                "NO_PBR": 1,
                "EXTRA_OPTS": _join(["render.render_test=false render.render_novel=null", render_loader_opts, render_extra_opts]),
            }),
            gen_refine_render_run,
            "bash scripts/render/render_checkpoint.sh &&",
            gen_refine_python,
            "tools/prepare_sdedit_inputs.py",
            "--mode inverse",
            "--prefer raw_intrinsic",
            f"--video_root {_quote(stage1a_video_root)}",
            f"--output_root {_quote(raw_intrinsic_root)}",
            f"--cam_ids {cam_ids}",
            "&&",
            f"cd {_quote(cosmos_root)} &&",
            _env_prefix({
                "INTRINSICS": "normal normalized_depth albedo roughness metallic",
                "INVERSE_INPUT_ROOT": raw_intrinsic_root,
                "INVERSE_OUTPUT_ROOT": intrinsic_refine_root,
                "EXPECTED_FRAMES": num_timesteps,
                "STRENGTHS": gen_refine_strengths,
            }),
            gen_refine_run,
            f"bash {sdedit_script} inverse",
        ]),
        "train_stage1b": _join([
            train_stage1b_python,
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage1_raster.yaml",
            f"--resume_from {_quote(stage1a_ckpt)}" if stage1b_resume_from_stage1a else "",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage1b",
            train_common_flags_str,
            opts,
            refined_prior_opts,
            stage1b_extra_opts,
            f"trainer.optim.num_iters={stage1b_iters}",
        ]),
        "train_stage2_light": _join([
            train_stage2_python,
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage2_light.yaml",
            f"--resume_from {_quote(stage1b_ckpt)}",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage2_light",
            train_common_flags_str,
            light_opts,
            refined_prior_opts,
            stage2_extra_opts,
            f"trainer.optim.num_iters={stage2_iters}",
        ]),
        "train_stage3_finetune": _join([
            train_stage3_python,
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage3_finetune.yaml",
            f"--resume_from {_quote(stage2_ckpt)}",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage3_finetune",
            train_common_flags_str,
            light_opts,
            refined_prior_opts,
            stage3_extra_opts,
            f"trainer.optim.num_iters={stage3_iters}",
        ]),
        "render": _chain([_render_target_command(target) for target in render_targets]),
        "gen_render": _chain([_gen_render_target_command(target) for target in render_targets]),
        "compute_metrics": _chain([command for target in render_targets for command in _metric_target_commands(target)]),
    }


def _stage_path_satisfied(item: Dict[str, str]) -> bool:
    path = item.get("path")
    kind = item.get("kind", "exists")
    if not path:
        return True
    if kind == "file":
        return os.path.isfile(path)
    if kind == "dir":
        return os.path.isdir(path)
    if kind == "nonempty_dir":
        return os.path.isdir(path) and any(os.scandir(path))
    return os.path.exists(path)


def _default_stage_outputs(cfg) -> Dict[str, List[Dict[str, str]]]:
    repo_root = os.path.abspath(cfg.get("repo_root", os.getcwd()))
    output_root = _abs(repo_root, cfg.get("output_root", "work_dirs"))
    project = cfg.get("project", "brdfusion")
    dataset = cfg.get("dataset", "self/brdfusion_1cam")
    dataset_kind = str(dataset).split("/")[0]
    data_root = _abs(repo_root, cfg.get("data_root", None))
    scene_idx = cfg.get("scene_idx", None)
    gen_refine_tag = _strength_tag(cfg.get("gen_refine_strengths", "0.3"))
    gen_render_tag = _strength_tag(cfg.get("gen_render_strengths", "0.5"))

    stage1a_dir = os.path.join(output_root, project, "stage1a")
    stage1b_dir = os.path.join(output_root, project, "stage1b")
    stage2_dir = os.path.join(output_root, project, "stage2_light")
    stage3_dir = os.path.join(output_root, project, "stage3_finetune")
    refined_intrinsic_root = os.path.join(
        stage1a_dir,
        "intrinsic_refinement",
        f"refined_intrinsic_w{gen_refine_tag}",
        "images",
    )
    render_targets = _enabled_render_targets(
        cfg,
        repo_root,
        dataset_kind,
        data_root,
        scene_idx,
        stage3_dir,
        gen_render_tag,
    )

    def file(path, desc):
        return {"kind": "file", "path": str(path), "description": desc}

    def nonempty_dir(path, desc):
        return {"kind": "nonempty_dir", "path": str(path), "description": desc}

    def directory(path, desc):
        return {"kind": "dir", "path": str(path), "description": desc}

    render_outputs = []
    gen_render_outputs = []
    metric_outputs = []
    for target in render_targets:
        render_outputs.append(nonempty_dir(target["video_root"], f"{target['name']} render video directory"))
        render_outputs.append(file(os.path.join(str(target["video_root"]), "envmap.hdr"), f"{target['name']} rendered envmap"))
        gen_render_outputs.append(nonempty_dir(target["gen_video_root"], f"{target['name']} Gen. Render video directory"))
        metric_outputs.append(file(os.path.join(str(target["metric_dir"]), "raw_image_metrics.json"), f"{target['name']} raw image metrics"))
        metric_outputs.append(file(os.path.join(str(target["metric_dir"]), "gen_image_metrics.json"), f"{target['name']} Gen. Render image metrics"))
        if dataset_kind == "self" and not target["relight"]:
            metric_outputs.append(file(os.path.join(str(target["metric_dir"]), "gen_intrinsic_metrics.json"), f"{target['name']} Gen. Render intrinsic metrics"))

    return {
        "train_stage1a": [file(os.path.join(stage1a_dir, "checkpoint_final.pth"), "stage1a checkpoint")],
        "gen_refine_intrinsics": [
            nonempty_dir(os.path.join(refined_intrinsic_root, name), f"refined intrinsic {name} outputs")
            for name in ["normal", "normalized_depth", "albedo", "roughness", "metallic"]
        ],
        "train_stage1b": [file(os.path.join(stage1b_dir, "checkpoint_final.pth"), "stage1b checkpoint")],
        "train_stage2_light": [file(os.path.join(stage2_dir, "checkpoint_final.pth"), "stage2 checkpoint")],
        "train_stage3_finetune": [file(os.path.join(stage3_dir, "checkpoint_final.pth"), "stage3 checkpoint")],
        "render": render_outputs,
        "gen_render": gen_render_outputs,
        "compute_metrics": metric_outputs,
    }


def _default_stage_prerequisites(cfg) -> Dict[str, List[Dict[str, str]]]:
    repo_root = os.path.abspath(cfg.get("repo_root", os.getcwd()))
    output_root = _abs(repo_root, cfg.get("output_root", "work_dirs"))
    project = cfg.get("project", "brdfusion")
    dataset = cfg.get("dataset", "self/brdfusion_1cam")
    dataset_kind = str(dataset).split("/")[0]
    data_root = _abs(repo_root, cfg.get("data_root", None))
    scene_idx = cfg.get("scene_idx", None)
    end = cfg.get("end_timestep", 50)
    gen_refine_strengths = cfg.get("gen_refine_strengths", "0.3")
    gen_render_strengths = cfg.get("gen_render_strengths", "0.5")
    stage1b_resume_from_stage1a = bool(cfg.get("stage1b_resume_from_stage1a", False))
    gen_refine_tag = _strength_tag(gen_refine_strengths)
    gen_render_tag = _strength_tag(gen_render_strengths)

    stage1a_dir = os.path.join(output_root, project, "stage1a")
    stage1b_dir = os.path.join(output_root, project, "stage1b")
    stage2_dir = os.path.join(output_root, project, "stage2_light")
    stage3_dir = os.path.join(output_root, project, "stage3_finetune")
    stage1a_ckpt = os.path.join(stage1a_dir, "checkpoint_final.pth")
    stage1b_ckpt = os.path.join(stage1b_dir, "checkpoint_final.pth")
    stage2_ckpt = os.path.join(stage2_dir, "checkpoint_final.pth")
    stage3_ckpt = os.path.join(stage3_dir, "checkpoint_final.pth")
    refined_intrinsic_root = os.path.join(stage1a_dir, "intrinsic_refinement", f"refined_intrinsic_w{gen_refine_tag}", "images")
    refined_prior_dirs = [
        os.path.join(refined_intrinsic_root, name)
        for name in ["normal", "normalized_depth", "albedo", "roughness", "metallic"]
    ]
    render_targets = _enabled_render_targets(
        cfg,
        repo_root,
        dataset_kind,
        data_root,
        scene_idx,
        stage3_dir,
        gen_render_tag,
    )

    light_prior_path = _derive_light_prior_path(repo_root, dataset_kind, data_root, scene_idx, cfg.get("light_prior_path", None))

    def file(path, desc):
        return {"kind": "file", "path": str(path), "description": desc}

    def directory(path, desc):
        return {"kind": "dir", "path": str(path), "description": desc}

    light_reqs = [file(light_prior_path, "merged light prior envmap")] if light_prior_path else []
    refined_reqs = [directory(path, f"refined intrinsic prior directory {os.path.basename(path)}") for path in refined_prior_dirs]
    relight_reqs = [
        file(target["relight_envmap_path"], f"{target['name']} relight envmap")
        for target in render_targets
        if target.get("relight") and target.get("relight_envmap_path")
    ]
    render_output_reqs = []
    gen_output_reqs = []
    metric_reqs = []
    for target in render_targets:
        render_output_reqs.append(directory(target["video_root"], f"{target['name']} render video directory"))
        render_output_reqs.append(file(os.path.join(str(target["video_root"]), "envmap.hdr"), f"{target['name']} rendered envmap"))
        gen_output_reqs.append(file(os.path.join(str(target["raw_root"]), "pbr_rgb.mp4"), f"{target['name']} staged raw PBR video"))
        gen_output_reqs.append(directory(target["gen_video_root"], f"{target['name']} Gen. Render video directory"))
        metric_reqs.extend(gen_output_reqs[-2:])
    return {
        "train_stage1a": [],
        "gen_refine_intrinsics": [file(stage1a_ckpt, "stage1a checkpoint")],
        "train_stage1b": ([file(stage1a_ckpt, "stage1a checkpoint")] if stage1b_resume_from_stage1a else []) + refined_reqs,
        "train_stage2_light": [file(stage1b_ckpt, "stage1b checkpoint")] + refined_reqs + light_reqs,
        "train_stage3_finetune": [file(stage2_ckpt, "stage2 checkpoint")] + refined_reqs + light_reqs,
        "render": [file(stage3_ckpt, "stage3 checkpoint")] + relight_reqs,
        "gen_render": render_output_reqs,
        "compute_metrics": metric_reqs,
    }


def _derived_paths(cfg: Dict[str, object]) -> Dict[str, object]:
    repo_root = os.path.abspath(cfg.get("repo_root", os.getcwd()))
    output_root = _abs(repo_root, cfg.get("output_root", "work_dirs"))
    project = cfg.get("project", "brdfusion")
    dataset = cfg.get("dataset", "self/brdfusion_1cam")
    dataset_kind = str(dataset).split("/")[0]
    data_root = _abs(repo_root, cfg.get("data_root", None))
    scene_idx = cfg.get("scene_idx", None)
    gen_refine_tag = _strength_tag(cfg.get("gen_refine_strengths", "0.3"))
    gen_render_tag = _strength_tag(cfg.get("gen_render_strengths", "0.5"))

    stage1a_dir = os.path.join(output_root, project, "stage1a")
    stage1b_dir = os.path.join(output_root, project, "stage1b")
    stage2_dir = os.path.join(output_root, project, "stage2_light")
    stage3_dir = os.path.join(output_root, project, "stage3_finetune")
    refined_root = os.path.join(stage1a_dir, "intrinsic_refinement", f"refined_intrinsic_w{gen_refine_tag}", "images")
    scene_root = _scene_root(dataset_kind, data_root, scene_idx)
    targets = _enabled_render_targets(cfg, repo_root, dataset_kind, data_root, scene_idx, stage3_dir, gen_render_tag)
    initial_postfix = str(cfg.get("init_dr_postfix", "") or "")
    initial_prior_dirs = {}
    if scene_root is not None:
        initial_prior_dirs = {
            "albedo": os.path.join(scene_root, "diffusion_renderer_albedo" + initial_postfix),
            "roughness": os.path.join(scene_root, "diffusion_renderer_roughness" + initial_postfix),
            "metallic": os.path.join(scene_root, "diffusion_renderer_metallic" + initial_postfix),
            "normal": os.path.join(scene_root, "diffusion_renderer_normal" + initial_postfix),
            "mono_depth": os.path.join(scene_root, "diffusion_renderer_depth" + initial_postfix),
        }
    return {
        "light_prior_path": _derive_light_prior_path(repo_root, dataset_kind, data_root, scene_idx, cfg.get("light_prior_path", None)),
        "scene_root": scene_root,
        "initial_prior_dirs": initial_prior_dirs,
        "refined_prior_dirs": {
            "normal": os.path.join(refined_root, "normal"),
            "mono_depth": os.path.join(refined_root, "normalized_depth"),
            "albedo": os.path.join(refined_root, "albedo"),
            "roughness": os.path.join(refined_root, "roughness"),
            "metallic": os.path.join(refined_root, "metallic"),
        },
        "checkpoints": {
            "stage1a": os.path.join(stage1a_dir, "checkpoint_final.pth"),
            "stage1b": os.path.join(stage1b_dir, "checkpoint_final.pth"),
            "stage2": os.path.join(stage2_dir, "checkpoint_final.pth"),
            "stage3": os.path.join(stage3_dir, "checkpoint_final.pth"),
        },
        "stage1a_render": {
            "video_root": os.path.join(output_root, project, "stage1a_render", "original_path", "videos"),
            "raw_intrinsic_root": os.path.join(output_root, project, "stage1a_render", "original_path", "raw_intrinsic"),
        },
        "render_targets": [
            {
                "name": target["name"],
                "target_label": target["target_label"],
                "video_root": target["video_root"],
                "raw_root": target["raw_root"],
                "gen_root": target["gen_root"],
                "metric_dir": target["metric_dir"],
                "relight_envmap_path": target.get("relight_envmap_path"),
                "source": target.get("source"),
            }
            for target in targets
        ],
    }


def _canonical_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _known_stage_names(config_stages: List[str], commands: Dict[str, str]) -> List[str]:
    names = []
    for collection in (config_stages, STAGE_GROUPS.keys(), STAGE_ALIASES.keys(), commands.keys()):
        for name in collection:
            if name not in names:
                names.append(name)
    return names


def _expand_stage(stage: str, commands: Dict[str, str], known: List[str]) -> List[str]:
    canonical = _canonical_stage(stage)
    if canonical in STAGE_GROUPS:
        return list(STAGE_GROUPS[canonical])
    if canonical in commands:
        return [canonical]
    raise ValueError(f"Unknown stage '{stage}'. Known stages: {known}")


def _expand_stages(stages: List[str], commands: Dict[str, str], known: List[str]) -> List[str]:
    selected = []
    for stage in stages:
        for expanded in _expand_stage(stage, commands, known):
            if expanded not in selected:
                selected.append(expanded)
    return selected


def _explicit_stages(config_stages: List[str], requested: List[str], commands: Dict[str, str]) -> List[str]:
    known = _known_stage_names(config_stages, commands)
    return _expand_stages(requested, commands, known)


def _selected_stages(stages: List[str], start_stage: str, stop_stage: str, commands: Dict[str, str]) -> List[str]:
    canonical_stages = [_canonical_stage(stage) for stage in stages]
    start_stage = _canonical_stage(start_stage)
    stop_stage = _canonical_stage(stop_stage)
    if start_stage not in canonical_stages:
        raise ValueError(f"Unknown start_stage '{start_stage}'. Known stages: {stages}")
    if stop_stage not in canonical_stages:
        raise ValueError(f"Unknown stop_stage '{stop_stage}'. Known stages: {stages}")
    start = canonical_stages.index(start_stage)
    stop = canonical_stages.index(stop_stage)
    if start > stop:
        raise ValueError("start_stage must come before or equal stop_stage.")
    return _expand_stages(canonical_stages[start : stop + 1], commands, _known_stage_names(stages, commands))


def build_manifest(args) -> Dict[str, object]:
    cfg, metadata = _materialize_config(args)
    stages = list(cfg.get("stages", DEFAULT_STAGES))
    commands = _default_stage_commands(cfg)
    commands.update(cfg.get("commands", {}) or {})
    if "render" in commands:
        commands.setdefault("render_eval", commands["render"])
    prerequisites = _default_stage_prerequisites(cfg)
    prerequisites.update(cfg.get("stage_prerequisites", {}) or {})
    if "render" in prerequisites:
        prerequisites.setdefault("render_eval", prerequisites["render"])
    outputs = _default_stage_outputs(cfg)
    outputs.update(cfg.get("stage_outputs", {}) or {})
    if "render" in outputs:
        outputs.setdefault("render_eval", outputs["render"])
    selected = (
        _explicit_stages(stages, args.stage, commands)
        if args.stage
        else _selected_stages(stages, args.start_stage, args.stop_stage, commands)
    )
    preset_path = metadata["static_preset"]["path"]
    return {
        "created_at": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        "pipeline_config": os.path.abspath(preset_path),
        "dry_run": not args.execute,
        "repo_root": os.path.abspath(cfg.get("repo_root", os.getcwd())),
        "static_preset": metadata["static_preset"],
        "dynamic_context": metadata["dynamic_context"],
        "derived_paths": _derived_paths(cfg),
        "config": cfg,
        "stages": [
            {
                "name": name,
                "command": commands[name],
                "enabled": bool(commands.get(name)),
                "prerequisites": list(prerequisites.get(name, [])),
                "outputs": list(outputs.get(name, [])),
            }
            for name in selected
        ],
    }


def validate_prerequisites(manifest: Dict[str, object]) -> None:
    missing = []
    for stage in manifest["stages"]:
        for prereq in stage.get("prerequisites", []):
            path = prereq.get("path")
            kind = prereq.get("kind", "exists")
            desc = prereq.get("description", "prerequisite")
            if not path:
                continue
            if not _stage_path_satisfied(prereq):
                missing.append(f"{stage['name']}: missing {desc} ({kind}) at {path}")
    if missing:
        raise FileNotFoundError("Missing prerequisites for selected stage(s):\n  " + "\n  ".join(missing))


def stage_completed(stage: Dict[str, object]) -> bool:
    outputs = stage.get("outputs", []) or []
    return bool(outputs) and all(_stage_path_satisfied(item) for item in outputs)


def _stage_output_summary(stage: Dict[str, object]) -> str:
    outputs = stage.get("outputs", []) or []
    descriptions = [item.get("description") or item.get("path") for item in outputs]
    return ", ".join(str(item) for item in descriptions if item)


def main() -> None:
    parser = argparse.ArgumentParser("Run the BRDFusion staged pipeline.")
    parser.add_argument("--pipeline_config", default=None, help="Static preset YAML. Optional when --dataset and --cams are provided.")
    parser.add_argument("--env_config", default=None, help="Environment config YAML. Defaults to BRDFUSION_ENV_CONFIG or configs/env.yaml.")
    parser.add_argument("--dataset", choices=["self", "waymo"], default=None, help="Dataset family for dynamic preset selection.")
    parser.add_argument("--cams", type=int, choices=[1, 3], default=None, help="Camera preset for dynamic preset selection.")
    parser.add_argument("--path_id", default=None, help="Self path id, e.g. 1 for path1_fixed_tree_gamma_full.")
    parser.add_argument("--scene_idx", default=None, help="Waymo scene index, or self scene name override.")
    parser.add_argument("--data_root", default=None, help="Override derived final data root/path root.")
    parser.add_argument("--output_root", default=None, help="Override training/render output root.")
    parser.add_argument("--project", default=None, help="Override derived project name.")
    parser.add_argument("--start_timestep", type=int, default=None, help="First timestep. Default: 0.")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of frames. Default: 51.")
    parser.add_argument("--end_timestep", type=int, default=None, help="Inclusive end timestep. Overrides --num_frames if set.")
    parser.add_argument("--test_image_stride", type=int, default=None, help="Evaluation test stride. Default: 10.")
    parser.add_argument("--relight_scene_idx", default=None, help="Self relight scene name. Default: <scene_idx>_rot90.")
    parser.add_argument("--relight_tag", default=None, help="Relight output tag. Default: relight_<envmap-stem>.")
    parser.add_argument("--relight_envmap_path", default=None, help="Explicit relight envmap path. Required for Waymo relight target.")
    parser.add_argument("--light_prior_path", default=None, help="Explicit merged DiffusionLight envmap prior path.")
    parser.add_argument("--external_data_root", default=None, help="Self shifted/external path root. Default: data/self/path<id>-3-calib_fixed_tree_gamma_full.")
    parser.add_argument("--external_scene_idx", default=None, help="Self shifted/external scene name. Default: primary scene_idx.")
    parser.add_argument("--external_relight_scene_idx", default=None, help="Self shifted/external relight scene name. Default: relight_scene_idx.")
    parser.add_argument("--view_source", choices=["main", "external"], default=None, help="Compatibility selector. external maps to shifted_recon unless --render_target is set.")
    parser.add_argument("--render_target", action="append", default=[], choices=RENDER_TARGET_NAMES + ["all"], help="Enabled render target. Repeat for multiple targets. Default comes from preset.")
    parser.add_argument("--output_dir", default=None, help="Directory for pipeline_manifest.json.")
    parser.add_argument("--start_stage", default=DEFAULT_STAGES[0], help="First stage for range execution.")
    parser.add_argument("--stop_stage", default=DEFAULT_STAGES[-1], help="Last stage for range execution.")
    parser.add_argument("--stage", action="append", default=[], help="Run one explicit stage or group. Repeat to run multiple stages in the requested order. Groups: train, render, gen_render, compute_metrics.")
    parser.add_argument("--stage_env", action="append", default=[], help="Override stage env: stage=env or stage.role=env. Roles include main and diffusion_renderer.")
    parser.add_argument("--check_prereqs", action="store_true", help="Check selected stage prerequisites during --dry_run.")
    parser.add_argument("--skip_prereq_check", action="store_true", help="Skip prerequisite checks when executing.")
    parser.add_argument("--rerun_existing", action="store_true", help="Re-run selected stages even if their completion outputs already exist.")
    parser.add_argument("--dry_run", action="store_true", help="Only write the manifest and print selected commands.")
    parser.add_argument("--execute", action="store_true", help="Compatibility flag; execution is now the default.")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        raise SystemExit("--execute and --dry_run cannot be used together.")
    args.execute = not args.dry_run

    try:
        manifest = build_manifest(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    if args.check_prereqs and not args.skip_prereq_check:
        try:
            validate_prerequisites(manifest)
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from None
    output_dir = args.output_dir or os.path.join(
        manifest["config"].get("output_root", "work_dirs"),
        manifest["config"].get("project", "brdfusion"),
        "pipeline",
    )
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "pipeline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest: {manifest_path}")

    for stage in manifest["stages"]:
        print(f"[{stage['name']}] {stage['command']}")
        if args.execute:
            if not args.rerun_existing and stage_completed(stage):
                summary = _stage_output_summary(stage)
                print(f"[{stage['name']}] completed outputs found; skipping. {summary}")
                continue
            if not args.skip_prereq_check:
                try:
                    validate_prerequisites({"stages": [stage]})
                except FileNotFoundError as exc:
                    raise SystemExit(str(exc)) from None
            subprocess.run(stage["command"], shell=True, check=True, cwd=manifest["repo_root"])


if __name__ == "__main__":
    main()

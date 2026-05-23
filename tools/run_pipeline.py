import argparse
import json
import os
import shlex
import subprocess
import time
from typing import Dict, Iterable, List

from omegaconf import OmegaConf


DEFAULT_STAGES = [
    "check_data",
    "inverse_prior_initial",
    "light_prior",
    "train_stage1a",
    "gen_refine_intrinsics",
    "train_stage1b",
    "train_stage2_light",
    "train_stage3_finetune",
    "render_eval",
    "gen_render",
    "compute_metrics",
]


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


def _default_stage_commands(cfg) -> Dict[str, str]:
    repo_root = cfg.get("repo_root", os.getcwd())
    output_root = cfg.get("output_root", "work_dirs")
    project = cfg.get("project", "brdfusion")
    dataset = cfg.get("dataset", "self/3cams")
    config_file = cfg.get("config_file", "configs/omnire.yaml")
    scene_idx = cfg.get("scene_idx", None)
    data_root = cfg.get("data_root", None)
    start = cfg.get("start_timestep", 0)
    end = cfg.get("end_timestep", -1)
    test_stride = cfg.get("test_image_stride", 10)
    env = cfg.get("env", {})
    main_env = env.get("main", "brdfusion")
    cosmos_env = env.get("diffusion_renderer", "cosmos-predict1")
    dl_env = env.get("diffusion_light", "diffusionlight")
    conda_root = env.get("conda_root", "/home_nfs/yi-ray/miniconda3/envs")

    common_opts = [
        f"dataset={_quote(dataset)}",
        f"data.start_timestep={_quote(start)}",
        f"data.end_timestep={_quote(end)}",
    ]
    if scene_idx is not None:
        common_opts.append(f"data.scene_idx={_quote(scene_idx)}")
    if data_root is not None:
        common_opts.append(f"data.data_root={_quote(data_root)}")
    opts = " ".join(common_opts)

    base_python = os.path.join(conda_root, main_env, "bin", "python")
    cosmos_root = cfg.get(
        "diffusion_renderer_root",
        os.path.join(repo_root, "third_party", "cosmos1-diffusion-renderer"),
    )
    dl_root = cfg.get(
        "diffusion_light_root",
        os.path.join(repo_root, "third_party", "DiffusionLight-Turbo"),
    )
    num_timesteps = cfg.get("num_timesteps", end)

    return {
        "check_data": _join([
            _env_prefix({
                "DATASET": dataset.split("/")[0],
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "CAM_IDS": cfg.get("cam_ids", "0 1 2"),
                "CHECK_PRIORS": 1,
            }),
            "scripts/data/check_dataset_layout.sh",
        ]),
        "inverse_prior_initial": _join([
            _env_prefix({
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "NUM_TIMESTEPS": num_timesteps,
                "DR_ROOT": cosmos_root,
            }),
            f"conda run -n {_quote(cosmos_env)}",
            "bash scripts/priors/run_dr_self.sh",
        ]),
        "light_prior": _join([
            _env_prefix({
                "DATA_ROOT": data_root,
                "SCENE": scene_idx,
                "NUM_TIMESTEPS": num_timesteps,
                "DL_ROOT": dl_root,
            }),
            f"conda run -n {_quote(dl_env)}",
            "bash scripts/priors/run_dl_self.sh",
        ]),
        "train_stage1a": _join([
            _quote(base_python),
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage1_raster.yaml",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage1a",
            opts,
        ]),
        "gen_refine_intrinsics": _join([
            f"cd {_quote(cosmos_root)} &&",
            f"conda run -n {_quote(cosmos_env)}",
            "bash run_sdedit_pipeline.sh",
        ]),
        "train_stage1b": _join([
            _quote(base_python),
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage1_raster.yaml",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage1b",
            opts,
        ]),
        "train_stage2_light": _join([
            _quote(base_python),
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage2_light.yaml",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage2_light",
            opts,
        ]),
        "train_stage3_finetune": _join([
            _quote(base_python),
            "tools/train.py",
            f"--config_file {_quote(config_file)}",
            "--config_overlay configs/stage/stage3_finetune.yaml",
            f"--output_root {_quote(output_root)}",
            f"--project {_quote(project)}",
            "--run_name stage3_finetune",
            opts,
        ]),
        "render_eval": _join([
            f"CKPT=${{CKPT:-{_quote(os.path.join(output_root, project, 'stage3_finetune', 'checkpoint_final.pth'))}}}",
            "scripts/render/render_checkpoint.sh",
        ]),
        "gen_render": _join([
            f"cd {_quote(cosmos_root)} &&",
            f"conda run -n {_quote(cosmos_env)}",
            "bash run_sdedit_pipeline.sh",
        ]),
        "compute_metrics": _join([
            _quote(base_python),
            "tools/compute_video_metrics.py",
            f"--config_file {_quote(config_file)}",
            f"--start_timestep {_quote(start)}",
            f"--end_timestep {_quote(end)}",
            f"--test_image_stride {_quote(test_stride)}",
            f"--dataset {_quote(dataset)}",
            "--video_root ${VIDEO_ROOT}",
            "--image_output_json ${IMAGE_JSON}",
            opts,
        ]),
    }


def _selected_stages(stages: List[str], start_stage: str, stop_stage: str) -> List[str]:
    if start_stage not in stages:
        raise ValueError(f"Unknown start_stage '{start_stage}'. Known stages: {stages}")
    if stop_stage not in stages:
        raise ValueError(f"Unknown stop_stage '{stop_stage}'. Known stages: {stages}")
    start = stages.index(start_stage)
    stop = stages.index(stop_stage)
    if start > stop:
        raise ValueError("start_stage must come before or equal stop_stage.")
    return stages[start : stop + 1]


def build_manifest(args) -> Dict[str, object]:
    cfg = OmegaConf.load(args.pipeline_config)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    stages = list(cfg.get("stages", DEFAULT_STAGES))
    commands = _default_stage_commands(cfg)
    commands.update(cfg.get("commands", {}))
    selected = _selected_stages(stages, args.start_stage, args.stop_stage)
    return {
        "created_at": time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()),
        "pipeline_config": os.path.abspath(args.pipeline_config),
        "dry_run": not args.execute,
        "repo_root": os.path.abspath(cfg.get("repo_root", os.getcwd())),
        "config": cfg_dict,
        "stages": [
            {
                "name": name,
                "command": commands[name],
                "enabled": bool(commands.get(name)),
            }
            for name in selected
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser("Run or dry-run the BRDFusion staged pipeline.")
    parser.add_argument("--pipeline_config", required=True, help="YAML pipeline config.")
    parser.add_argument("--output_dir", default=None, help="Directory for pipeline_manifest.json.")
    parser.add_argument("--start_stage", default=DEFAULT_STAGES[0])
    parser.add_argument("--stop_stage", default=DEFAULT_STAGES[-1])
    parser.add_argument("--execute", action="store_true", help="Execute stages. Default is dry-run only.")
    args = parser.parse_args()

    manifest = build_manifest(args)
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
            subprocess.run(stage["command"], shell=True, check=True, cwd=manifest["repo_root"])


if __name__ == "__main__":
    main()

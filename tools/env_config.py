#!/usr/bin/env python3
"""Resolve BRDFusion runtime conda environments from configs/env.yaml."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional


DEFAULT_ENV_CONFIG = {
    "conda_root": "auto",
    "operations": {
        "dr_prior_generation": "cosmos-predict1",
        "diffusionlight_prior_generation": "diffusionlight",
        "diffusionlight_prior_merge": "brdfusion",
        "train_stage1a": "brdfusion",
        "train_stage1b": "brdfusion",
        "train_stage2_light": "brdfusion",
        "train_stage3_finetune": "brdfusion",
        "render": "brdfusion",
        "metric_compute": "brdfusion",
        "sky_mask_extract": "segformer",
        "gen_refine": "cosmos-predict1",
        "gen_render": "cosmos-predict1",
    },
}


MAIN_CODE_OPERATIONS = {
    "diffusionlight_prior_merge",
    "train_stage1a",
    "train_stage1b",
    "train_stage2_light",
    "train_stage3_finetune",
    "render",
    "metric_compute",
}


def compiler_env(env_cfg: Dict[str, object], operation: str) -> Dict[str, str]:
    if operation not in MAIN_CODE_OPERATIONS:
        return {}
    compiler_cfg = env_cfg.get("compiler", {}) or {}
    if not hasattr(compiler_cfg, "get"):
        compiler_cfg = {}
    cc = os.environ.get("CC") or str(env_cfg.get("cc", compiler_cfg.get("cc", "gcc-11")))
    cxx = os.environ.get("CXX") or str(env_cfg.get("cxx", compiler_cfg.get("cxx", "g++-11")))
    out: Dict[str, str] = {}
    if cc:
        out["CC"] = cc
    if cxx:
        out["CXX"] = cxx
    return out


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if (value[0], value[-1]) in [("'", "'"), ('"', '"')]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
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
    lines = _yaml_content_lines(path)
    root: Dict[str, object] = {}
    stack: List[tuple[int, object]] = [(-1, root)]
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
            if not isinstance(parent, MutableMapping):
                raise ValueError(f"Mapping item without dict parent near line: {content}")
            parent[key] = _parse_scalar(value)
            continue
        next_container: object = {}
        for next_indent, next_content in lines[idx + 1:]:
            if next_indent <= indent:
                break
            next_container = [] if next_content.startswith("- ") else {}
            break
        if not isinstance(parent, MutableMapping):
            raise ValueError(f"Nested mapping without dict parent near line: {content}")
        parent[key] = next_container
        stack.append((indent, next_container))
    return root


def _load_yaml(path: str) -> Dict[str, object]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_yaml(path)
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Env config must be a mapping: {path}")
    return loaded


def default_config_path(repo_root: Optional[str] = None) -> str:
    root = Path(repo_root or Path(__file__).resolve().parents[1])
    return str(root / "configs" / "env.yaml")


def _detect_conda_root() -> str:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix).expanduser().resolve()
        if prefix.parent.name == "envs":
            return str(prefix.parent)

    conda_exe = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda_exe:
        exe = Path(conda_exe).expanduser().resolve()
        root = exe.parent.parent
        return str(root / "envs")

    default = Path("~/miniconda3/envs").expanduser()
    if default.exists():
        return str(default.resolve())

    raise RuntimeError(
        "Unable to auto-detect conda_root. Activate a conda env, set CONDA_EXE, "
        "put conda on PATH, or set conda_root explicitly in configs/env.yaml."
    )


def resolve_conda_root(conda_root: object) -> str:
    value = str(conda_root or "auto")
    if value.lower() == "auto":
        return _detect_conda_root()
    return str(Path(os.path.expandvars(value)).expanduser())


def load_env_config(path: Optional[str] = None, repo_root: Optional[str] = None) -> Dict[str, object]:
    config_path = path or os.environ.get("BRDFUSION_ENV_CONFIG") or default_config_path(repo_root)
    if os.path.exists(config_path):
        loaded = _load_yaml(config_path)
    else:
        loaded = {}
    cfg = {
        "conda_root": loaded.get("conda_root", DEFAULT_ENV_CONFIG["conda_root"]),
        "operations": dict(DEFAULT_ENV_CONFIG["operations"]),
    }
    operations = loaded.get("operations", {}) or {}
    if not isinstance(operations, dict):
        raise ValueError(f"operations must be a mapping in env config: {config_path}")
    cfg["operations"].update({str(k): str(v) for k, v in operations.items()})
    cfg["path"] = os.path.abspath(config_path)
    return cfg


def apply_legacy_env_overrides(env_cfg: Dict[str, object], legacy: object) -> Dict[str, object]:
    out = {
        "conda_root": env_cfg.get("conda_root", DEFAULT_ENV_CONFIG["conda_root"]),
        "operations": dict(env_cfg.get("operations", {}) or {}),
        "path": env_cfg.get("path"),
    }
    if not hasattr(legacy, "get"):
        return out
    legacy = legacy  # type: ignore[assignment]
    if legacy.get("conda_root") is not None:
        out["conda_root"] = legacy.get("conda_root")
    operations = legacy.get("operations", {}) or {}
    if hasattr(operations, "items"):
        out["operations"].update({str(k): str(v) for k, v in operations.items()})

    # Backward compatibility for old pipeline env blocks.
    main = legacy.get("main")
    diffusion_renderer = legacy.get("diffusion_renderer")
    diffusion_light = legacy.get("diffusion_light")
    if main:
        for op in [
            "diffusionlight_prior_merge",
            "train_stage1a",
            "train_stage1b",
            "train_stage2_light",
            "train_stage3_finetune",
            "render",
            "metric_compute",
        ]:
            out["operations"].setdefault(op, str(main))
    if diffusion_renderer:
        for op in ["dr_prior_generation", "gen_refine", "gen_render"]:
            out["operations"].setdefault(op, str(diffusion_renderer))
    if diffusion_light:
        out["operations"].setdefault("diffusionlight_prior_generation", str(diffusion_light))
    return out


def conda_exe_for_root(conda_root: str) -> str:
    root = resolve_conda_root(conda_root)
    return os.path.join(os.path.dirname(os.path.abspath(root)), "bin", "conda")


def operation_env(env_cfg: Dict[str, object], operation: str) -> str:
    operations = env_cfg.get("operations", {}) or {}
    if not hasattr(operations, "get") or operation not in operations:
        known = ", ".join(sorted(str(k) for k in getattr(operations, "keys", lambda: [])()))
        raise KeyError(f"Unknown env operation '{operation}'. Known operations: {known}")
    return str(operations[operation])


def conda_run_args(env_cfg: Dict[str, object], operation: str) -> List[str]:
    conda_root = resolve_conda_root(env_cfg.get("conda_root", DEFAULT_ENV_CONFIG["conda_root"]))
    return [
        conda_exe_for_root(conda_root),
        "run",
        "-v",
        "--no-capture-output",
        "-n",
        operation_env(env_cfg, operation),
    ]


def shell_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def conda_run_command(env_cfg: Dict[str, object], operation: str) -> str:
    env_parts: List[object] = ["env", "BRDFUSION_IN_CONDA_RUN=1"]
    env_parts.extend(f"{key}={value}" for key, value in compiler_env(env_cfg, operation).items())
    return shell_join([*env_parts, *conda_run_args(env_cfg, operation)])


def conda_python_command(env_cfg: Dict[str, object], operation: str, pythonpath: Optional[str] = None) -> str:
    parts: List[object] = [conda_run_command(env_cfg, operation), "env"]
    if pythonpath:
        parts.append(f"PYTHONPATH={pythonpath}")
    parts.append("python")
    return " ".join(shlex.quote(str(part)) if i > 0 else str(part) for i, part in enumerate(parts))


def exec_in_env(env_cfg: Dict[str, object], operation: str, command: List[str]) -> None:
    if not command:
        raise SystemExit("--exec requires a command")
    env = os.environ.copy()
    env["BRDFUSION_IN_CONDA_RUN"] = "1"
    for key, value in compiler_env(env_cfg, operation).items():
        env.setdefault(key, value)
    args = [*conda_run_args(env_cfg, operation), *command]
    os.execvpe(args[0], args, env)


def main() -> None:
    parser = argparse.ArgumentParser("Resolve BRDFusion environment config.")
    parser.add_argument("--env_config", default=None, help="Override env config path. Defaults to BRDFUSION_ENV_CONFIG or configs/env.yaml.")
    parser.add_argument("--operation", required=True, help="Operation name under operations in env config.")
    parser.add_argument("--print_env", action="store_true", help="Print the conda env name for the operation.")
    parser.add_argument("--print_conda_run", action="store_true", help="Print shell-quoted conda run prefix for the operation.")
    parser.add_argument("--exec", nargs=argparse.REMAINDER, help="Execute the remaining command through the operation env.")
    args = parser.parse_args()
    cfg = load_env_config(args.env_config)
    if args.print_env:
        print(operation_env(cfg, args.operation))
    if args.print_conda_run:
        print(conda_run_command(cfg, args.operation))
    if args.exec is not None:
        command = args.exec
        if command and command[0] == "--":
            command = command[1:]
        exec_in_env(cfg, args.operation, command)


if __name__ == "__main__":
    main()

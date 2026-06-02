import os
from typing import Iterable, Optional

from omegaconf import OmegaConf


def _as_list(paths: Optional[Iterable[str]]) -> list:
    if paths is None:
        return []
    return [p for p in paths if p]


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a / b)
    if not OmegaConf.has_resolver("eq"):
        OmegaConf.register_new_resolver("eq", lambda a, b: a == b)


def resolve_config(
    config_file: str,
    opts: Optional[Iterable[str]] = None,
    dataset_override: Optional[str] = None,
    overlays: Optional[Iterable[str]] = None,
) -> OmegaConf:
    """Load a BRDFusion config with deterministic merge precedence.

    Precedence, from low to high:
    base config -> dataset config -> overlays in command order -> CLI opts.
    """
    _register_resolvers()

    cfg = OmegaConf.load(config_file)
    cli_cfg = OmegaConf.from_cli(list(opts or []))

    if dataset_override is not None:
        cfg.dataset = dataset_override
    if "dataset" in cli_cfg:
        cfg.dataset = cli_cfg.pop("dataset")

    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        cfg = OmegaConf.merge(cfg, dataset_cfg)

    for overlay_path in _as_list(overlays):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(overlay_path))

    return OmegaConf.merge(cfg, cli_cfg)


def resolve_saved_config(
    config_file: str,
    opts: Optional[Iterable[str]] = None,
    overlays: Optional[Iterable[str]] = None,
) -> OmegaConf:
    """Load an already-materialized run config without dataset re-merge.

    Precedence, from low to high:
    saved config -> overlays in command order -> CLI opts.
    """
    _register_resolvers()

    cfg = OmegaConf.load(config_file)
    for overlay_path in _as_list(overlays):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(overlay_path))
    return OmegaConf.merge(cfg, OmegaConf.from_cli(list(opts or [])))

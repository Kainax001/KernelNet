import re
import yaml
from pathlib import Path

# src/config/config.py 기준 → 프로젝트 루트 / experiments
_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_placeholders(obj, root: dict):
    if isinstance(obj, dict):
        return {k: _resolve_placeholders(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_placeholders(v, root) for v in obj]
    if isinstance(obj, str):
        def replacer(match):
            key_path = match.group(1).split(".")
            node = root
            for k in key_path:
                if not isinstance(node, dict) or k not in node:
                    return match.group(0)
                node = node[k]
            return str(node)
        return re.sub(r"\{([^}]+)\}", replacer, obj)
    return obj


def load_config(run_name: str) -> dict:
    base_path = _EXPERIMENTS_DIR / "base.yaml"
    run_path  = _EXPERIMENTS_DIR / "runs" / f"{run_name}.yaml"

    base     = load_yaml(base_path)
    override = load_yaml(run_path) if run_path.exists() else {}
    cfg      = deep_merge(base, override)

    # paths 섹션에만 플레이스홀더 치환 적용
    if "paths" in cfg:
        cfg["paths"] = _resolve_placeholders(cfg["paths"], cfg)

    return cfg

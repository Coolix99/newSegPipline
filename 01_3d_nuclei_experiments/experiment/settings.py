"""Resolve data / output / pretrained-weight caches outside the source tree."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
PATH_KEYS = ('manual_dir','old_results_dir','weak_dir','feature_dir','output_dir','model_cache_dir')

def settings():
    cfg = yaml.safe_load((ROOT/'config.yaml').read_text())
    base = Path(os.environ.get('VSEG_DATA_ROOT',cfg['data_root'])).expanduser().resolve()
    cfg['data_root'] = base
    if base == ROOT or ROOT in base.parents:
        raise ValueError(f'data_root MUST be outside the source repository: {base}')
    for key in PATH_KEYS:
        value=Path(cfg[key]).expanduser()
        dest=(base/value).resolve() if not value.is_absolute() else value.resolve()
        if dest==ROOT or ROOT in dest.parents:
            raise ValueError(f'{key} MUST be outside the source repository: {dest}')
        cfg[key]=dest
    return cfg

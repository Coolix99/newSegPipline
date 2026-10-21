"""Check external paths and pretrained caches. Runs without arguments; never writes data."""
from pathlib import Path
import importlib.util
import numpy as np
from experiment.settings import settings
from experiment.data import list_cases

def main():
    cfg = settings()
    print('External data root:', cfg['data_root'])
    for key in ('manual_dir','old_results_dir','weak_dir','feature_dir','output_dir','model_cache_dir'):
        path = cfg[key]
        print(f'{key:16} {"OK" if path.exists() else "missing (may be normal)"}: {path}')
    cases = list_cases(cfg['manual_dir'])
    print('Manual paired examples:',len(cases))
    if not cases:
        print('ERROR: Provide annotated patch folders under manual_dir. No copying into Git needed.')
    old = sorted(cfg['old_results_dir'].glob('*/*_seg.tiff')) if cfg['old_results_dir'].exists() else []
    weak = list_cases(cfg['weak_dir'])
    print('Old 3D label volumes:',len(old),' / Prepared paired weak patches:',len(weak))
    if old and not weak:
        print('NOTE: old segmentation masks alone are not sufficient for pretraining; matching nuclei images + flow targets are required.')
    for model,src,views in (('N5','microsam',['xy']), ('N6','microsam',['xy','xz','yz']),('N7','cpdino',['xy'])):
        missing=[]
        for path in cases:
            cache=cfg['feature_dir']/src/(path.name+'.npz')
            if not cache.exists():
                missing.append(path.name)
                continue
            try:
                with np.load(cache) as z:
                    if any(v not in z for v in views):missing.append(path.name)
            except Exception:missing.append(path.name)
        print(f'{model}:', 'READY' if cases and not missing else f'{len(missing)} missing/corrupt caches')
    for pkg in ('torch', 'micro_sam','cellpose','dinov3'):
        try: available=importlib.util.find_spec(pkg) is not None
        except (ModuleNotFoundError,ValueError): available=False
        print(f'{pkg}:', 'installed' if available else 'not installed')
    print('To prepare N5/N6/N7: enable allow_foundation_tiny_patches only for small-crop trials,')
    print('install the relevant pretrained dependencies, then run python 04_cache_foundation.py.')
if __name__=='__main__':main()

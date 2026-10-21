"""Run without parameters: python 01_compare_models.py"""
from pathlib import Path
import csv
import json
import subprocess
import sys
import traceback
import torch
import numpy as np
from experiment.settings import ROOT, settings
from experiment.data import list_cases, make_splits, load_case
from experiment.models import PRETRAINED
from experiment.training import run_model
from experiment.plotting import compare_summary, compare_learning_curves, debug_images, debug_motion
from experiment.reconstruction import reconstruct


def main():
    cfg = settings()
    cfg['output_dir'].mkdir(exist_ok=True, parents=True)
    # Automatic unit tests BEFORE any expensive training.
    print('Running unit tests...', flush=True)
    test = subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests')], cwd=str(ROOT))
    if test.returncode != 0: raise SystemExit('Unit tests failed; training aborted.')
    paths = list_cases(cfg['manual_dir'])
    if not paths:
        raise SystemExit(f'No manually annotated patches in {cfg["manual_dir"]}')
    train,val = make_splits(paths, cfg['seed'])
    manifest = {'train':[str(p.name) for p in train], 'val':[str(p.name) for p in val],
                'WARNING': 'Git-hash grouping is a heuristic. Put groups.json with specimen IDs in examples_gt/ before publishing results. Five examples cannot establish reliable model ranking.'}
    split_file = cfg['output_dir']/'split.json'
    if split_file.exists():
        previous = json.loads(split_file.read_text())
        if previous.get('train') != manifest['train'] or previous.get('val') != manifest['val']:
            raise SystemExit('ERROR: train/validation split changed. Use a NEW output_dir for this experiment; '
                             'do not mix scores from different validation sets.')
    split_file.write_text(json.dumps(manifest,indent=2))
    print('Train cases:',manifest['train'],'Val cases:',manifest['val'],flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('WARNING: no CUDA; running a SHORT CPU demonstration of N1 only, not a GPU benchmark.')
        cfg['models'] = ['N1_residual']
        cfg['base_channels'] = 8
        cfg['max_epochs'] = 1
        cfg['max_minutes_per_model'] = 5
    previous_csv = cfg['output_dir']/'comparison.csv'
    previous_rows = {}
    if previous_csv.exists():
        with previous_csv.open(newline='') as f:
            previous_rows = {r['model']:r for r in csv.DictReader(f) if r.get('model')}
    # Recover summaries already on the SSD even when comparison.csv is missing.
    for name in cfg['models']:
        folder = cfg['output_dir']/name
        summary = folder/'summary.json'
        if summary.exists():
            try:
                saved = json.loads(summary.read_text())
                if saved.get('status') == 'complete':
                    previous_rows[name] = saved
            except (ValueError, OSError) as exc:
                print(f'WARN: unreadable summary {summary}: {exc}', flush=True)

    def save_comparison():
        all_rows = sorted(previous_rows.values(), key=lambda r:r['model'])
        if not all_rows:
            return
        with previous_csv.open('w', newline='') as f:
            columns = sorted(set().union(*(row.keys() for row in all_rows)))
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(all_rows)
        figures = [cfg['output_dir']/'model_comparison.png',
                   cfg['output_dir']/'model_learning_curves.png']
        compare_summary(all_rows, figures[0])
        compare_learning_curves(cfg['output_dir'], all_rows, figures[1])
        print(f'COMPARISON CSV: {previous_csv}', flush=True)
        for figure in figures:
            if figure.exists():
                print(f'COMPARISON FIGURE: {figure}', flush=True)

    def refresh_saved_debug(model_dir):
        """New debug plots from cached predictions; no GPU / retraining needed."""
        if not val: return
        for case_dir in val:
            saved = model_dir/'val_predictions'/(case_dir.name+'.npz')
            if not saved.exists(): continue
            case = load_case(case_dir)
            with np.load(saved) as saved_npz:
                prob = saved_npz['probability'].astype(np.float32)
                flow = saved_npz['flow'].astype(np.float32)
            segmented = reconstruct(case['image'],prob,flow,
                                    steps=cfg['propagation_steps'],rate=cfg['propagation_rate'],
                                    eps=cfg['dbscan_eps'],min_samples=cfg['dbscan_min_samples'],
                                    threshold=cfg['foreground_threshold'])
            debug_dir = model_dir/'debug'
            a = debug_images(debug_dir/(case_dir.name+'.png'),
                             case['image'],case['labels'],prob,flow,segmented)
            b = debug_motion(debug_dir/(case_dir.name+'_motion.png'),
                             case['image'],prob,flow,cfg['propagation_steps'],
                             cfg['propagation_rate'],cfg['foreground_threshold'])
            print('  DEBUG FIGURES:',a,';',b,flush=True)
            break  # one representative case per reused model

    for model in cfg['models']:
        model_dir = cfg['output_dir']/model
        print(f'\n===== {model} on {device} =====', flush=True)
        summary = previous_rows.get(model)
        completed = (summary is not None and summary.get('status') == 'complete'
                     and (model_dir/'best.pt').is_file()
                     and (model_dir/'val_predictions').is_dir())
        if cfg.get('reuse_completed', True) and completed:
            print(f'REUSE completed model: {model_dir}', flush=True)
            print(f'  metrics: {model_dir/"summary.json"}', flush=True)
            print(f'  diagnostics: {model_dir/"learning_curves.png"} ; {model_dir/"debug"}', flush=True)
            refresh_saved_debug(model_dir)
            save_comparison()
            continue
        unavailable = None
        if model in PRETRAINED:
            src = 'microsam' if 'microsam' in model else 'cpdino'
            required_views = ('xy','xz','yz') if 'xyz' in model else ('xy',)
            absent = []
            for path in paths:
                f = cfg['feature_dir']/src/(path.name+'.npz')
                if not f.exists():
                    absent.append(path.name)
                    continue
                try:
                    with np.load(f) as z:
                        if any(view not in z for view in required_views):
                            absent.append(path.name)
                except (OSError,ValueError):
                    absent.append(path.name)
            if absent:
                unavailable = (f'Missing or incomplete {src} features for {len(absent)} cases '
                               f'(views {required_views}). Run 04_cache_foundation.py.')
        if unavailable is not None:
            print('SKIP: '+unavailable, flush=True)
            previous_rows[model] = {'model':model,'status':'skipped','error':unavailable}
        else:
            try:
                previous_rows[model] = run_model(model,train,val,cfg,device)
                print(f'MODEL OUTPUT: {model_dir}', flush=True)
                for name in ('summary.json','learning_curves.png'):
                    print(f'  {name}: {model_dir/name}', flush=True)
                for figure in sorted((model_dir/'debug').glob('*.png')):
                    print(f'  DEBUG FIGURE: {figure}', flush=True)
            except (KeyboardInterrupt, SystemExit):
                save_comparison()
                raise
            except Exception as exc:
                traceback.print_exc()
                previous_rows[model] = {'model':model,'status':'failed','error':str(exc)}
        save_comparison()
    print(f'All experiment outputs: {cfg["output_dir"]}', flush=True)

if __name__ == '__main__': main()

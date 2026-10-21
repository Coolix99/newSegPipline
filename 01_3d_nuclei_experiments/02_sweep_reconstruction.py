"""Grid-search reconstruction on *saved validation predictions*. No parameters."""
from pathlib import Path
import csv
import itertools
import time
import json
import numpy as np
from experiment.settings import settings
from experiment.data import load_case, list_cases
from experiment.reconstruction import propagate, cluster_endpoints
from experiment.metrics import match_instances
from experiment.plotting import reconstruction_summary


def main():
    cfg = settings()
    manifest = cfg['output_dir']/'split.json'
    if not manifest.exists():
        print('Run 01_compare_models.py first; no validation split found.');return
    ids = json.loads(manifest.read_text())['val']
    models = sorted(x for x in cfg['output_dir'].iterdir() if (x/'val_predictions').exists())
    if not models:
        print('No validation predictions yet. Run 01_compare_models.py.');return
    cases = {p.name:load_case(p) for p in list_cases(cfg['manual_dir']) if p.name in ids}
    rows=[]
    for modeldir in models:
        start_time=time.monotonic()
        print('Sweep:',modeldir.name,flush=True)
        predictions={}
        for key in list(cases)[:cfg.get('sweep_max_cases',1)]:
            f = modeldir/'val_predictions'/(key+'.npz')
            if f.exists():
                with np.load(f) as a:
                    predictions[key]=(a['probability'].astype(np.float32),a['flow'].astype(np.float32))
        for steps in cfg['sweep_steps']:
            if time.monotonic()-start_time >= 60*cfg.get('sweep_max_minutes_per_model',5):
                print('Stopping sweep at time budget for',modeldir.name);break
            # Propagation is independent of DBSCAN parameters; compute it ONCE per image and length.
            advected={}
            for name,(prob,flow) in predictions.items():
                mask=(prob>=cfg['foreground_threshold']) & (cases[name]['image']>0)
                starts,ends=propagate(mask,flow,steps,cfg['propagation_rate'])
                advected[name]=(mask,starts,ends)
            for eps,min_samples in itertools.product(cfg['sweep_eps'],cfg['sweep_min_samples']):
                if time.monotonic()-start_time >= 60*cfg.get('sweep_max_minutes_per_model',5):
                    print('Stopping after completed parameter combinations');break
                scores=[]
                for name,(mask,starts,ends) in advected.items():
                    seg=cluster_endpoints(mask,starts,ends,eps=eps,min_samples=min_samples)
                    scores.append(match_instances(cases[name]['labels'],seg))
                if not scores:continue
                rows.append({'model':modeldir.name,'steps':steps,'eps':eps,'min_samples':min_samples,'n_cases':len(scores),
                    **{k:float(np.mean([s[k] for s in scores]))
                       for k in ('instance_f1_05','instance_f1_075','pq_05','count_error')}})
        model_csv = modeldir/'reconstruction_sweep.csv'
        with model_csv.open('w',newline='') as f:
            sub=[r for r in rows if r['model']==modeldir.name]
            if sub:
                writer=csv.DictWriter(f,fieldnames=sub[0].keys());writer.writeheader();writer.writerows(sub)
        if sub: print('  MODEL SWEEP CSV:',model_csv,flush=True)
    if rows:
        with (cfg['output_dir']/'reconstruction_comparison.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
        print('COMPARISON SWEEP CSV:',cfg['output_dir']/'reconstruction_comparison.csv',flush=True)
        for fig in reconstruction_summary(rows,cfg['output_dir']):
            print('SWEEP FIGURE:',fig,flush=True)
        ranking=sorted(rows,key=lambda r:r['instance_f1_05'],reverse=True)
        print('Top five (VALIDATION only):')
        for row in ranking[:5]:print(row)
    else:print('No available validation predictions')

if __name__ == '__main__': main()

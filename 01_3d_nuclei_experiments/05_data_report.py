"""Inspect existing-normalization statistics, annotations and specimen groups."""
import csv
import json
from collections import Counter
import numpy as np
from experiment.settings import settings
from experiment.data import list_cases, load_case, source_group
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    cfg=settings();out=cfg['output_dir'];out.mkdir(exist_ok=True,parents=True)
    rows=[]
    for p in list_cases(cfg['manual_dir']):
        c=load_case(p);im=c['image'];nz=im[im>0]
        rows.append({'case':p.name,'group_guess':source_group(p),
                     'fraction_zero':float(np.mean(im==0)),
                     'max':float(im.max()),'p50_nonzero':float(np.percentile(nz,50)) if nz.size else 0.,
                     'p99_nonzero':float(np.percentile(nz,99)) if nz.size else 0.,
                     'n_labels':int(np.unique(c['labels'][c['labels']>0]).size),
                     'labeled_at_zero':int(((c['labels']>0)&(im==0)).sum())})
    if not rows:
        print('No patches found');return
    with (out/'data_report.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for p in list_cases(cfg['manual_dir']):
        nz=load_case(p)['image'];nz=nz[nz>0]
        if nz.size:ax.hist(nz,bins=np.linspace(0,1,80),histtype='step',density=True,label=p.name[:8])
    ax.set_xlabel('Existing normalized intensity (>0)');ax.set_ylabel('Density');ax.legend()
    fig.savefig(out/'intensity_distributions.png',dpi=120);plt.close(fig)
    print('Saved data_report.csv and intensity_distributions.png')
    print('CAUTION: grouping by source Git hash is only a fallback. Supply data/groups.json '
          'with sample-ID -> biological specimen ID for definitive separation.')

if __name__=='__main__':main()

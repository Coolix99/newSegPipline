"""Headless plots; all figures are saved under the external experiment output_dir."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


def _fig_out(fig, output, dpi=130):
    output = Path(output)
    output.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output


def _slice(array, view, coordinate):
    """array has Z,Y,X axes; return XY / XZ / YZ slices."""
    return {'XY': lambda a:a[coordinate,:,:],
            'XZ': lambda a:a[:,coordinate,:],
            'YZ': lambda a:a[:,:,coordinate]}[view](array)


def _overlap_overlay(raw, gt, pred):
    """Green: TP; red: FP; blue: FN, superimposed on input intensities."""
    base = np.clip(raw.astype(np.float32), 0, 1)
    rgb = np.repeat((0.25 + 0.65*base)[...,None], 3, axis=2)
    true = gt > 0
    guess = pred > 0
    rgb[true & guess] = (0.15,0.8,0.28)
    rgb[~true & guess] = (0.95,0.1,0.13)
    rgb[true & ~guess] = (0.13,0.38,0.97)
    return rgb


def debug_images(output, image, truth, probability, flow, instances):
    """Three orthogonal planes with intensity, GT, prediction, probabilities, errors."""
    image, truth = np.asarray(image), np.asarray(truth)
    probability, instances = np.asarray(probability), np.asarray(instances)
    views = ('XY','XZ','YZ')
    axes_for_view = {'XY':0,'XZ':1,'YZ':2}
    labels = ('Input', 'Ground-truth instances','Predicted instances',
              'Foreground probability','TP green / FP red / FN blue')
    fig, ax = plt.subplots(3, 5, figsize=(18, 11), layout='constrained')
    for row, view in enumerate(views):
        along = axes_for_view[view]
        # Prefer a slice containing nuclei over an arbitrarily empty central plane.
        occupancy = (truth > 0).sum(axis=tuple(a for a in range(3) if a != along))
        k = int(np.argmax(occupancy))
        raw = _slice(image,view,k)
        gt = _slice(truth,view,k)
        pr = _slice(instances,view,k)
        prob = _slice(probability,view,k)
        overlay = _overlap_overlay(raw,gt,pr)
        pictures = ((raw,'gray',0,1), (np.ma.masked_equal(gt,0),'nipy_spectral',None,None),
                    (np.ma.masked_equal(pr,0),'nipy_spectral',None,None),
                    (prob,'magma',0,1), (overlay,None,None,None))
        for col,(picture,cmap,vmin,vmax) in enumerate(pictures):
            a = ax[row,col]
            a.imshow(picture, cmap=cmap, vmin=vmin,vmax=vmax,interpolation='nearest')
            a.set_title(f'{view}, slice {k}: {labels[col]}',fontsize=10)
            a.axis('off')
    fig.suptitle('Orthogonal slices; XY / XZ / YZ (Z is never rotated into XY during augmentation)',fontsize=13)
    return _fig_out(fig,output)


def debug_motion(output, image, probability, flow, steps, rate, threshold):
    """Visualize partial flow contraction independently of DBSCAN clustering."""
    from .reconstruction import propagate
    raw = np.asarray(image)
    prob = np.asarray(probability)
    mask = (prob >= threshold) & (raw > 0)
    starts, ends = propagate(mask,flow,steps=steps,rate=rate)
    fig, axes = plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    raw_projection = raw.max(axis=0)
    for ax in axes:
        ax.imshow(raw_projection,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xlim(-0.5,raw.shape[2]-0.5)
        ax.set_ylim(raw.shape[1]-0.5,-0.5)
        ax.axis('off')
    axes[0].set_title('Maximum XY projection + starting foreground')
    axes[1].set_title('XY paths (sampled), short propagation')
    axes[2].set_title('Final endpoint density, XY projection')
    if starts.shape[1]:
        occupancy = mask.any(axis=0)
        axes[0].imshow(np.ma.masked_where(~occupancy,occupancy),alpha=.40,cmap='winter',interpolation='nearest')
        inds = np.linspace(0,starts.shape[1]-1,min(250,starts.shape[1]),dtype=int)
        for j in inds:
            axes[1].plot((starts[2,j],ends[2,j]),(starts[1,j],ends[1,j]),
                         linewidth=.50,alpha=.50,color='tab:orange')
        hist,_,_ = np.histogram2d(ends[1],ends[2],bins=(raw.shape[1],raw.shape[2]),
                                  range=((0,raw.shape[1]),(0,raw.shape[2])))
        axes[2].imshow(np.ma.masked_where(hist==0,np.log1p(hist)),cmap='inferno',alpha=.85,
                       interpolation='nearest')
        displacement = np.sqrt(((ends-starts)**2).sum(axis=0))
        fig.suptitle(f'Flow contraction: steps={steps}, rate={rate}, mean displacement={displacement.mean():.2f} voxels',fontsize=12)
    else:
        fig.suptitle('No foreground voxels; no propagation')
    return _fig_out(fig,output)


def learning_curves(records, output):
    if not records: return
    fig, axes = plt.subplots(1,4,figsize=(15,3.6),layout='constrained')
    for key,a in zip(('total','foreground','flow','dice'),axes):
        for split in ('train','val'):
            vals = [r for r in records if r['split'] == split and r.get(key) is not None]
            if vals:
                a.plot([r['epoch'] for r in vals],[r[key] for r in vals],label=split)
        a.set_xlabel('Epoch');a.set_ylabel(key if key == 'dice' else key+' loss')
        a.grid(alpha=.2);a.legend(fontsize=8)
    return _fig_out(fig,output)


def compare_summary(rows, output):
    rows=[r for r in rows if r.get('status') == 'complete']
    if not rows:return
    rows.sort(key=lambda r:float(r.get('instance_f1_05') or 0),reverse=True)
    names=[r['model'] for r in rows]
    fig, axes=plt.subplots(2,3,figsize=(max(15,len(rows)*1.8),9),layout='constrained')
    specs=[('instance_f1_05','Instance F1 @ IoU 0.50',0,1),
           ('instance_f1_075','Instance F1 @ IoU 0.75',0,1),
           ('foreground_dice','Foreground Dice',0,1),
           ('val_total','Total validation loss',0,None),
           ('val_foreground','Foreground validation loss',0,None),
           ('val_flow','Flow validation loss',0,None)]
    for a,(key,title,low,high) in zip(axes.flat,specs):
        values=[float(r.get(key) or 0) for r in rows]
        a.bar(np.arange(len(names)),values)
        a.set_xticks(np.arange(len(names)),names,rotation=55,ha='right',fontsize=8)
        a.set_title(title);a.grid(axis='y',alpha=.20)
        a.set_ylim(bottom=low,top=high)
        for j,v in enumerate(values):
            a.annotate(f'{v:.3f}',(j,v),xytext=(0,3),textcoords='offset points',
                       ha='center',va='bottom',fontsize=7)
    fig.suptitle('Validation comparison (all completed architectures)',fontsize=13)
    return _fig_out(fig,output)


def compare_learning_curves(output_dir, rows, output):
    complete={r['model'] for r in rows if r.get('status')=='complete'}
    histories=[]
    for name in sorted(complete):
        path=Path(output_dir)/name/'history.csv'
        if not path.exists():continue
        with path.open(newline='') as f:
            vals=[r for r in csv.DictReader(f) if r['split']=='val']
        if vals:histories.append((name,vals))
    if not histories:return
    fig,axes=plt.subplots(1,4,figsize=(17,4),layout='constrained')
    for metric,a in zip(('total','foreground','flow','dice'),axes):
        for name,history in histories:
            a.plot([int(v['epoch']) for v in history],[float(v[metric]) for v in history],label=name)
        a.set_title(('Validation Dice' if metric=='dice' else 'Validation '+metric+' loss'))
        a.set_xlabel('Epoch');a.grid(alpha=.2)
    axes[-1].legend(fontsize=7,loc='best')
    return _fig_out(fig,output)


def reconstruction_summary(rows, output_dir):
    """Diagnostic plots for saved reconstruction parameter sweeps."""
    rows=list(rows)
    if not rows:return []
    models=sorted(set(r['model'] for r in rows))
    steps=sorted(set(int(r['steps']) for r in rows))
    eps_values=sorted(set(float(r['eps']) for r in rows))
    samples=sorted(set(int(r['min_samples']) for r in rows))
    # For each step, best across eps/min_samples, with separate 0.50/0.75 IoUs.
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for model in models:
        for a,key in zip(axes,('instance_f1_05','instance_f1_075')):
            vals=[max((float(r[key]) for r in rows if r['model']==model and int(r['steps'])==step),
                      default=float('nan')) for step in steps]
            a.plot(steps,vals,marker='o',label=model)
    for a,title in zip(axes,('Best F1 @ 0.50 by propagation length','Best F1 @ 0.75 by propagation length')):
        a.set_title(title);a.set_xlabel('Propagation steps');a.set_ylim(0,1)
        a.grid(alpha=.25)
    axes[-1].legend(fontsize=7,loc='best')
    fig.suptitle('Each point chooses best DBSCAN settings; validation data only')
    p1=_fig_out(fig,Path(output_dir)/'reconstruction_by_steps.png')

    # Heatmap at best achieved step for each model; compare eps vs min_samples.
    ncols=min(4,len(models)); nrows=(len(models)+ncols-1)//ncols
    fig,axes=plt.subplots(nrows,ncols,figsize=(3.7*ncols,3.3*nrows),layout='constrained',squeeze=False)
    for idx,model in enumerate(models):
        a=axes.flat[idx]
        sub=[r for r in rows if r['model']==model]
        best=max(sub,key=lambda r:float(r['instance_f1_05']))
        best_step=int(best['steps'])
        grid=np.full((len(samples),len(eps_values)),np.nan)
        for r in sub:
            if int(r['steps'])==best_step:
                grid[samples.index(int(r['min_samples'])),eps_values.index(float(r['eps']))]=float(r['instance_f1_05'])
        a.imshow(np.ma.masked_invalid(grid),vmin=0,vmax=1,cmap='viridis')
        a.set_xticks(range(len(eps_values)),eps_values)
        a.set_yticks(range(len(samples)),samples)
        a.set_xlabel('DBSCAN eps');a.set_ylabel('DBSCAN min_samples')
        a.set_title(f'{model}\\nsteps={best_step}, F1@.5',fontsize=9)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i,j]):
                    a.text(j,i,f'{grid[i,j]:.2f}',ha='center',va='center',
                           fontsize=8,color='white' if grid[i,j]<.50 else 'black')
    for idx in range(len(models),nrows*ncols):axes.flat[idx].axis('off')
    p2=_fig_out(fig,Path(output_dir)/'reconstruction_dbscan_heatmaps.png')
    return [p1,p2]

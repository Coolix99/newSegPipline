"""Napari viewer for the first available validation prediction. No parameters."""
import numpy as np
from experiment.settings import settings
from experiment.data import load_case, list_cases
from experiment.reconstruction import reconstruct, propagate


def main():
    cfg=settings()
    candidates=sorted(cfg['output_dir'].glob('*/val_predictions/*.npz'))
    if not candidates:
        print('No predictions found. Run 01_compare_models.py first.');return
    try:import napari
    except ImportError:
        print('Napari is optional: pip install "napari[all]"');return
    f=candidates[0]
    sample=next((p for p in list_cases(cfg['manual_dir']) if p.name==f.stem),None)
    if sample is None:
        print('Missing matching ground truth for',f);return
    c=load_case(sample)
    with np.load(f) as a:
        prob=a['probability'].astype(np.float32);flow=a['flow'].astype(np.float32)
    result=reconstruct(c['image'],prob,flow,steps=cfg['propagation_steps'],
                       rate=cfg['propagation_rate'],eps=cfg['dbscan_eps'],
                       min_samples=cfg['dbscan_min_samples'],threshold=cfg['foreground_threshold'])
    start,end=propagate((prob>=cfg['foreground_threshold'])&(c['image']>0),flow,
                         cfg['propagation_steps'],cfg['propagation_rate'])
    viewer=napari.Viewer(title=f'{f.parent.parent.name}: {f.stem}')
    viewer.add_image(c['image'],name='Input')
    viewer.add_labels(c['labels'],name='Ground truth')
    viewer.add_image(prob,name='Foreground probability')
    valid=(prob>=cfg['foreground_threshold']) & (c['image']>0)
    positions=np.array(np.nonzero(valid)).T
    if len(positions):
        sampled=positions[::max(1,len(positions)//4000)]
        vectors=flow[:,sampled[:,0],sampled[:,1],sampled[:,2]].T
        viewer.add_vectors(np.stack([sampled,vectors],axis=1),
                           name='Predicted 3D flow (subsampled)',edge_width=.2,length=2)
    viewer.add_labels(result,name='Predicted instances')
    step=max(1, end.shape[1]//10000)
    viewer.add_points(end[:,::step].T,name='Advected endpoints (subsampled)',size=1)
    napari.run()

if __name__=='__main__':main()

"""Cache REAL pretrained embeddings. Optional dependencies / weights may download.

No command-line arguments. Edit config.yaml before running on sufficiently large
patches. Caches are [C,16,16,16] per view, float16; trainable projection is not cached.
"""
import sys
import os
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from experiment.settings import settings
from experiment.data import list_cases, load_case


def get_encoder(name):
    if name == 'microsam':
        try:from micro_sam.util import get_sam_model
        except ImportError as exc:
            raise RuntimeError('Install optional dependency: pip install micro-sam') from exc
        print('Loading μSAM vit_b_lm pretrained weights (may download)...',flush=True)
        predictor = get_sam_model(model_type='vit_b_lm',device='cuda' if torch.cuda.is_available() else 'cpu')
        def predict(slice01):
            # API accepts uint8 HWC RGB, already preprocessed [0,1] microscopy data.
            a = np.clip(slice01 * 255,0,255).astype(np.uint8)
            rgb = np.repeat(a[:,:,None],3,axis=-1)
            predictor.set_image(rgb)
            return predictor.get_image_embedding()[0].detach().float().cpu()
        return predict
    if name == 'cpdino':
        try:from cellpose import models
        except ImportError as exc:
            raise RuntimeError('Install optional dependencies: pip install cellpose; '
                               'then follow Cellpose CPDINO instructions for dinov3') from exc
        print('Loading pretrained cpdino-vitb weights (may download)...',flush=True)
        cp = models.CellposeModel(gpu=torch.cuda.is_available(), pretrained_model='cpdino-vitb')
        net=cp.net.eval()
        device=next(net.parameters()).device
        holder=[]
        def hook(_,inp,out): holder.append(out.detach())
        net.encoder.norm.register_forward_hook(hook)
        def predict(slice01):
            # DINO model expects two image channels in Cellpose; input is 384^2.
            x=torch.from_numpy(np.asarray(slice01,dtype=np.float32))[None,None].to(device)
            x=F.interpolate(x,size=(384,384),mode='bilinear',align_corners=False)
            x=torch.cat([x,torch.zeros_like(x)],dim=1)
            # CellposeDINO may load its patch projection in bfloat16.
            # Match the encoder input dtype to its projection weights.
            patch_proj = net.encoder.patch_embed.proj
            x = x.to(device=patch_proj.weight.device, dtype=patch_proj.weight.dtype)
            holder.clear()
            net(x)
            if not holder:raise RuntimeError('CellposeDINO norm hook captured no tokens; API changed')
            tokens=holder[-1]
            if tokens.ndim != 3:raise RuntimeError('Unexpected CellposeDINO token shape')
            tok=tokens[:,5:,:]  # strip class and storage tokens
            n=tok.shape[1];side=int(n**.5)
            if side*side != n:raise RuntimeError(f'Non-square DINO token grid {n}')
            return tok[0].T.reshape(tok.shape[-1],side,side).float().cpu()
        return predict
    raise ValueError(name)


def encode_view(volume, fn, view, target_shape=(16,16,16)):
    if view=='xy': slices=[volume[z] for z in range(volume.shape[0])]
    elif view=='xz': slices=[volume[:,y,:] for y in range(volume.shape[1])]
    elif view=='yz': slices=[volume[:,:,x] for x in range(volume.shape[2])]
    else:raise ValueError(view)
    feats=[]
    with torch.inference_mode():
        for i,sl in enumerate(slices):
            if not np.any(sl):  # Fill all empty slices after finding a valid feature template.
                feats.append(None)
            else:
                feats.append(fn(sl))
    template=next((f for f in feats if f is not None),None)
    if template is None:raise ValueError('Empty image cannot create pretrained features')
    features=torch.stack([f if f is not None else torch.zeros_like(template) for f in feats],dim=1)
    if view=='xz':features=features.permute(0,2,1,3)
    if view=='yz':features=features.permute(0,2,3,1)
    features=F.interpolate(features.unsqueeze(0),size=target_shape,
                           mode='trilinear',align_corners=False)[0]
    if features.shape[0] not in (256,768):
        raise ValueError(f'Unexpected embedding channels: {features.shape[0]}')
    return features.numpy().astype(np.float16)


def main():
    cfg=settings()
    # Force ALL downloaded model weights outside the repository.
    cfg['model_cache_dir'].mkdir(exist_ok=True, parents=True)
    os.environ['MICROSAM_CACHEDIR'] = str(cfg['model_cache_dir']/'microsam')
    os.environ['CELLPOSE_LOCAL_MODELS_PATH'] = str(cfg['model_cache_dir']/'cellpose')
    paths=list_cases(cfg['manual_dir']) + list_cases(cfg['weak_dir'])
    if not paths:
        raise SystemExit(f'No training patches in {cfg["manual_dir"]} or {cfg["weak_dir"]}')
    if not cfg.get('allow_foundation_tiny_patches',False):
        tiny=[path.name for path in paths if min(load_case(path)['image'].shape[1:]) < 256]
        if tiny:
            print('Found',len(tiny),'patches with 2D sections below 256 pixels. '
                  'Foundation extraction requires larger original tiles for a meaningful test. '
                  'Set allow_foundation_tiny_patches: true to explicitly opt in '
                  'to a technical small-patch experiment. No caches were produced.')
            return
    for encoder in cfg.get('foundation_encoders', ['microsam']):
        if encoder not in ('microsam','cpdino'):
            raise ValueError(f'Unknown foundation encoder {encoder}')
        try:fn=get_encoder(encoder)
        except Exception as e:
            print(f'SKIP {encoder}: {e}');continue
        out=cfg['feature_dir']/encoder
        out.mkdir(exist_ok=True,parents=True)
        for path in paths:
            target=out/(path.name+'.npz')
            if target.exists():
                print('Already cached:', target)
                continue
            case=load_case(path)
            if min(case['image'].shape)<32:
                print('SKIP tiny volume:',path.name);continue
            views=('xy','xz','yz') if encoder=='microsam' else ('xy',)
            try:
                embeddings={view:encode_view(case['image'],fn,view) for view in views}
                np.savez_compressed(target,**embeddings)
                print('Saved',target,flush=True)
            except Exception as e:
                print(f'FAILED {encoder}, {path.name}: {e}',file=sys.stderr)
                raise  # do not write a partial cache or pretend completion

if __name__=='__main__':main()

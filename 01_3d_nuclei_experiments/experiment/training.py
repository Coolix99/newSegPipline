"""Shared time-budget trainer and validation inference for the experiment models."""
import csv
import json
import random
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import NucleiPatches, source_group
from .models import build_model, PRETRAINED
from .losses import components, dice_score
from .reconstruction import reconstruct
from .metrics import match_instances
from .plotting import debug_images, debug_motion, learning_curves


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def loader(paths, train, cfg, model_name):
    dataset = NucleiPatches(paths, train=train, feature_dir=cfg['feature_dir'],
                            model_name=model_name, seed=cfg['seed'] + int(train))
    if not len(dataset): raise ValueError(f'Empty {"training" if train else "validation"} dataset')
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=train,
                      num_workers=cfg['num_workers'], pin_memory=torch.cuda.is_available())


def forward_batch(model, batch, device):
    image = batch['image'].to(device, non_blocking=True)
    profile = batch['profile'].to(device, non_blocking=True)
    features = batch.get('features')
    if features is not None: features = features.to(device, non_blocking=True)
    seg_logits, flows = model(image, profile, features=features)
    return seg_logits, flows, image


def train_epoch(model, dl, optimizer, cfg, device, amp, deadline=None):
    model.train()
    acc = np.zeros(3, dtype=np.float64)
    count = 0
    for batch in dl:
        if deadline and time.monotonic() >= deadline: break
        optimizer.zero_grad(set_to_none=True)
        # On RTX 4090, BF16 has FP32-like exponent range. Never use unscaled FP16.
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.bfloat16):
            seg, flow_pred, img = forward_batch(model, batch, device)
        # Always calculate both losses in full precision, including flow normalization.
        losses = components(seg.float(), flow_pred.float(), img.float(),
                            batch['labels'].to(device), batch['flow'].to(device),
                            cfg['foreground_weight'], cfg['flow_weight'])
        case_ids = batch.get('case_id', ['unknown'])
        if not torch.isfinite(losses['total']):
            raise FloatingPointError(f'Non-finite loss for case(s) {case_ids}; '
                                     f'seg={losses["foreground"].item()}, flow={losses["flow"].item()}')
        losses['total'].backward()
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        except RuntimeError as exc:
            bad = [name for name, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            raise FloatingPointError(f'Non-finite gradients for case(s) {case_ids}; '
                                     f'first affected layers: {bad[:5]}') from exc
        optimizer.step()
        acc += [float(losses[k].detach()) for k in ('total','foreground','flow')]
        count += 1
    return dict(zip(('total','foreground','flow'), (acc/max(1,count)).tolist())), count


@torch.no_grad()
def validate(model, dl, cfg, device, amp):
    model.eval()
    acc = np.zeros(4, dtype=np.float64)
    count = 0
    for batch in dl:
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.bfloat16):
            seg, flow_pred, img = forward_batch(model, batch, device)
        losses = components(seg.float(), flow_pred.float(), img.float(),
                            batch['labels'].to(device), batch['flow'].to(device),
                            cfg['foreground_weight'], cfg['flow_weight'])
        if not torch.isfinite(losses['total']):
            raise FloatingPointError(f'Non-finite validation loss for {batch.get("case_id")}')
        acc[:3] += [float(losses[k].detach()) for k in ('total','foreground','flow')]
        acc[-1] += dice_score(seg, img, batch['labels'].to(device))
        count += 1
    if not count: raise RuntimeError('No validation samples')
    return dict(zip(('total','foreground','flow','dice'), (acc/count).tolist()))


def atomic_checkpoint(path, state):
    path = Path(path)
    tmp = path.with_suffix('.writing')
    torch.save(state, tmp)
    tmp.replace(path)


@torch.no_grad()
def generate_validation_predictions(model, dl, cfg, device, amp, model_dir):
    model.eval()
    results = []
    pred_path = model_dir / 'val_predictions'
    pred_path.mkdir(exist_ok=True, parents=True)
    for batch in dl:
        with torch.autocast(device_type=device.type, enabled=amp, dtype=torch.bfloat16):
            logits, flow_pred, img = forward_batch(model, batch, device)
        probs = logits.float().softmax(dim=1)[:,1].cpu().numpy()
        flows = flow_pred.float().cpu().numpy()
        imgs = img.cpu().numpy()[:,0]
        gt = batch['labels'].numpy()
        for b in range(len(imgs)):
            name = batch['case_id'][b]
            np.savez_compressed(pred_path / (name+'.npz'),
                                probability=probs[b].astype(np.float16),
                                flow=flows[b].astype(np.float16))
            instances = reconstruct(imgs[b], probs[b], flows[b],
                                    steps=cfg['propagation_steps'],rate=cfg['propagation_rate'],
                                    eps=cfg['dbscan_eps'],min_samples=cfg['dbscan_min_samples'],
                                    threshold=cfg['foreground_threshold'])
            stats = match_instances(gt[b], instances)
            stats['case_id'] = name
            stats['foreground_dice'] = float(2*np.logical_and(instances>0, gt[b]>0).sum() /
                       max(1, (instances>0).sum()+(gt[b]>0).sum()))
            results.append(stats)
            debug_images(model_dir / 'debug' / (name+'.png'), imgs[b],gt[b],probs[b],flows[b],instances)
            if len(results)==1:  # one detailed contraction plot per model; save compute/time
                debug_motion(model_dir / 'debug' / (name+'_motion.png'),
                             imgs[b],probs[b],flows[b],cfg['propagation_steps'],
                             cfg['propagation_rate'],cfg['foreground_threshold'])
    (model_dir / 'validation_instances.json').write_text(json.dumps(results, indent=2))
    if not results: raise ValueError('No validation results')
    return {key: float(np.mean([row[key] for row in results]))
            for key in ('instance_f1_05','instance_f1_075','pq_05','pq_075','count_error','foreground_dice')}


def run_model(model_name, train_paths, val_paths, cfg, device):
    seed_all(cfg['seed'])
    model_dir = cfg['output_dir'] / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    train_dl = loader(train_paths, True, cfg, model_name)
    val_dl = loader(val_paths, False, cfg, model_name)
    net = build_model(model_name, cfg['base_channels']).to(device)
    npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
    checkpoint_path = model_dir / 'best.pt'
    recovery_mode = cfg.get('incomplete_checkpoint', 'finalize')
    if recovery_mode not in ('finalize', 'resume'):
        raise ValueError('incomplete_checkpoint must be finalize or resume')
    optimizer = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate'],
                                  weight_decay=cfg['weight_decay'])
    # BF16 is natively supported by RTX 4090; unsupported GPUs use FP32 instead.
    amp = device.type == 'cuda' and torch.cuda.is_bf16_supported()
    records = []
    best_loss = float('inf')
    best_epoch = 0
    begin = time.monotonic()
    recovered = checkpoint_path.exists()
    if recovered:
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if saved.get('model_name') != model_name or saved.get('base_channels') != cfg['base_channels']:
            raise ValueError(f'Existing checkpoint does not match model/config: {checkpoint_path}')
        # A finite validation score does not by itself guarantee finite weights.
        bad_weights = [k for k, tensor in saved['model'].items()
                       if torch.is_tensor(tensor) and tensor.is_floating_point()
                       and not torch.isfinite(tensor).all()]
        if bad_weights:
            raise ValueError(f'Checkpoint has non-finite weights ({bad_weights[:3]}): {checkpoint_path}')
        net.load_state_dict(saved['model'])
        best_epoch = int(saved['epoch'])
        best_loss = float(saved['validation']['total'])
        if not np.isfinite(best_loss):
            raise ValueError(f'Existing checkpoint has non-finite validation loss: {checkpoint_path}')
        # Keep the original history when warm-starting; it must not be replaced
        # with a single checkpoint row and a short new run.
        old_history = model_dir / 'history.csv'
        if recovery_mode == 'resume' and old_history.exists():
            with old_history.open(newline='') as f:
                records = list(csv.DictReader(f))
        if not records:
            records.append({'epoch':best_epoch, 'split':'val', **saved['validation']})
        print(f'  RECOVER best.pt: epoch={best_epoch}, validation loss={best_loss:.4f}; '
              f'mode={recovery_mode}. Checkpoint kept intact: {checkpoint_path}', flush=True)
        if recovery_mode == 'resume':
            print('  Warm-starting with a NEW AdamW optimizer (old optimizer state was not saved).', flush=True)
    run_training = not recovered or recovery_mode == 'resume'
    if recovered and not run_training:
        print('  FINALIZE: no training epochs; evaluating the saved best checkpoint only.', flush=True)
    if run_training and not recovered and cfg.get('pretrain_weak'):
        weak_paths = sorted(p for p in cfg['weak_dir'].glob('*') if p.is_dir() and (p/'nuclei_patch.tif').exists())
        if not weak_paths:
            raise ValueError('Weak pretraining requested but no paired weak images/masks/flows are in data/weak')
        held_out_groups = {source_group(p) for p in val_paths}
        weak_paths = [p for p in weak_paths if source_group(p) not in held_out_groups]
        if not weak_paths:
            raise ValueError('All weak training patches overlap validation groups')
        weak_dl = loader(weak_paths, True, cfg, model_name)
        weak_cfg = {**cfg, 'foreground_weight':cfg.get('weak_foreground_weight',0.5),
                    'flow_weight':cfg.get('weak_flow_weight',0.2)}
        until = time.monotonic() + 60*cfg['weak_pretrain_minutes']
        while time.monotonic() < until:
            loss, steps = train_epoch(net, weak_dl, optimizer, weak_cfg, device, amp, until)
            if not steps: break
    until = time.monotonic() + 60*cfg['max_minutes_per_model']
    completed_epochs = 0
    start_epoch = best_epoch if recovered else 0
    # How many validation checks have passed without a new best score.
    # 0 disables early stopping. Checks occur every validation_every epochs.
    patience = int(cfg.get('early_stopping_patience', 6))
    no_improvement = 0
    if run_training:
        for epoch in range(start_epoch + 1, start_epoch + cfg['max_epochs'] + 1):
            if time.monotonic() >= until: break
            try:
                train_log, steps = train_epoch(net, train_dl, optimizer, cfg, device, amp, until)
                if not steps: break
                completed_epochs += 1
                records.append({'epoch':epoch,'split':'train', **train_log})
                val_log = ''
                stop_now = False
                if epoch % cfg['validation_every'] == 0 or epoch == start_epoch + 1:
                    valid = validate(net,val_dl,cfg,device,amp)
                    records.append({'epoch':epoch,'split':'val', **valid})
                    if valid['total'] < best_loss:
                        best_loss, best_epoch = valid['total'], epoch
                        no_improvement = 0
                        atomic_checkpoint(checkpoint_path,
                                          {'model':net.state_dict(),'model_name':model_name,'epoch':epoch,
                                           'base_channels':cfg['base_channels'],
                                           'config':{k:str(v) for k,v in cfg.items()}, 'validation':valid})
                        best_mark = ' NEW BEST'
                    else:
                        no_improvement += 1
                        best_mark = ''
                    val_log = (f' | VAL seg={valid["foreground"]:.4f} '
                               f'flow={valid["flow"]:.4f} total={valid["total"]:.4f} '
                               f'dice={valid["dice"]:.3f}{best_mark}')
                    stop_now = patience > 0 and no_improvement >= patience
                print(f'  {model_name}: epoch {epoch}/{start_epoch+cfg["max_epochs"]} '
                      f'({steps} steps) TRAIN total={train_log["total"]:.4f} '
                      f'seg={train_log["foreground"]:.4f} flow={train_log["flow"]:.4f}'
                      f'{val_log} | BEST epoch={best_epoch}, val={best_loss:.4f}', flush=True)
                if stop_now:
                    print(f'  EARLY STOP: {no_improvement} validation checks without improvement; '
                          f'restoring best.pt from epoch {best_epoch}.', flush=True)
                    break
            except FloatingPointError as exc:
                if not checkpoint_path.exists():
                    raise
                print(f'  WARNING: {exc}. Stopping and evaluating the last good checkpoint: '
                      f'{checkpoint_path}', flush=True)
                break
    if not checkpoint_path.exists():
        raise RuntimeError(f'No checkpoint saved for {model_name}')
    # Do not clobber prior training history when only finalizing a saved checkpoint.
    if run_training or not (model_dir/'history.csv').exists():
        with (model_dir/'history.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch','split','total','foreground','flow','dice'])
            writer.writeheader(); writer.writerows(records)
        learning_curves(records, model_dir/'learning_curves.png')
    state = torch.load(checkpoint_path,map_location=device,weights_only=True)
    net.load_state_dict(state['model'])
    scores = generate_validation_predictions(net,val_dl,cfg,device,amp,model_dir)
    result = {'model':model_name, 'status':'complete','best_epoch':best_epoch,
              'trained_epochs': start_epoch + completed_epochs, 'trainable_parameters':npar,
              'recovered_existing_checkpoint': recovered, 'recovery_mode': recovery_mode if recovered else None,
              'minutes_total':round((time.monotonic()-begin)/60.,2),
              'val_total':best_loss,
              'val_foreground':state['validation']['foreground'],
              'val_flow':state['validation']['flow'],
              'val_dice':state['validation']['dice'],**scores}
    (model_dir/'summary.json').write_text(json.dumps(result,indent=2))
    return result

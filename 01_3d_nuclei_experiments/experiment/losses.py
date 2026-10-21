"""Losses with exact BZYX masks; zero intensities never supervise."""
import torch
from torch.nn import functional as F


def components(logits, pred_flow, image, labels, flow, fg_weight=1., flow_weight=1.):
    if image.ndim != 5 or labels.ndim != 4 or flow.ndim != 5:
        raise ValueError('Expected image B1ZYX, labels BZYX, flow B3ZYX')
    valid = image[:, 0] > 0
    target = labels > 0
    seg_map = F.cross_entropy(logits, target.long(), reduction='none')
    seg = (seg_map * valid).sum() / valid.sum().clamp(min=1)
    # Unit target vector is defined only where a label AND actual flow exists.
    target_norm = torch.linalg.vector_norm(flow, dim=1)
    flow_valid = valid & target & (target_norm > 1e-6)
    truth = F.normalize(flow, dim=1, eps=1e-6)
    cosine = torch.sum(pred_flow * truth, dim=1).clamp(-1., 1.)
    direction_error = 1. - cosine
    angle = (direction_error * flow_valid).sum() / flow_valid.sum().clamp(min=1)
    loss = fg_weight * seg + flow_weight * angle
    return {'total': loss, 'foreground': seg, 'flow': angle,
            'valid_n': valid.sum().detach(), 'flow_n': flow_valid.sum().detach()}


@torch.no_grad()
def dice_score(logits, image, labels):
    valid = image[:, 0] > 0
    predicted = (logits.softmax(dim=1)[:, 1] >= 0.5) & valid
    truth = (labels > 0) & valid
    tp = torch.sum(predicted & truth).float()
    denom = predicted.sum() + truth.sum()
    return float((2*tp / denom.clamp(min=1)).item()) if denom else 1.

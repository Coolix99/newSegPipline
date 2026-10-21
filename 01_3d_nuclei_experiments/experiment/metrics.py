"""Instance metrics based on one-to-one IoU matching (3D, Hungarian assignment)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_instances(truth, predicted, thresholds=(0.5, 0.75)):
    truth = np.asarray(truth)
    predicted = np.asarray(predicted)
    if truth.shape != predicted.shape: raise ValueError('Mismatch instance volumes')
    g, gt_inv = np.unique(truth, return_inverse=True)
    p, pr_inv = np.unique(predicted, return_inverse=True)
    # Voxel count as sparse contingency table; ignore background in matching.
    n_p = len(p)
    pairs, counts = np.unique(gt_inv.astype(np.int64)*n_p + pr_inv, return_counts=True)
    areas_gt = np.bincount(gt_inv.ravel(), minlength=len(g))
    areas_pr = np.bincount(pr_inv.ravel(), minlength=len(p))
    valid_gt = g > 0
    valid_pr = p > 0
    gi = np.flatnonzero(valid_gt)
    pi = np.flatnonzero(valid_pr)
    iou = np.zeros((len(gi), len(pi)), dtype=np.float32)
    g_lookup = {int(v): i for i,v in enumerate(gi)}
    p_lookup = {int(v): i for i,v in enumerate(pi)}
    for pair,n in zip(pairs,counts):
        a, b = divmod(int(pair), n_p)
        if a in g_lookup and b in p_lookup:
            iou[g_lookup[a], p_lookup[b]] = n / (areas_gt[a]+areas_pr[b]-n)
    ans = {'n_gt': len(gi), 'n_pred': len(pi), 'count_error': abs(len(gi)-len(pi))}
    for threshold in thresholds:
        # Assignment prioritizes number of threshold-passing pairs over high IoUs.
        if iou.size:
            assignment_score = (iou >= threshold).astype(np.float32) * (max(iou.shape)+1) + iou
            row, col = linear_sum_assignment(-assignment_score)
            hits = iou[row, col] >= threshold
            tp = int(hits.sum())
            iou_sum = float(iou[row[hits], col[hits]].sum())
        else:
            tp, iou_sum = 0, 0.
        fp = len(pi) - tp
        fn = len(gi) - tp
        f1 = 2*tp / max(2*tp + fp + fn, 1)
        pq = iou_sum / max(tp + 0.5*(fp+fn), 1)
        suffix = str(threshold).replace('.','')
        ans.update({f'instance_f1_{suffix}': f1, f'pq_{suffix}': pq,
                    f'tp_{suffix}': tp, f'fp_{suffix}': fp, f'fn_{suffix}': fn})
    return ans

"""Original intent: short fixed propagation, CC on ORIGINAL pixels, DBSCAN on endpoints."""
import numpy as np
from scipy.ndimage import label as connected_labels
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
# Numba is optional. A mismatched llvmlite should not prevent importing/testing
# the whole experiment suite. Use the vectorized NumPy path in that case.
try:
    from numba import njit
except (ImportError, OSError):
    njit = None


def _advect_numpy(p, flow, steps, rate):
    """Vectorized nearest-voxel advection; no Numba/llvmlite required."""
    zmax, ymax, xmax = flow.shape[1:]
    for _ in range(steps):
        # Sample ONCE at old coordinates, before updating any component.
        z = p[0].astype(np.intp)
        y = p[1].astype(np.intp)
        x = p[2].astype(np.intp)
        direction = flow[:, z, y, x]
        p += direction * rate
        np.clip(p[0], 0., zmax - 1., out=p[0])
        np.clip(p[1], 0., ymax - 1., out=p[1])
        np.clip(p[2], 0., xmax - 1., out=p[2])
    return p


def _advect_impl(p, flow, steps, rate):
    """Identical nearest-voxel sampling convention to the original implementation."""
    zmax, ymax, xmax = flow.shape[1:]
    for _ in range(steps):
        for j in range(p.shape[1]):
            z, y, x = int(p[0, j]), int(p[1, j]), int(p[2, j])
            p[0,j] = min(zmax-1., max(0., p[0,j] + flow[0,z,y,x]*rate))
            p[1,j] = min(ymax-1., max(0., p[1,j] + flow[1,z,y,x]*rate))
            p[2,j] = min(xmax-1., max(0., p[2,j] + flow[2,z,y,x]*rate))
    return p


# Compile only when the installed Numba and llvmlite are compatible.
_advect = njit(cache=True)(_advect_impl) if njit is not None else _advect_numpy


def propagate(mask, flow, steps=50, rate=0.2):
    mask = np.asarray(mask, dtype=bool)
    flow = np.asarray(flow, dtype=np.float32)
    if mask.ndim != 3 or flow.shape != (3,*mask.shape):
        raise ValueError('Need ZYX mask and 3ZYX flow')
    if steps < 0 or rate <= 0:
        raise ValueError('Propagation steps >=0 and rate >0 required')
    starts = np.stack(np.nonzero(mask)).astype(np.int32)
    ends = np.ascontiguousarray(starts, dtype=np.float32)
    if starts.shape[1] > 0 and steps:
        ends = _advect(ends, np.ascontiguousarray(flow), int(steps), np.float32(rate))
    return starts, ends


def cluster_endpoints(mask, starts, ends, eps=3., min_samples=27):
    """Assign new globally unique labels; never retain source component IDs.

    DBSCAN noise is assigned to its nearest valid endpoint cluster; if DBSCAN
    finds no clusters, preserve the whole original connected component.
    """
    if eps <= 0 or min_samples < 1: raise ValueError('Invalid DBSCAN parameters')
    shape = np.asarray(mask).shape
    out = np.zeros(shape, dtype=np.int32)
    if starts.shape[1] == 0: return out
    cc, _ = connected_labels(mask)  # 6-connectivity, as in original
    src_cc = cc[tuple(starts)]
    next_id = 1
    for src in np.unique(src_cc):
        if src == 0: continue
        indices = np.flatnonzero(src_cc == src)
        original = tuple(starts[:, indices])
        pts = ends[:, indices].T
        if len(pts) < min_samples:
            out[original] = next_id
            next_id += 1
            continue
        local = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(pts)
        clusters = np.unique(local[local >= 0])
        if clusters.size == 0:
            out[original] = next_id
            next_id += 1
            continue
        # Reassign DBSCAN noise rather than keeping previous CC labels or dropping foreground.
        good = local >= 0
        if not np.all(good):
            near = cKDTree(pts[good]).query(pts[~good], k=1)[1]
            local[~good] = local[good][near]
        labels_local = np.zeros(local.shape, dtype=np.int32)
        for cluster in clusters:
            labels_local[local == cluster] = next_id
            next_id += 1
        out[original] = labels_local
    return out


def reconstruct(image, foreground_prob, flow, steps=50, rate=0.2,
                eps=3., min_samples=27, threshold=0.5):
    mask = (np.asarray(foreground_prob) >= threshold) & (np.asarray(image) > 0)
    starts, ends = propagate(mask, flow, steps=steps, rate=rate)
    return cluster_endpoints(mask, starts, ends, eps=eps, min_samples=min_samples)

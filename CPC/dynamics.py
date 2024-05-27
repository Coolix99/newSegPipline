from scipy.ndimage import  find_objects
import torch
import numpy as np

from numba import njit
#import cv2
#import fastremap



#from . import utils, metrics, transforms

import torch
from torch import optim, nn
import torch.nn.functional as F
#from . import resnet_torch

TORCH_ENABLED = True
torch_GPU = torch.device("cuda")
torch_CPU = torch.device("cpu")


def _extend_centers_gpu_3d(neighbors, meds, isneighbor, shape, n_iter=200,
                        device=torch.device("cuda")):
    """Runs diffusion on GPU to generate flows for training images or quality control.

    Args:
        neighbors (torch.Tensor): 9 x pixels in masks.
        meds (torch.Tensor): Mask centers.
        isneighbor (torch.Tensor): Valid neighbor boolean 9 x pixels.
        shape (tuple): Shape of the tensor.
        n_iter (int, optional): Number of iterations. Defaults to 200.
        device (torch.device, optional): Device to run the computation on. Defaults to torch.device("cuda").

    Returns:
        torch.Tensor: Generated flows.

    """
    if device is None:
        device = torch.device("cuda")

    T = torch.zeros(shape, dtype=torch.double, device=device)
    for i in range(n_iter):
        T[tuple(meds.T)] += 1
        Tneigh = T[tuple(neighbors)]
        Tneigh *= isneighbor
        T[tuple(neighbors[:, 0])] = Tneigh.mean(axis=0)
    del meds, isneighbor, Tneigh

    
    grads = T[tuple(neighbors[:,1:])]
    del neighbors
    dz = grads[0] - grads[1]
    dy = grads[2] - grads[3]
    dx = grads[4] - grads[5]
    del grads
    mu_torch = np.stack(
        (dz.cpu().squeeze(0), dy.cpu().squeeze(0), dx.cpu().squeeze(0)), axis=-2)
    return mu_torch


def masks_to_flows_gpu_3d(masks, device=None):
    """Convert masks to flows using diffusion from center pixel.

    Args:
        masks (3D array): Labelled masks. 0=NO masks; 1,2,...=mask labels.

    Returns:

        mu0 (float, 4D array): Flows 

    """
    if device is None:
        device = torch.device("cuda")

    Lz0, Ly0, Lx0 = masks.shape
    #Lz, Ly, Lx = Lz0 + 2, Ly0 + 2, Lx0 + 2

    masks_padded = torch.from_numpy(masks.astype("int64")).to(device)
    masks_padded = F.pad(masks_padded, (1, 1, 1, 1, 1, 1))
    
    # get mask pixel neighbors
    z, y, x = torch.nonzero(masks_padded).T
    neighborsZ = torch.stack((z, z + 1, z - 1, z, z, z, z))
    neighborsY = torch.stack((y, y, y, y + 1, y - 1, y, y), axis=0)
    neighborsX = torch.stack((x, x, x, x, x, x + 1, x - 1), axis=0)

    neighbors = torch.stack((neighborsZ, neighborsY, neighborsX), axis=0)
    
    # get mask centers
    slices = find_objects(masks)
    centers = np.zeros((masks.max(), 3), "int")
    ext=[]
    for i, si in enumerate(slices):
        if si is not None:
            sz, sy, sx = si
            #lz, ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            zi, yi, xi = np.nonzero(masks[sz, sy, sx] == (i + 1))
            zi = zi.astype(np.int32) + 1  # add padding
            yi = yi.astype(np.int32) + 1  # add padding
            xi = xi.astype(np.int32) + 1  # add padding
            zmed = np.mean(zi)
            ymed = np.mean(yi)
            xmed = np.mean(xi)
            imin = np.argmin((zi - zmed)**2 + (xi - xmed)**2 + (yi - ymed)**2)
            zmed = zi[imin]
            ymed = yi[imin]
            xmed = xi[imin]
            centers[i, 0] = zmed + sz.start
            centers[i, 1] = ymed + sy.start
            centers[i, 2] = xmed + sx.start

            ext.append([sz.stop - sz.start + 1, sy.stop - sy.start + 1, sx.stop - sx.start + 1])

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[tuple(neighbors)]
    isneighbor = neighbor_masks == neighbor_masks[0]

    ext = np.array(ext)
    n_iter = 6 * (ext.sum(axis=1)).max()

    # run diffusion
    shape = masks_padded.shape
    mu = _extend_centers_gpu_3d(neighbors, centers, isneighbor, shape, n_iter=n_iter,
                             device=device)
    # normalize
    mu /= (1e-60 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((3, Lz0, Ly0, Lx0))
    mu0[:, z.cpu().numpy() - 1, y.cpu().numpy() - 1, x.cpu().numpy() - 1] = mu

    return mu0

@njit("(float32[:, :], float32[:, :, :, :], int32,float32)", nogil=True)
def steps3D(p, dP, niter,rate):
    """ Run dynamics of pixels to recover masks in 3D.

    Euler integration of dynamics dP for niter steps.

    Args:
        p (np.ndarray): Pixel locations [3 x Npx] 
        dP (np.ndarray): Flows [3 x Lz x Ly x Lx].
        niter (int): Number of iterations of dynamics to run.

    Returns:
        np.ndarray: Final locations of each pixel after dynamics.
    """

    shape = dP.shape[1:]  # Shape of the 3D space (Lz, Ly, Lx)
    for t in range(niter):
        for j in range(p.shape[1]):
            p0, p1, p2 = int(p[0, j]), int(p[1, j]), int(p[2, j])
            p[0, j] = min(shape[0] - 1, max(0, p[0, j] + dP[0, p0, p1, p2]*rate))
            p[1, j] = min(shape[1] - 1, max(0, p[1, j] + dP[1, p0, p1, p2]*rate))
            p[2, j] = min(shape[2] - 1, max(0, p[2, j] + dP[2, p0, p1, p2]*rate))
    return p



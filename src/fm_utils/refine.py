## Refinement
import numpy as np
from tqdm import tqdm

from fm_utils.conversion import FM_to_p2p, p2p_to_FM

def farthest_point_sampling_call(d_func, k, n_points=None, verbose=False):
    """
    Samples points using farthest point sampling, initialized randomly

    Parameters
    -------------------------
    d_func   : callable - for index i, d_func(i) is a (n_points,) array of geodesic distance to
               other points
    k        : int - number of points to sample
    n_points : Number of points. If not specified, checks d_func(0)

    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    rng = np.random.default_rng()

    if n_points is None:
        n_points = d_func(0).shape

    else:
        assert n_points > 0

    inds = [rng.integers(n_points).item(0)]
    dists = d_func(inds[0])

    iterable = range(k-1) if not verbose else tqdm.tqdm(range(k))
    for i in iterable:
        if i == k-1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid))

    # print(inds)
    return np.asarray(inds)

# def icp_refine(self, nit=10, tol=None, use_adj=False, overwrite=True, verbose=False, n_jobs=1):
#     """
#     Refines the functional map using ICP and saves the result

#     Parameters
#     -------------------
#     nit       : int - number of iterations of icp to apply
#     tol       : float - threshold of change in functional map in order to stop refinement
#                 (only applies if nit is None)
#     overwrite : bool - If True changes FM type to 'icp' so that next call of self.FM
#                 will be the icp refined FM
#     """
#     if not self.fitted:
#         raise ValueError("The Functional map must be fit before refining it")

#     self._FM_icp = pyFM.refine.mesh_icp_refine(self.FM, self.mesh1, self.mesh2, nit=nit, tol=tol,
#                                                 use_adj=use_adj, n_jobs=n_jobs, verbose=verbose)

#     if overwrite:
#         self.FM_type = 'icp'

def farthest_point_sampling_call(d_func, k, n_points=None, verbose=False):
    """
    Samples points using farthest point sampling, initialized randomly

    Parameters
    -------------------------
    d_func   : callable - for index i, d_func(i) is a (n_points,) array of geodesic distance to
               other points
    k        : int - number of points to sample
    n_points : Number of points. If not specified, checks d_func(0)

    Output
    --------------------------
    fps : (k,) array of indices of sampled points
    """
    rng = np.random.default_rng()

    if n_points is None:
        n_points = d_func(0).shape

    else:
        assert n_points > 0

    inds = [rng.integers(n_points).item(0)]
    dists = d_func(inds[0])

    iterable = range(k-1) if not verbose else tqdm(range(k))
    for i in iterable:
        if i == k-1:
            continue
        newid = np.argmax(dists)
        inds.append(newid)
        dists = np.minimum(dists, d_func(newid))

    # print(inds)
    return np.asarray(inds)

#REGION ZoomOut
def zoomout_iteration(FM_12, evects1, evects2, step=1, A2=None, n_jobs=1):
    """
    Performs an iteration of ZoomOut.

    Parameters
    --------------------
    FM_12    : (k2,k1) Functional map from evects1[:,:k1] to evects2[:,:k2]
    evects1  : (n1,k1') eigenvectors on source shape with k1' >= k1 + step.
                 Can be a subsample of the original ones on the first dimension.
    evects2  : (n2,k2') eigenvectors on target shape with k2' >= k2 + step.
                 Can be a subsample of the original ones on the first dimension.
    step     : int - step of increase of dimension.
    A2       : (n2,n2) sparse area matrix on target mesh, for vertex to vertex computation.
                 If specified, the eigenvectors can't be subsampled !

    Output
    --------------------
    FM_zo : zoomout-refined functional map
    """
    k2, k1 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p_21 = FM_to_p2p(FM_12, evects1, evects2, n_jobs=n_jobs)  # (n2,)
    # Compute the (k2+step, k1+step) FM
    FM_zo = p2p_to_FM(p2p_21, evects1[:, :new_k1], evects2[:, :new_k2], A2=A2)

    return FM_zo

def zoomout_refine(FM_12, evects1, evects2, nit=10, step=1, A2=None, subsample=None,
                   return_p2p=False, n_jobs=1, verbose=False):
    """
    Refine a functional map with ZoomOut.
    Supports subsampling for each mesh, different step size, and approximate nearest neighbor.

    Parameters
    --------------------
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    FM_12      : (K,K) Functional map from from shape 1 to shape 2
    nit        : int - number of iteration of zoomout
    step       : increase in dimension at each Zoomout Iteration
    A2         : (n2,n2) sparse area matrix on target mesh.
    subsample  : tuple or iterable of size 2. Each gives indices of vertices to sample
                 for faster optimization. If not specified, no subsampling is done.
    return_p2p : bool - if True returns the vertex to vertex map.

    Output
    --------------------
    FM_12_zo  : zoomout-refined functional map from basis 1 to 2
    p2p_21_zo : only if return_p2p is set to True - the refined pointwise map from basis 2 to basis 1
    """
    k2_0, k1_0 = FM_12.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit*step1 <= evects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {evects1.shape[1]} are provided"
    assert k2_0 + nit*step2 <= evects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {evects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_12_zo = FM_12.copy()

    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:
        if use_subsample:
            FM_12_zo = zoomout_iteration(FM_12_zo, evects1[sub1], evects2[sub2], A2=None,
                                         step=step, n_jobs=n_jobs)

        else:
            FM_12_zo = zoomout_iteration(FM_12_zo, evects1, evects2, A2=A2,
                                         step=step, n_jobs=n_jobs)

    if return_p2p:
        p2p_21_zo = FM_to_p2p(FM_12_zo, evects1, evects2, n_jobs=n_jobs)  # (n2,)
        return FM_12_zo, p2p_21_zo

    return FM_12_zo

def graph_zoomout_refine(FM_12, evects1, evects2, G1=None, G2=None, nit=10, step=1, subsample=None, return_p2p=False, n_jobs=1, verbose=False):
    """
    Refines the functional map using ZoomOut and saves the result

    Parameters
    -------------------
    FM_12      : (K,K) Functional map from from shape 1 to shape 2
    eigvects1  : (n1,k1) eigenvectors on source shape with k1 >= K + nit
    eigvects2  : (n2,k2) eigenvectors on target shape with k2 >= K + nit
    nit       : int - number of iterations to do
    step      : increase in dimension at each Zoomout Iteration
    subsample : int - number of points to subsample for ZoomOut. If None or 0, no subsampling is done.
    return_p2p : bool - if True returns the vertex to vertex map.
    overwrite : bool - If True changes FM type to 'zoomout' so that next call of self.FM
                will be the zoomout refined FM (larger than the other 2)
    """
    if subsample is None or subsample == 0 or G1 is None or G2 is None:
        sub = None
    else:
        sub1 = G1.extract_fps(subsample)
        sub2 = G2.extract_fps(subsample)
        sub = (sub1,sub2)

    _FM_zo = zoomout_refine(FM_12, evects1, evects2, nit,step=step, subsample=sub,
                   return_p2p=False, n_jobs=1, verbose=verbose)

    return _FM_zo
#ENDREGION
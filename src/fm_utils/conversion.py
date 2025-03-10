import scipy.linalg
from sklearn.neighbors import NearestNeighbors
import numpy as np


def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    """
    Query nearest neighbors.

    Parameters
    -------------------------------
    X : (n1,p) first collection
    Y : (n2,p) second collection
    k : int - number of neighbors to look for
    return_distance : whether to return the nearest neighbor distance
    n_jobs          : number of parallel jobs. Set to -1 to use all processes

    Output
    -------------------------------
    dists   : (n2,k) or (n2,) if k=1 - ONLY if return_distance is False. Nearest neighbor distance.
    matches : (n2,k) or (n2,) if k=1 - nearest neighbor
    """
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches


def p2p_to_FM(p2p_21, evects1, evects2, A2=None):
    """
    Compute a Functional Map from a vertex to vertex maps (with possible subsampling).
    Can compute with the pseudo inverse of eigenvectors (if no subsampling) or least square.

    Parameters
    ------------------------------
    p2p_21    : (n2,) vertex to vertex map from target to source.
                For each vertex on the target shape, gives the index of the corresponding vertex on mesh 1.
                Can also be presented as a (n2,n1) sparse matrix.
    eigvects1 : (n1,k1) eigenvectors on source mesh. Possibly subsampled on the first dimension.
    eigvects2 : (n2,k2) eigenvectors on target mesh. Possibly subsampled on the first dimension.
    A2        : (n2,n2) area matrix of the target mesh. If specified, the eigenvectors can't be subsampled

    Outputs
    -------------------------------
    FM_12       : (k2,k1) functional map corresponding to the p2p map given.
                  Solved with pseudo inverse if A2 is given, else using least square.
    """
    # Pulled back eigenvectors
    evects1_pb = evects1[p2p_21, :] if np.asarray(p2p_21).ndim == 1 else p2p_21 @ evects1

    if A2 is not None:
        if A2.shape[0] != evects2.shape[0]:
            raise ValueError("Can't compute exact pseudo inverse with subsampled eigenvectors")

        if A2.ndim == 1:
            return evects2.T @ (A2[:, None] * evects1_pb)  # (k2,k1)

        return evects2.T @ (A2 @ evects1_pb)  # (k2,k1)

    # Solve with least square
    return scipy.linalg.lstsq(evects2, evects1_pb)[0]  # (k2,k1)


def FM_to_p2p(FM_12, evects1, evects2, use_adj=False, n_jobs=1):
    """
    Obtain a point to point map from a functional map C.
    Compares embeddings of dirac functions on the second mesh Phi_2.T with embeddings
    of dirac functions of the first mesh Phi_1.T

    Either one can transport the first diracs with the functional map or the second ones with
    the adjoint, which leads to different results (adjoint is the mathematically correct way)

    Parameters
    --------------------------
    FM_12     : (k2,k1) functional map from mesh1 to mesh2 in reduced basis
    eigvects1 : (n1,k1') first k' eigenvectors of the first basis  (k1'>k1).
                First dimension can be subsampled.
    eigvects2 : (n2,k2') first k' eigenvectors of the second basis (k2'>k2)
                First dimension can be subsampled.
    use_adj   : use the adjoint method
    n_jobs    : number of parallel jobs. Use -1 to use all processes


    Outputs:
    --------------------------
    p2p_21     : (n2,) match vertex i on shape 2 to vertex p2p_21[i] on shape 1,
                 or equivalent result if the eigenvectors are subsampled.
    """
    k2, k1 = FM_12.shape

    assert k1 <= evects1.shape[1], \
        f'At least {k1} should be provided, here only {evects1.shape[1]} are given'
    assert k2 <= evects2.shape[1], \
        f'At least {k2} should be provided, here only {evects2.shape[1]} are given'

    if use_adj:
        emb1 = evects1[:, :k1]
        emb2 = evects2[:, :k2] @ FM_12

    else:
        emb1 = evects1[:, :k1] @ FM_12.T
        emb2 = evects2[:, :k2]

    p2p_21 = knn_query(emb1, emb2,  k=1, n_jobs=n_jobs)
    return p2p_21  # (n2,)

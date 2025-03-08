import torch
import torch.nn.functional as F


def cosine_similarity_matrix(point_cloud):
    """
    Compute the cosine similarity matrix for a given point cloud.

    Args:
    - point_cloud (Tensor): A tensor of shape (N, D) where N is the number of points and D is the dimension of each point.

    Returns:
    - Tensor: A cosine similarity matrix of shape (N, N).
    """
    # Normalize the point cloud to unit vectors
    normalized_point_cloud = F.normalize(point_cloud, p=2, dim=1)

    # Compute cosine similarity
    similarity_matrix = torch.mm(normalized_point_cloud, normalized_point_cloud.t())

    return similarity_matrix


from scipy.optimize import leastsq

def distance_function(p, N, metric='euclidean'):
    """
    Computes the distance function from a point p to the point cloud N, using a specific metric.
    
    Args:
    - p (Tensor): A tensor of shape (D,) representing a point in D-dimensional space.
    - N (Tensor): A tensor of shape (num_points, D) representing a point cloud in D-dimensional space.
    - metric (str): The metric to use for computing distances. Can be 'euclidean' or 'cosine'.
    
    Returns:
    - Tensor: A tensor of shape (num_points,) containing the distances from p to each point in N.
    """
    if metric == 'euclidean':
        return torch.norm(N - p, dim=1)
    elif metric == 'cosine':
        return 1 - F.cosine_similarity(N, p.unsqueeze(0), dim=1).squeeze()
    else:
        raise ValueError("Unknown metric. Must be 'euclidean' or 'cosine'.")
    
def inverse_distance_function(d,N,dist_f,metric='euclidean', solver='leastsq',
                              n_iter=100, lr=0.01, device='cpu'):
    """ 
    Given a distance d, compute the inverse distance function and return the point p.

    Args:
    - d (Tensor): A tensor of shape (num_points,) containing distances.
    - N (Tensor): A tensor of shape (num_points, D) representing a point cloud in D-dimensional space.
    - metric (str): The metric to use for computing distances. Can be 'euclidean' or 'cosine'.

    Returns:
    - Tensor: A tensor of shape (D,) representing the point p.
    """

    x0 = N[d.argmin()]

    # compute the maximum distance between points in N using cdist
    # max_dist = torch.cdist(N, N, p=2).max()


    def f(x):
        return d - dist_f(x, N, metric=metric)
    
    if solver == "leastsq":
        p, cov_p, info, message = leastsq(f, x0, full_output=True)[:4]

        energy_diff = (sum(info["fvec"])**2 - sum(f(x0).numpy())**2) 
        print("Energy final: ", sum(info["fvec"])**2)
        print("Energy diff: ", energy_diff)
        print("Messaeg: ", message)
        p = torch.tensor(p, dtype=d.dtype)
    elif solver == "adam":
        p = torch.tensor(x0, requires_grad=True)
        optimizer = torch.optim.Adam([p], lr=lr)
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = f(p).norm()
            loss.backward()
            optimizer.step()
        p = p.detach()
    elif solver == "sgd":
        p = torch.tensor(x0, requires_grad=True)
        optimizer = torch.optim.SGD([p], lr=lr)
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = f(p).norm()
            loss.backward()
            optimizer.step()
        p = p.detach()
    else:
        raise ValueError("Unknown solver.")
        
    return p
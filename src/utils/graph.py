import torch
import numpy as np
import plotly.graph_objects as go

def laplacian_matrix(A, normalized=False, symmetric=False):
    """
    Constructs a Laplacian matrix from an adjacency matrix.

    Args:
    adj_matrix (Tensor): The adjacency matrix, shape (num_points, num_points).
    normalized (bool): Whether to normalize the Laplacian matrix.

    Returns:
    Tensor: The Laplacian matrix, shape (num_points, num_points).
    """

    # Compute the degree matrix
    if normalized:
        D = torch.diag(1/A.sum(0))**(0.5)
    else:
        D = torch.diag(1/A.sum(0))

    # Compute the Laplacian matrix
    if symmetric:
        L=torch.eye(A.shape[0]) - (D)@A@(D)
    else:
        L = D - A
    
    return L

def adjacency_matrix(knn_indices, num_points, self_loops=False, symmetric=True):
    """
    Constructs an adjacency matrix from k-NN indices.

    Args:
    knn_indices (Tensor): The indices of k nearest neighbors for each point, shape (num_points, k).
    num_points (int): The total number of points in the dataset.

    Returns:
    Tensor: The adjacency matrix, shape (num_points, num_points).
    """

    # Create an empty adjacency matrix
    adj_matrix = torch.zeros((num_points, num_points))

    # Fill the adjacency matrix
    for i in range(len(knn_indices)):
        for j in range(len(knn_indices[i])):
            adj_matrix[i, knn_indices[i][j]] = 1
        
    # Remove self-loops
    if not self_loops:
        adj_matrix = adj_matrix - torch.diag(torch.diag(adj_matrix))
    
    # Make the adjacency matrix symmetric
    if symmetric:
        adj_matrix = torch.max(adj_matrix, adj_matrix.t())  # Ensure symmetry
    
    return adj_matrix

#knn graph functions
def knn_graph(data, k, distance_matrix=None):
    """
    Constructs a k-NN graph for the input data.

    Args:
    data (Tensor): The input data, shape (num_points, num_features).
    k (int): The number of neighbors to find.

    Returns:
    Tensor: The k-NN graph, shape (num_points, k).
    """

    # Compute pairwise distances
    # r = torch.sum(data**2, dim=1).reshape(-1, 1)
    # distances = r - 2 * torch.matmul(data, data.t()) + r.t()
    if distance_matrix is not None:
        distances =distance_matrix
        #distances.sort(0)
    else:
        distances = torch.cdist(data, data, p=2)

    # Get the k nearest neighbors; distances are in row-wise manner
    _, indices = torch.topk(distances, k=k+1, largest=False)
    
    return indices[:,1:]

#TO Check
def knn_ball(point_cloud, threshold,max_neigh=None,distance_matrix=None):
    """
    Compute the adjacency matrix for a given point cloud and distance threshold.
    Optionally return the list of neighbor indices for each vertex.

    Args:
    - point_cloud (Tensor): A tensor of shape (N, D) where N is the number of points and D is the dimension of each point.
    - threshold (float): The distance threshold for connecting points in the graph.
    - max_neigh (int): max number of neighbors allowed.
    - distance_matrix(Tensor): provide directly the distance matrix as input.

    Returns:
    - List of Lists: A list containing the neighbor indices for each vertex.
    """
    # Compute pairwise distances
    if distance_matrix is not None:
        distances =distance_matrix
        #distances.sort(0)
    else:
        distances = torch.cdist(point_cloud, point_cloud, p=2)

    # add inf to the diagonal to avoid self-loops
    distances += torch.diag(torch.Tensor([float('inf')]*distances.shape[0]))

    neighbor_list = []
    # Iterate over each row in the distance matrix to find neighbors
    for row in distances:
        neighbors = (row<=threshold).float()
        if neighbors.sum()<=1:
            neighbor_list.append([])
        else:
            neighbors = torch.nonzero(neighbors).squeeze()
            if max_neigh is not None and len(neighbors) > max_neigh:
                vals,idx=row[neighbors.long()].topk(max_neigh, largest=False)
                neighbors = neighbors[idx]
            neighbors=neighbors.tolist()
            if type(neighbors)!=list:
                neighbors=[neighbors]
            neighbor_list.append(neighbors)

    return  neighbor_list 

## graphlearning ##
import graphlearning as gl
from scipy.sparse.linalg import eigsh
from scipy.special import gamma
from tqdm import tqdm

# def build_graph(data, k, algo='knn', kernel='uniform',sigma=None):
#     if algo=='knn':
#         J,D = gl.weightmatrix.knnsearch(data,k) # J is the index of the k nearest neighbors, D is the distance to the k nearest neighbors
#         W = gl.weightmatrix.knn(None,k,knn_data=(J,D),kernel=kernel) # sparse matrix
#     else:
#         raise ValueError(f'algo {algo} not implemented')
#     G = gl.graph(W)
#     return G

def compute_laplacian(G, p, m, n, alpha, sigma, k):
    L = (2*p**(2/m)/sigma)*G.laplacian()*((n*alpha/k)**(1+2/m))/n
    return L

def MDS_embedding(G):
    H = G.distance_matrix(centered=True)

    #Need to sort eigenvalues, since H may not be positive semidef
    vals,V = eigsh(H,k=5,which='LM')
    ind = np.argsort(-vals)
    V = V[:,ind]
    vals = vals[ind]

    #Get top eigenvectors and square roots of positive parts of eigenvalues
    P = V[:,:2]
    S = np.maximum(vals[:2],0)**(1/2)

    #MDS embedding
    X = P@np.diag(S)

    return X

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

class KnnGraph:

    def __init__(self, data, k=None, p=None, m=3, alpha=None, sigma=None, similarity='euclidean', kernel='gaussian', eta=None):
        self.n = len(data)
        self.m = m
        self.eigvals=None
        self.eigvecs=None
        if k is None:
            self.k = int(self.n**(4/(m+4)))
        else:
            self.k = k
        if self.k < np.log(self.n) or self.k > self.n:
            raise ValueError(f'k should be in the range [log(n), n], got {k}')
        

        if alpha is None:
            self.alpha = np.pi**(m/2)/gamma(m/2+1)
        else:
            self.alpha = alpha

        if p is None:  #Density
            self.p = 1/(m+1)/np.pi**((m+1)/2)/gamma((m+1)/2+1)
        else:
            self.p = p

        if sigma is None:
            self.sigma = self.alpha/(m+2)
        else:
            self.sigma = sigma

        J,D = gl.weightmatrix.knnsearch(data,self.k, similarity=similarity) # J is the index of the k nearest neighbors, D is the distance to the k nearest neighbors
        W = gl.weightmatrix.knn(None,self.k,knn_data=(J,D),kernel=kernel, eta=eta) # sparse matrix
        self.G = gl.graph(W)
        self.G.W =W
        self.data = data
        self._plot_emb = None
        self.geodesic_distance = None
    
    def compute_laplacian(self, normalization='normalized', alpha=1):
        L = self.G.laplacian(normalization=normalization ,alpha=alpha)
        # L = (2*self.p**(2/self.m)/self.sigma)*L*((self.n*self.alpha/self.k)**(1+2/self.m))/self.n
        return L
    
    def eigen_decomp(self, normalization='normalized', method='exact', k=10, c=None, gamma=0, tol=0, q=1):
        return self.G.eigen_decomp(normalization=normalization, method=method, k=k, c=c, gamma=gamma, tol=tol, q=q)
    
    @property
    def edges(self):
        return np.stack(self.G.weight_matrix.nonzero(), axis=1)
    
    @property
    def plot_emb(self):
        if self._plot_emb is None:
            self._plot_emb = MDS_embedding(self.G)
        return self._plot_emb
    
    def plotly_trace(self, size=5, color='blue', **kwargs):
        x_lines = np.concatenate([ [self.plot_emb[i,0], self.plot_emb[j,0], None] for i,j in self.edges])
        y_lines = np.concatenate([ [self.plot_emb[i,1], self.plot_emb[j,1], None] for i,j in self.edges])
        
        trace = [go.Scatter(x=x_lines, y=y_lines, mode='lines', name='edges', line=dict(width=0.5 , color="grey")),
                go.Scatter(x=self.plot_emb[:,0], y=self.plot_emb[:,1], mode='markers', name='nodes', marker=dict(size=size, color=color), **kwargs)]
        
        return trace
    
    def extract_fps(self, n_points):
        d_func = lambda i: self.G.distance(i, 0,return_distance_vector=True)[1]
        return farthest_point_sampling_call(d_func, n_points, n_points=self.n, verbose=False)
    
    def distance_matrix(self, centered=False):
        if self.geodesic_distance is None:
            self.geodesic_distance = self.G.distance_matrix(centered=centered)
        return self.geodesic_distance
    
    def distance(self, source, target=None):
        if target is None:
            return self.G.distance(source, 0, return_distance_vector=True)[1]
        return self.G.distance(source, target)
    


class BallGraph:

    def __init__(self, data, k=None, p=None, m=3, alpha=None, sigma=None, similarity='euclidean', kernel='gaussian', eta=None):
        self.n = len(data)
        data=data.numpy()
        self.m = m
        if k is None:
            self.k = int(self.n**(4/(m+4)))
        else:
            self.k = k

        

        if alpha is None:
            self.alpha = np.pi**(m/2)/gamma(m/2+1)
        else:
            self.alpha = alpha

        if p is None:  #Density
            self.p = 1/(m+1)/np.pi**((m+1)/2)/gamma((m+1)/2+1)
        else:
            self.p = p

        if sigma is None:
            self.sigma = self.alpha/(m+2)
        else:
            self.sigma = sigma
        W = gl.weightmatrix.epsilon_ball(data,self.k, kernel=kernel) # J is the index of the k nearest neighbors, D is the distance to the k nearest neighbors
        #W = gl.weightmatrix.knn(None,self.k,knn_data=(J,D),kernel=kernel, eta=eta) # sparse matrix
        self.G = gl.graph(W)
        self.data = data
        self._plot_emb = None
        self.geodesic_distance = None
    
    def compute_laplacian(self, normalization='normalized', alpha=1):
        L = self.G.laplacian(normalization=normalization ,alpha=alpha)
        # L = (2*self.p**(2/self.m)/self.sigma)*L*((self.n*self.alpha/self.k)**(1+2/self.m))/self.n
        return L
    
    def eigen_decomp(self, normalization='normalized', method='exact', k=10, c=None, gamma=0, tol=0, q=1):
        return self.G.eigen_decomp(normalization=normalization, method=method, k=k, c=c, gamma=gamma, tol=tol, q=q)
    
    @property
    def edges(self):
        return np.stack(self.G.weight_matrix.nonzero(), axis=1)
    
    @property
    def plot_emb(self):
        if self._plot_emb is None:
            self._plot_emb = MDS_embedding(self.G)
        return self._plot_emb
    
    def plotly_trace(self, size=5, color='blue', **kwargs):
        x_lines = np.concatenate([ [self.plot_emb[i,0], self.plot_emb[j,0], None] for i,j in self.edges])
        y_lines = np.concatenate([ [self.plot_emb[i,1], self.plot_emb[j,1], None] for i,j in self.edges])
        
        trace = [go.Scatter(x=x_lines, y=y_lines, mode='lines', name='edges', line=dict(width=0.5 , color="grey")),
                go.Scatter(x=self.plot_emb[:,0], y=self.plot_emb[:,1], mode='markers', name='nodes', marker=dict(size=size, color=color), **kwargs)]
        
        return trace
    
    def extract_fps(self, n_points):
        d_func = lambda i: self.G.distance(i, 0,return_distance_vector=True)[1]
        return farthest_point_sampling_call(d_func, n_points, n_points=self.n, verbose=False)
    
    def distance_matrix(self, centered=False):
        if self.geodesic_distance is None:
            self.geodesic_distance = self.G.distance_matrix(centered=centered)
        return self.geodesic_distance
    
    def distance(self, source, target=None):
        if target is None:
            return self.G.distance(source, 0, return_distance_vector=True)[1]
        return self.G.distance(source, target)
    

def build_graph(data, k=None, algo='knn', similarity='euclidean', kernel='uniform', sigma=None,
                 p=None, m=3, eta=None, alpha=1, **kwargs ):
    if algo=='knn':
        G = KnnGraph(data, k, similarity=similarity, p=p, m=m, alpha=alpha, sigma=sigma, kernel=kernel, eta=eta)
    elif algo=='epsball':
        G = BallGraph(data, k, similarity=similarity, p=p, m=m, alpha=alpha, sigma=sigma, kernel=kernel, eta=eta)
    elif algo=='knnball':
        raise NotImplementedError('knnball not implemented')
    else:
        raise ValueError(f'algo {algo} not implemented')
    
    return G

#### The Shape of Data: Intrinsic Distance for Data Distributions
# import numpy as np

# from scipy.sparse import lil_matrix, diags, eye


# def np_euc_cdist(data):
#     dd = np.sum(data*data, axis=1)
#     dist = -2*np.dot(data, data.T)
#     dist += dd + dd[:, np.newaxis] 
#     np.fill_diagonal(dist, 0)
#     np.sqrt(dist, dist)
#     return dist


# def construct_graph_sparse(data, k):
#     n = len(data)
#     spmat = lil_matrix((n, n))
#     dd = np.sum(data*data, axis=1)
    
#     for i in range(n):
#         dists = dd - 2*data[i, :].dot(data.T)
#         inds = np.argpartition(dists, k+1)[:k+1]
#         inds = inds[inds!=i]
#         spmat[i, inds] = 1
            
#     return spmat.tocsr()


# def construct_graph_kgraph(data, k):
#     #
#     import pykgraph

#     n = len(data)
#     spmat = lil_matrix((n, n))
#     index = pykgraph.KGraph(data, 'euclidean')
#     index.build(reverse=0, K=2 * k + 1, L=2 * k + 50)
#     result = index.search(data, K=k + 1)[:, 1:]
#     spmat[np.repeat(np.arange(n), k, 0), result.ravel()] = 1
#     return spmat.tocsr()


# def _laplacian_sparse(A, normalized=True):
#     D = A.sum(1).A1
#     if normalized:
#         Dsqrt = diags(1/np.sqrt(D))
#         L = eye(A.shape[0]) - Dsqrt.dot(A).dot(Dsqrt)
#     else:
#         L = diags(D) - A
#     return L
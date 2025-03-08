
from utils.fmap import *
from utils.descriptors import *
from utils.metrics import distance_function, inverse_distance_function

class FM:
    def __init__(self, X: np.ndarray, Y: np.ndarray, anchors: np.ndarray, 
                 X_labels: np.ndarray = None, Y_labels: np.ndarray = None,
                 graph_algo: str = 'knn', graph_kernel: str = 'gaussian', graph_similarity: str = 'euclidean',
                 num_eigs: int = 50, descriptors: tuple = ('dist_geod', 'label'), dim_manifold: int = 3, k: int = None, 
                 n_descr: int = 10, compute_gt_map: bool = False, graphs: tuple = None, 
                 compute_measures: bool = False, reg_weights: dict = None, refine: bool = True):
        """From X to Y
        :param X: source point space, (n_samplesX, n_featuresX)
        :param Y: target point space, (n_samplesY, n_featuresY)
        :param anchors: anchor points in correspondence as index pairs (i_x, i_y), (n_anchors, 2)
        :param X_labels: labels for each point in X, (n_samplesX,)
        :param Y_labels: labels for each point in Y, (n_samplesY,)
        :param graph_algo: algorithm to build the graph, 'knn' or 'eps'
        :param graph_kernel: kernel to build the graph, 'gaussian' or 'distance'
        :param graph_similarity: similarity metric to build the graph, 'euclidean' or 'cosine'
        :param num_eigs: number of eigenpairs to compute
        :param k: number of neighbors for k-NN graph construction 
        :param descriptor: tuple of descriptors to use between {'dist_geod', 'dist_cosine', 'dist_l2', 'label', 'HKS', 'WKS', 'LM'} 
        :param compute_gt_map: compute the ground truth map assuming identity as correspondence (optional)
        :param graphs: precomputed graphs (optional)
        :param compute_measures: compute the measures of the functional map (optional)
        :param reg_weights: regularization weights for the functional map (optional)
        :param refine: refine the functional map using graph zoomout (optional)
        """
        self.X = X
        self.Y = Y
        self.anchors = anchors
        self.X_labels = X_labels
        self.Y_labels = Y_labels

        self.attr = []

        self.num_eigs = num_eigs
        self.k = k
        if graphs is not None:
            self.G1 = graphs[0]
            self.G2 = graphs[1]
            self.eigvals1, self.eigvecs1 = self.G1.eigvals, self.G1.eigvecs
            self.eigvals2, self.eigvecs2 = self.G2.eigvals, self.G2.eigvecs
        else:
            # build graph
            self.G1 = build_graph(X, algo=graph_algo, kernel=graph_kernel, similarity=graph_similarity, m=dim_manifold, k=k)
            print(f"Is connected? {self.G1.G.isconnected()}")
            self.G2 = build_graph(Y, algo=graph_algo, kernel=graph_kernel, similarity=graph_similarity, m=dim_manifold, k=k)
            print(f"Is connected? {self.G2.G.isconnected()}")

            # compute eigenvalues and eigenvectors
            self.eigvals1, self.eigvecs1 = self.G1.eigen_decomp(k=num_eigs)
            self.eigvals2, self.eigvecs2 = self.G2.eigen_decomp(k=num_eigs)

        self.descriptors = descriptors
        if reg_weights is not None:
            self.w_descr = reg_weights['w_descr']
            self.w_lap = reg_weights['w_lap']
            self.w_dcomm = reg_weights['w_dcomm']
            self.w_orient = reg_weights['w_orient']
        else:
            self.w_descr = 1e0
            self.w_lap = 1e-3 
            self.w_dcomm = 1e-1
            self.w_orient = 0
        if X_labels is None and Y_labels is None:
            # remove label descriptor
            self.descriptors = [d for d in descriptors if d != 'label']

        if compute_gt_map:
            self.C = self.eigvecs2.T @ self.eigvecs1
            if compute_measures:
                self.CA = self.C.T @ self.C
                Dx = np.diag(-self.eigvals1)
                Dy = np.diag(-self.eigvals2)
                Dx_inv = np.diag(1 / np.diag(Dx + 1e-8))
                self.CO = Dx_inv @ self.C.T @ Dy @ self.C

            self.Phi_flat = FM_to_p2p(self.C, self.eigvecs1, self.eigvecs2, use_adj=False, n_jobs=1)  # (n1,), indices of the closest point in the first shape for each point in the second shape
            # compute the transformation matrix
            if self.transformation == 'linear':
                self.T = torch.linalg.lstsq(self.X, self.Y[self.Phi_flat]).solution
        else:
            self.compute_descriptors(n_descr)
            self.fit(refine=refine)

        
    def compute_descriptors(self, n_descr=10):
        """
        Compute descriptors for each point in X and Y
        """

        self.X_desc = []
        self.Y_desc = []
        if 'dist_cosine' in self.descriptors:
            # compute the distance function
            def probe_func(p, zz, metric='cosine'):
                dist =distance_function(p, zz, metric=metric)
                return dist
            # compute the distance function
            self.X_desc.append(np.stack([ distance_function(self.X[p,:], self.X, metric='cosine') for p in self.anchors[:,0]], axis=1))
            self.Y_desc.append(np.stack([ distance_function(self.Y[p2,:], self.Y, metric='cosine') for p2 in self.anchors[:,1]], axis=1))

        if 'dist_l2' in self.descriptors:
            # compute the distance function
            def probe_func(p, zz, metric='euclidean'):
                dist = distance_function(p, zz, metric=metric)
                return np.exp(-dist**2)
            # compute the distance function
            self.X_desc.append(np.stack([ distance_function(p, self.X, metric='euclidean') for p in self.anchors[:,0]], axis=1))
            self.Y_desc.append(np.stack([ distance_function(p2, self.Y, metric='euclidean') for p2 in self.anchors[:,1]], axis=1))
        if "dist_geod" in self.descriptors:
            # compute the distance function
            def probe_func(G, i_anch):
                dist = G.distance(i_anch, i_anch, return_distance_vector=True)[1]
                return np.exp(-dist**2)
            self.X_desc.append(np.stack([ probe_func(self.G1.G, i_anch) for i_anch in self.anchors[:,0]], axis=1))
            self.Y_desc.append(np.stack([ probe_func(self.G2.G, i_anch) for i_anch in self.anchors[:,1]], axis=1))
        if "label" in self.descriptors:
            labels = np.concatenate([np.unique(self.X_labels), np.unique(self.Y_labels)])
            for label in labels:
                label_descr = self.X_labels == label
                self.X_desc.append(self.G1.compute_laplacian() @ label_descr[...,None])
                label_descr = self.Y_labels == label
                self.Y_desc.append(self.G2.compute_laplacian() @ label_descr[...,None])
        if "HKS_lm" in self.descriptors:
            self.X_desc.append(auto_HKS(self.eigvals1, self.eigvecs1, n_descr, landmarks=self.anchors[:,0], scaled=True))
            self.Y_desc.append(auto_HKS(self.eigvals2, self.eigvecs2, n_descr, landmarks=self.anchors[:,1], scaled=True))
        if "HKS" in self.descriptors:
            self.X_desc.append(auto_HKS(self.eigvals1, self.eigvecs1, n_descr, scaled=True))
            self.Y_desc.append(auto_HKS(self.eigvals2, self.eigvecs2, n_descr, scaled=True))
        if "WKS" in self.descriptors:
            self.X_desc.append(auto_WKS(self.eigvals1, self.eigvecs1, n_descr, scaled=True))
            self.Y_desc.append(auto_WKS(self.eigvals2, self.eigvecs2, n_descr, scaled=True))
        if "LM" in self.descriptors:
            self.X_desc.append(lm_HKS(self.eigvals1, self.eigvecs1, self.anchors[:,0], n_descr, scaled=True))
            self.Y_desc.append(lm_HKS(self.eigvals2, self.eigvecs2, self.anchors[:,1], n_descr, scaled=True))

    def fit(self, refine=True, nit=10):
        if len(self.X_desc)==1:
            all_X_desc = np.array(self.X_desc)
            all_Y_desc = np.array(self.Y_desc)
        else:
            #print(len(self.X_desc))
            all_X_desc=np.concatenate(self.X_desc,axis=1)
            all_Y_desc=np.concatenate(self.Y_desc,axis=1)


        print("Computing functional map with descriptors: ", self.descriptors)
        self.C = rfm(all_X_desc.squeeze(),all_Y_desc.squeeze(),
                self.eigvecs1, self.eigvecs2, self.eigvals1, self.eigvals2,
                w_descr=self.w_descr, w_lap=self.w_lap, w_dcomm=self.w_dcomm, w_orient=self.w_orient, #w_lap=1e-3, w_dcomm=1e-1,
                optinit="identity") # [k2,k1],  X -> Y
        
        if refine:
            self.C = graph_zoomout_refine(self.C[:self.num_eigs-nit,:self.num_eigs-nit], 
                            self.eigvecs1[:,:self.num_eigs+nit], self.eigvecs2[:,:self.num_eigs+nit],  self.G2, self.G1, 
                          nit=nit, step=1, subsample=0, return_p2p=False, n_jobs=1)
        # self.Phi_flat = FM_to_p2p(self.C, self.eigvecs1, self.eigvecs2,  use_adj=True, n_jobs=1) # (n1,), indices of the closest point in the first shape for each point in the second shape
        # Phi = np.zeros((n,n))
        # Phi[np.arange(n),Phi_flat] = 1 # (n2,n1), S2 -> S1
    def get_similarity(self,C=None):
        if C is not None:
            return 1- np.linalg.norm(C-np.diag(np.diag(C)) )/np.linalg.norm(C)
        else:
            if self.C is not None:
                return 1- np.linalg.norm(self.C-np.diag(np.diag(self.C)) )/np.linalg.norm(self.C)
class FM_distfunc(FM):

    def transformX(self, x, metric='euclidean', solver='adam', n_iter=1000, lr=1e-5):
        """
        Transform point from X to Y
        """
        if type(x) is not torch.Tensor:
          x=torch.tensor(x)
        if type(self.X) is not torch.Tensor:
          x=torch.tensor(self.X)
        dX = distance_function(x, self.X, metric=metric).numpy()
        dt_C = self.eigvecs2 @ self.C @ (self.eigvecs1.T @ dX)

        y = inverse_distance_function(torch.tensor(dt_C),torch.tensor(self.Y), 
                                      d_func=distance_function, metric=metric, 
                                        solver=solver, n_iter=n_iter, lr=lr)
        
        return y
    

class FM_T(FM):

    def __init__(self, X: np.ndarray, Y: np.ndarray, anchors: np.ndarray, transformation: str = 'linear', 
                 X_labels: np.ndarray = None, Y_labels: np.ndarray = None, graph_algo: str = 'knn', 
                 graph_kernel: str = 'distance', graph_similarity: str = 'euclidean', num_eigs: int = 50, 
                 descriptors: tuple = ('dist_geod', 'label'), **kwargs):
        self.transformation = transformation
        super().__init__(X, Y, anchors, X_labels, Y_labels, graph_algo, graph_kernel, graph_similarity, num_eigs, descriptors, **kwargs)

    def fit(self, refine=True, nit=10):
        super().fit(refine=refine, nit=nit)
        self.Phi_flat = FM_to_p2p(self.C, self.eigvecs1, self.eigvecs2,  use_adj=False, n_jobs=1) # (n1,), indices of the closest point in the first shape for each point in the second shape
        # compute the transformation matrix
        if self.transformation == 'linear':
            self.T = torch.linalg.lstsq(self.X, self.Y[self.Phi_flat]).solution
        else:
            raise NotImplementedError(f"{self.transformation} transformation not implemented")

    def transformX(self, x):
        """
        Transform point from X to Y
        """
        if self.transformation == 'linear':
            return x @ self.T
        else:
            raise NotImplementedError(f"{self.transformation} transformation not implemented")
        

# Coupled quasi-harmonic Bases
import torch

def off_diag(A, eigvals):
    """
    Compute the off-diagonal penalty: || A^T eigvals A - eigvals ||^2
    """
    # check if eigvals is a vector
    if len(eigvals.shape) == 1:
        eigvals = torch.diag(eigvals)
    return torch.norm(A.T @ eigvals @ A - eigvals)**2

# def coupling_term(A, B, eigvecs1, eigvecs2, landmarks, V=None):
#     """
#     Compute the coupling term: || (P eigvecs1 A - Q eigvecs2 B) V ||^2
#     """
#     if V is None:
#         V = torch.eye(A.shape[1])

#     return torch.norm((eigvecs1[landmarks[:,0],:] @ A - eigvecs2[landmarks[:,1],:] @ B) @ V)**2

def coupling_term(A, B, eigvecs1, eigvecs2, F, G, V=None):
    """
    Compute the coupling term: || (F^T eigvecs1 A - G^T eigvecs2 B) V ||^2
    """
    if V is None:
        V = torch.eye(A.shape[1])

    return torch.norm((F.T @ eigvecs1 @ A - G.T @ eigvecs2 @ B) @ V)**2

def coupled_basis(A, B, eigvecs1, eigvals1, eigvecs2,eigvals2, F, G, V=None):
    return off_diag(A, eigvals1) + off_diag(B, eigvals2) + coupling_term(A, B, eigvecs1, eigvecs2, F, G, V=V)

def id_loss(A):
    return torch.norm(A.T @ A - torch.eye(A.shape[1], device=A.device))**2

def init_matrix(n1, n2, type='rand', device='cuda'):
    if type == 'rand':
        return torch.rand(n1, n2, requires_grad=True, device=device)
    elif type == 'eye':
        return torch.eye(n1, n2, requires_grad=True, device=device)
    elif type == "rand_eye":
        A = torch.rand(n1, n2, requires_grad=True, device=device)
        u, s, v = torch.svd(A)
        A = u @ v.t()
        return torch.nn.Parameter(A)
    else:
        raise ValueError(f"Unknown type: {type}")


def optmize_coupled_basis(eigvecs1, eigvals1, eigvecs2,eigvals2, F, G, V=None, 
                          w_off_diag=1, w_coupling=1, w_id=1, device='cuda',
                          init='rand_eye',
                          k=None, nit=int(4e4), lr=1e-4, enforce_ID=False, soft_ID=True):
    """
    Optimize the coupled basis
    """
    # A = torch.rand(eigvecs1.shape[1], k if k is not None else eigvecs1.shape[1], 
    #                requires_grad=True, device=device)
    # eye init
    A = init_matrix(eigvecs1.shape[1], k if k is not None else eigvecs1.shape[1], 
                    type=init, device=device)
    B = init_matrix(eigvecs2.shape[1], k if k is not None else eigvecs2.shape[1],
                    type=init, device=device)
    optimizer = torch.optim.Adam([A, B], lr=lr)

    for i in range(nit):
        optimizer.zero_grad()
        # loss = coupled_basis(A, B,  eigvecs1, torch.diag(eigvals1), eigvecs2,
        #                       torch.diag(eigvals2), F, G, V=V)
        loss = w_off_diag * (off_diag(A, torch.diag(eigvals1)) + off_diag(B, torch.diag(eigvals2))) 
        loss += w_coupling * coupling_term(A, B, eigvecs1, eigvecs2, F, G, V=V)
        if soft_ID:
            loss += w_id * (id_loss(A) + id_loss(B))
                              
        loss.backward()
        optimizer.step()

        # Projection step to enforce A^T @ A = Id
        if enforce_ID:
            with torch.no_grad():
                A.data = F.normalize(A.data, dim=0)  # Normalize columns of A
                u, s, v = torch.svd(A)
                A.data = u @ v.t()  # Orthogonalize A

                B.data = F.normalize(B.data, dim=0)  # Normalize columns of B
                u, s, v = torch.svd(B)
                B.data = u @ v.t()  # Orthogonalize B
        
        if i % 100 == 0:
         print(f"Iteration {i}, Loss: {loss.item()}")


    return A, B    



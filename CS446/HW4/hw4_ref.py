import torch
import hw4_utils


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = load_data()
        
    [d,N] = X.shape
    c = init_c
    
    mu = []; mu.append(c)
    r = []
    
    for i in range(n_iters):
        
        r_temp = torch.zeros([N,2])
        for j in range(N):
            if torch.norm(X[:,j]-c[:,0]) < torch.norm(X[:,j]-c[:,1]):
                r_temp[j,0] = 1
            else:
                r_temp[j,1] = 1
                
        mu_temp = torch.zeros([2,2])
        for k in range(2):
            num = 0; den = 0
            for j in range(N):
                num += r_temp[j,k]*X[:,j]
                den += r_temp[j,k]

            mu_temp[:,k] = num/den
        
        if torch.norm(c-mu_temp) == 0:
            c = mu_temp
            break
        
        c = mu_temp
        mu.append(mu_temp)
        r.append(r_temp)
    return c

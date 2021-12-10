import numpy as np
import numpy.linalg as la

def calc_frob_norm(X:np.ndarray, Y:np.ndarray): 
    '''
    Calc frob norm with rounding to avoid floating point issues
    '''   
    # print("Frob Norm:", )
    return la.norm(np.round(X, 4) - np.round(Y, 4), ord='fro')


def svt(Xobs, tau=5, stop_threshold = .001, max_iter=1000, verbose=False):

    # change missing to 0 (or svd will not converge)
    Omega = np.isnan(Xobs)
    Xhat = Xobs.copy()
    Xhat[Omega] = 0

    u, s, vT = la.svd(Xhat)
    if verbose:
        print("s in first iter", s)
    iter = 0
    while iter < max_iter:
        iter +=1
        Xold = Xhat.copy()
        u, s, vT = la.svd(Xhat)
        
        # take only singular values greater than tau
        s_hat = s[s >= tau]

        # reconstruct Xhat
        k = len(s_hat)
        Xhat = u[:,:k] @ np.diag(s_hat) @ vT[:k,:]

        # fill back in observed data
        Xhat = np.where(Omega, Xhat, Xobs)

        # stopping condition
        if la.norm(Xhat - Xold, ord='fro') < stop_threshold:
            break
    if verbose:
        print("total iter:", iter)
    return Xhat



if __name__ == "__main__":
    sample_data = np.array([[4, 7, np.nan, 9, 6, np.nan, 5],
                        [1, np.nan, 6, 5, 9, np.nan, 1],
                        [7, 0, np.nan, np.nan, 4, 7, 8],
                        [np.nan, 4, 8, 7, 8, np.nan, 5],
                        [7, 2, np.nan, 1, np.nan, 3, 6]])
    
    completed_data = svt(sample_data)
    print(completed_data)
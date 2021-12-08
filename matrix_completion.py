import numpy as np
import numpy.linalg as la

def calc_frob_norm(X:np.ndarray, Y:np.ndarray): 
    '''
    Calc frob norm with rounding to avoid floating point issues
    '''   
    # print("Frob Norm:", )
    return la.norm(np.round(X, 4) - np.round(Y, 4), ord='fro')


def svt(Xobs, tau=5, stop_threshold = .001):

    # change missing to 0 (or svd will not converge)
    Omega = ~np.isnan(Xobs)
    Xobs[~Omega] = 0

    Xhat = Xobs # .copy() ?

    while True:
        Xold = Xhat
        u, s, vT = la.svd(Xhat)
        
        # take only singular values greater than tau
        s_hat = s[s >= tau]
        
        # reconstruct Xhat
        Xhat = u[:,:len(s_hat)]@np.diag(s_hat)@vT[:len(s_hat),:]

        # fill back in observed data
        Xhat = Xobs*Omega + Xhat*(1-Omega)
        
        # stopping condition
        # if la.norm(Xhat - Xold) < .001:
        if calc_frob_norm(Xhat - Xold) < stop_threshold:
            return Xhat


if __name__ == "__main__":
    sample_data = np.array([[4, 7, np.nan, 9, 6, np.nan, 5],
                        [1, np.nan, 6, 5, 9, np.nan, 1],
                        [7, 0, np.nan, np.nan, 4, 7, 8],
                        [np.nan, 4, 8, 7, 8, np.nan, 5],
                        [7, 2, np.nan, 1, np.nan, 3, 6]])
    
    completed_data = svt(sample_data)
    print(completed_data)
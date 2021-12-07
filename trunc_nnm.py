import numpy as np
import numpy.linalg as la

def create_rank_r_matrix(r: int, n: int, p: int):
    '''
    Creates a random n x p matrix of rank r.
    ''' 
    np.random.seed(0)
    A = np.floor(np.random.rand (n, r) * 10)
    B = np.floor(np.random.rand (r, p) * 10) 
    X = A @ B 
    #print("Shape of X", X.shape)
    print(f'Rank X = {la.matrix_rank(X)}') 
    return X


def remove_data(frac_obs: float, 
                X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    removes data at random

    returns a tuple of Xobs, the matrix with missing values replaced with zeros
    and Omega, a matrix of bools, where True = location of present values
    '''
    np.random.seed(0)
    Omega = np.array(np.random.rand(X.shape[0], X.shape[1]) < frac_obs)
    Xobs = Omega * X
    # Where matrix = 0: we are missing
    #print(f'Observed X = \n {Xobs}')
    return Xobs, Omega


def shrinkage(X: np.ndarray, tau: float) -> np.ndarray:
    '''
    Return shrinkage D matrix from truncated NNM 
    '''
    U, S, V_T = la.svd(X, full_matrices=False)
    # shrink S
    S_star = np.where(S - tau < 0, 0, S - tau)
    S_hat = np.diag(S_star)
    return U @ S_hat @ V_T


def ADMM(A:np.ndarray, B:np.ndarray, X:np.ndarray, params:dict, 
         Xobs:np.ndarray, Omega:np.ndarray) -> np.ndarray:
    '''
    Performs Alternating Direction Method of Multipliers minimization
    Returns a new X matrix once X converges
    X: current matrix we are improving
    Xobs: original observed matrix (with missing values as zeros)
    Omega: Matrix of locations of non-missing data
    '''
    beta = params["beta"]
    # Initialize all variables as X
    X_k, W_k, Y_k = X, X, X

    for _ in range(params["max_iter_inner"]):
        # Update X
        X_k_1 = shrinkage(W_k - ((1/beta) * Y_k), tau=(1/beta))

        # update W
        W_k_1 = X_k_1 + (1/beta) * (A.T@B + Y_k)
        W_k_1 = (1 - Omega) * W_k_1 + Xobs 

        # update Y
        Y_k_1 = Y_k + beta * (X_k_1 - W_k_1)

        if la.norm(X_k_1 - X_k, ord='fro') < params["eps_inner"]:
            break

        # Update X, Y, W for next k+1 iteration
        X_k, Y_k, W_k = X_k_1, Y_k_1, W_k_1

    return X_k_1


def truncated_NNM(rank:int, params:dict, Xobs:np.ndarray, 
                  Omega:np.ndarray) -> np.ndarray:
    '''
    Performs truncated NNM given a rank. Returns updated X matrix.
    '''
    # initialize X
    orig_X = Xobs.copy()

    for _ in range(params["max_iter_outer"]):
        
        # Take SVD of X_observed
        U, S, V_T = la.svd(orig_X , full_matrices=True)
        V = V_T.T
        # Get truncated U and V as A and B
        A = U[:, :rank].T
        B = V[:, :rank].T

        # Perform ADMM minimization
        new_X = ADMM(A, B, orig_X, params, Xobs, Omega)

        if la.norm(new_X - orig_X, ord='fro') < params["eps_outer"]:
            break
        
        # Else, update X
        orig_X = new_X
    return new_X


def calc_frob_norm(X:np.ndarray, Y:np.ndarray): 
    '''
    Calc frob norm with rounding to avoid floating point issues
    '''   
    print("Frob Norm:", la.norm(np.round(X, 2) - np.round(Y, 2), ord='fro'))


if __name__ == "__main__":
    # Run test function
    r = 2
    n = 10
    p = 20
    X = create_rank_r_matrix(r, n, p)
    
    Xobs, Omega = remove_data(frac_obs=0.9, X=X)

    parameters = {"eps_outer": 1e-4,
                  "eps_inner": 1e-4,
                  "beta": .01,
                  "max_iter_outer": 100,
                  "max_iter_inner": 100}

    new_X = truncated_NNM(rank=r, 
                          params=parameters, 
                          Xobs=Xobs,
                          Omega=Omega)
    calc_frob_norm(new_X, X)

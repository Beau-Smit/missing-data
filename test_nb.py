import simulate_data
import remove_data
import matrix_completion
import trunc_nnm
import errors

from sklearn.preprocessing import normalize
import numpy as np
import numpy.linalg as la
import pandas as pd


# simulate data
sim_df = simulate_data.sim_data(**simulate_data.sim_params)

# drop categorical feature for now
sim_df = sim_df.drop("Group", axis=1)

# Normalize data by feature (axis=0)
# norms we could use for re-scaling later
sim_norm, norms = normalize(sim_df, axis=0, return_norm=True)
pd.DataFrame(sim_norm).head()


# NNM parameters
parameters = {"eps_outer": 1e-6,
              "eps_inner": 1e-6,
              "beta": 1,
              "max_iter_outer": 1000,
              "max_iter_inner": 1000}


results_lst = []
n = 8
p = 10
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


if __name__=="__main__":

    for rank in range(p):

        # Produce r-rank data
        data = simulate_data.create_rank_r_matrix(rank, n, p)
        print("RANK = ", la.matrix_rank(data))

        # Set to only 10% missing
        sim_obs = remove_data.missing_at_random(data, perc_remove=10, rand_seed=23)
        #print("Missing X \n ", sim_obs)

        sim_recovered_svt = matrix_completion.svt(sim_obs, tau=80, stop_threshold=10)
        svt_error = la.norm(data - sim_recovered_svt, ord='fro')
        #print("Missing X \n ", sim_obs)

        #print("New X - SVT \n ", sim_recovered_svt)
        print("SVT FN = ", svt_error)
        
        sim_recovered_nnm = trunc_nnm.truncated_NNM(rank, parameters, sim_obs)
        nnm_error = la.norm(data - sim_recovered_nnm, ord='fro')
        print("TNNM FN = ", nnm_error)



        results_lst.append([rank, svt_error, nnm_error])
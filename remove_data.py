import numpy as np
import pandas as pd
import random 
import math

def missing_at_random(data, perc_remove, rand_seed):
    np.random.seed(rand_seed)

    idx_arr = np.concatenate([np.repeat(0, perc_remove * data.size), 
                            np.repeat(1, (1-perc_remove) * data.size)])
    discrep = idx_arr.size != data.size
    if discrep != 0:
        idx_arr = np.append(idx_arr, [0] * discrep)
    
    np.random.shuffle(idx_arr)

    random_missing = np.where(idx_arr.reshape(data.shape) == 1, data, np.nan)

    return random_missing


if __name__ == "__main__":
    # sample_data = np.array([[4, 7, 1, 9, 6],
    #                     [1, 7, 6, 5, 9],
    #                     [7, 0, 3, 4, 4],
    #                     [2, 4, 8, 7, 8],
    #                     [7, 2, 1, 1, 6]])

    nutrition_data = pd.read_csv("nutrition_data/clean.csv").to_numpy()
    data = nutrition_data[:,1:]

    new_data = missing_at_random(data, perc_remove=0.2, rand_seed=25)
    df = pd.DataFrame(new_data).set_index(nutrition_data[:,0])

    df.to_csv("nutrition_data/random_missing.csv")

def missing_conditional_continuous(data, target_col, reference_col, threshold, steepness = .001):
    p_missing = data[reference_col].apply(
            lambda x: 1/(1+ math.e**(-steepness*(x - threshold)))
        )
    target_col_mask = p_missing.apply(
        lambda x: random.choices([False,True], weights=[x,1], k = 1)[0]
    )
    data.loc[:,target_col] = data[target_col][target_col_mask]
    return data

def missing_conditional_discrete(data, target_col, reference_col, reference_likelihoods):
    target_col_mask = data[reference_col].apply(
        lambda x: random.choices([False,True], weights=[reference_likelihoods[x],1], k = 1)[0]
    )
    data.loc[:,target_col] = data[target_col][target_col_mask]
    return data
import numpy as np
import pandas as pd

def missing_at_random(data, perc_remove, rand_seed):
    np.random.seed(rand_seed)

    idx_arr = np.concatenate([np.repeat(0, perc_remove * data.size), 
                            np.repeat(1, (1-perc_remove) * data.size)])
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

import numpy as np
import pandas as pd

def missing_at_random(data, perc_remove, rand_seed):
    np.random.seed(rand_seed)

    idx_arr = np.concatenate([np.repeat(0, perc_remove * data.size), 
                            np.repeat(1, (1-perc_remove) * data.size)])
    np.random.shuffle(idx_arr)

    random_missing = np.where(idx_arr.reshape(data.shape) == 1, sample_data, np.nan)

    return random_missing


if __name__ == "__main__":
    # sample_data = np.array([[4, 7, 1, 9, 6],
    #                     [1, 7, 6, 5, 9],
    #                     [7, 0, 3, 4, 4],
    #                     [2, 4, 8, 7, 8],
    #                     [7, 2, 1, 1, 6]])

    sample_data = pd.read_csv("nutrition_data/NutritionalFacts_Fruit_Vegetables_Seafood.csv")

    print(sample_data)
    new_data = missing_at_random(sample_data, perc_remove=0.6, rand_seed=25)
    print(new_data)

import numpy as np
import pandas as pd
import random 
import math

def missing_at_random(data, perc_remove, rand_seed):
    np.random.seed(rand_seed)

    perc_remove = perc_remove / 100
    idx_arr = np.concatenate([np.repeat(0, perc_remove * data.size), 
                            np.repeat(1, (1-perc_remove) * data.size)])
    discrep = idx_arr.size != data.size
    if discrep != 0:
        idx_arr = np.append(idx_arr, [0] * discrep)
    
    np.random.shuffle(idx_arr)

    random_missing = np.where(idx_arr.reshape(data.shape) == 1, data, np.nan)

    return random_missing


def missing_conditional_continuous(data_original, target_col, reference_col, percent_missing, high_missing=False):
    """Note: only use this after normalization
    """
    data = data_original.copy()

    # create probability for each entry being missing
    threshold = np.percentile(data[reference_col], percent_missing)
    p_missing = data[reference_col].apply(
            lambda x: 1/(1+ math.e**(-100*(x - threshold)))
        )
    
    random_percent_missing = 10000
    # if not within 2 percent of our missing goal, try again
    while abs(random_percent_missing - percent_missing) > 2:
        # remove entries based on probabilities
        if high_missing == True:
            target_col_mask = p_missing.apply(
                lambda x: random.choices([False,True], weights=[x, 1-x], k = 1)[0]
            )
        else:
            target_col_mask = p_missing.apply(
                lambda x: random.choices([False,True], weights=[1-x, x], k = 1)[0]
            )
        random_percent_missing = (1 - target_col_mask.mean()) * 100

    data.loc[:,target_col] = data[target_col][target_col_mask]

    return data


# def missing_conditional_discrete(data_original, target_col, reference_col, reference_likelihoods):
#     data = data_original.copy()
#     target_col_mask = data[reference_col].apply(
#         lambda x: random.choices([False,True], weights=[reference_likelihoods[x],1], k = 1)[0]
#     )
#     data.loc[:,target_col] = data[target_col][target_col_mask]
#     return data

def missing_conditional_discrete(data_original, target_col, percent_missing):
    data = data_original.copy()
    
    value_prevalence = (data['Discrete_LowLikelihood_0'].value_counts() / data.shape[0]).sort_values()

    # choose level of missingess for less frequent value
    missing_lvl_1 = np.random.random()
    
    # remove proper amount of those values
    column_mask = data[target_col].apply(
            lambda x: random.choices([False,True], weights=[missing_lvl_1, 1-missing_lvl_1], k = 1)[0] if x==value_prevalence.index[0] else np.nan)

    # choose level of missingness for more frequent value so that total percent missing is correct
    entries_removing = (column_mask.loc[column_mask.notnull()] == False).sum()
    entries_left_to_remove = int((data.shape[0] * percent_missing/100) - entries_removing)

    # choose as many indexes as needed to make the right number missing
    indices_to_remove = random.choices(column_mask.loc[column_mask.isnull()].index, k=entries_left_to_remove)

    data.loc[indices_to_remove, target_col] = np.nan
    data.loc[column_mask == False, target_col] = np.nan
    
    return data


if __name__ == "__main__":
    nutrition_data = pd.read_csv("nutrition_data/clean.csv").to_numpy()
    data = nutrition_data[:,1:]

    new_data = missing_at_random(data, perc_remove=0.2, rand_seed=25)
    df = pd.DataFrame(new_data).set_index(nutrition_data[:,0])

    df.to_csv("nutrition_data/random_missing.csv")

import numpy as np
import numpy.linalg as la
import random
import pandas as pd

sim_params =  {
            'POP_SIZE' : 100,
            'POP_RATIO' : [1/5, 1/5],
            'GROUPS' : ['A', 'B'],
            'INCOME_MEAN_SD' : {'A' : (50000.0, 15000.0), 
                                  'B' : (50000.0, 15000.0)},                   
            'AGE_RANGE' : (18,100),
            'CENSUS_TALLIED_PCT': {'A' : .8, 
                                     'B' : .6},
            'QUESTIONS' : {'Discrete' : {
                                'LowLikelihood': (1, .05),
                                'EvenLikelihood' : (1, .5),
                                'HighLikelihood' : (1, .95)},
                           'Continuous' : {
                                'LowLikelihood' : (1, .05, 2000, 1200),
                                'EvenLikelihood' : (1, .5, 2000, 1200),
                                'HighLikelihood' : (1, .95, 2000, 1200)
                                }
                            }
            }

def sim_data(POP_SIZE, POP_RATIO, GROUPS, INCOME_MEAN_SD, AGE_RANGE, CENSUS_TALLIED_PCT, QUESTIONS):

    data = pd.DataFrame(
        {'Group'  : random.choices(GROUPS, weights=POP_RATIO, k = POP_SIZE),
         'Age'    : [random.randint(AGE_RANGE[0], AGE_RANGE[1]) for _ in range(POP_SIZE)]})

    data.loc[:,'Income'] = data['Group'].apply(
        lambda x: random.gauss(mu = INCOME_MEAN_SD[x][0],
                               sigma = INCOME_MEAN_SD[x][1]))

    data.loc[:,'NChild'] = data['Age'].apply(
        lambda x: int(max(random.gauss(mu = min(x/20, 1.5), sigma= 1), 0)))
    
    data.loc[:, 'InCensus'] = data['Group'].apply(
        lambda x: random.choices([1,0], cum_weights=[CENSUS_TALLIED_PCT[x], 1])[0])

    for outcome_type, dict_ in QUESTIONS.items():

        if outcome_type == 'Discrete':
            for likelihood_type, params in dict_.items():
                n_entries, prob = params
                for idx in range(n_entries):
                    data.loc[:, outcome_type + '_' + likelihood_type + '_' + str(idx)] = \
                        np.random.choice(2, p = [1 - prob, prob], size = POP_SIZE)

        elif outcome_type == 'Continuous':
            for likelihood_type, params in dict_.items():
                n_entries, prob, mu, sigma = params
                for idx in range(n_entries):
                        nonzero = np.random.choice(2, p = [1 - prob, prob], size = POP_SIZE)
                        outcome = np.random.normal(loc = mu, scale = sigma, size = POP_SIZE)
                        data.loc[:, outcome_type + '_' + likelihood_type + '_' + str(idx)] = \
                            nonzero * outcome

    data.loc[:, 'PerfectClassifierOutcome'] = data.iloc[:,1:].sum(axis = 1)
    return data

def create_rank_r_matrix(r: int, n: int, p: int):
    '''
    Creates a random n x p matrix of rank r.
    ''' 
    np.random.seed(0)
    A = np.floor(np.random.rand (n, r) * 10)
    B = np.floor(np.random.rand (r, p) * 10) 
    X = A @ B 
    #print("Shape of X", X.shape)
    #print(f'Rank X = {la.matrix_rank(X)}') 
    return X

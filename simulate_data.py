import numpy as np
import random
import pandas as pd

sim_params =  {
            'POP_SIZE' : 5000,
            'POP_RATIO' : [1/5, 1/5],
            'GROUPS' : ['A', 'B'],
            'INCOME_MEAN_SD' : {'A' : (50000.0, 15000.0), 
                                  'B' : (50000.0, 15000.0)},                   
            'AGE_RANGE' : (18,100),
            'CENSUS_TALLIED_PCT': {'A' : .8, 
                                     'B' : .6},
            'QUESTIONS': {'Occupants.ChildrenGrandchildren' : .05,
                'Occupants.Relatives' : .05,
                'Occupants.Nonrelatives' :.05,
                'Occupants.Temporary' : .05,
                'Occupants.NoAdditional' : .05,
                'Housing.Owned_PaidOff' : .05,
                'Housing.Owned_Mortgage': .05,
                'Housing.Rented' : .05,
                'Housing.NoPayment' : .05}
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

    # https://www2.census.gov/programs-surveys/decennial/2020/technical-documentation/questionnaires-and-instructions/questionnaires/2020-informational-questionnaire.pdf
    for question, prob in QUESTIONS.items():
        data.loc[:, question] = \
            [random.choices([1,0], cum_weights=[prob, 1])[0] for _ in range(POP_SIZE)]
    
    return data

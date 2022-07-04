
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler

import supervised_baseline as util # src/supervised_baseline.py used for utils

"""
Going to test different sampling techniques and its effects on
illicit f1 on elliptic dataset with RandomForest baseline

Aim is to find out if over/under sampling is even worth it. 
"""

def random_undersample(df,output_class = 'is_ellicit'):
    """
    Undersample majority class 
    to same freq as mincority class

    """
    
    
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample()

def random_oversample(df,output_class = 'is_ellicit'):
    """
    oversample minority class
    to same size as majority class

    """
    pass 

def dynamicsample(df,output_class = 'is_ellicit'):
    """
    Apply some sort of sampling technique
    to over+under sample a dataset
    """
    pass 


if __name__ == '__main__':
    
    # use this as training data
    df = pd.read_csv('../data/normalized.csv') 
        
    # Will train models with over sampled and under sampled data
    # But test with original temporal split
    _, X_test, _, Y_test = util.temporal_test_split(df)
    
    
    # oversample df
    # df_os = oversample(df)
    
    # undersampled
    # df_us = undersample(df)
    
    

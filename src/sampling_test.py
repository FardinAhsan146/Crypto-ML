from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN 

import supervised_baseline as util # src/supervised_baseline.py used for utils

"""
Going to test different sampling techniques and its effects on
illicit f1 on elliptic dataset with RandomForest baseline

Aim is to find out if over/under sampling is even worth it. 
"""

def split_and_sample(sampler_class,df,output_col = 'is_ellicit'):
    """
    sampler_class: CLass -> Sampler class pass by name
    
    Splits into input and output and returns samples dfs
    """
    X = df.loc[:,df.columns != output_col]
    Y = df[output_col]
    
    rus = sampler_class(random_state = 42)
    X_ret,Y_ret = rus.fit_resample(X,y)
    
    return X_ret,Y_ret

if __name__ == '__main__':
    
    # use this as training data
    df = pd.read_csv('../data/normalized.csv') 
        
    # Will train models with over sampled and under sampled data
    # But test with original temporal split
    X_train_ns, X_test, Y_train_ns, Y_test = util.temporal_test_split(df)
    
    # sampler classes dict
    samplers = {'RUS':RandomUnderSampler
               ,'ROS':RandomOverSampler
               ,'SMOTEEEN':SMOTEENN }

    # Test and train splits by sampling strat
    sampler_splits = dict()
    
    for strat_name,strat_class in samplers.items():
        


































    
    


from imblearn.under_sampling import RandomUnderSampler 
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTEENN 
from imblearn.ensemble import BalancedRandomForestClassifier

import supervised_baseline as util # src/supervised_baseline.py used for utils

"""
Going to test different sampling techniques and its effects on
illicit f1 on elliptic dataset with RandomForest baseline

Aim is to find out if over/under sampling is even worth it. 
"""

def split_and_sample(sampler_class,X,y,output_col = 'is_ellicit'):
    """
    sampler_class: CLass -> Sampler class pass by name
    
    """

    rus = sampler_class(random_state = 42)
    X_ret,Y_ret = rus.fit_resample(X,y)
    
    return X_ret,Y_ret

if __name__ == '__main__':
    
        
    # Will train models with over sampled and under sampled data
    # But test with original temporal split
    X_train_ns, X_test, Y_train_ns, Y_test = load_splits()
    
    # sampler classes dict
    samplers = {'RUS':RandomUnderSampler
               ,'ROS':RandomOverSampler
               ,'SMOTEEEN':SMOTEENN }

    # Test and train splits by sampling strat
    sampler_splits = dict()
    for strat_name,strat_class in samplers.items():
        X_train_s,Y_train_s = split_and_sample(strat_class,X_train_ns,Y_train_ns)      
        sampler_splits[strat_name] = (X_train_s,Y_train_s)

    # print sizes
    print(f'Unsampled df is {df.shape[0]} rows')
    for strat_name,df_tup in sampler_splits.items():
        print(f'{strat_name} df in {df_tup[0].shape[0]} rows')
        
    #train RF on different sampling scores and save predictions 
    samplers_preds = dict()
    for name,df in sampler_splits.items(): 
        
        train_input_sampled,train_output_sampled = df 
        
        _, Y_pred = util.train(train_input_sampled
                              ,train_output_sampled
                              ,X_test)
        
        samplers_preds[name] = Y_pred
        
        
    # Print out f1-scores for different strategies
    for name,Y_pred in samplers_preds.items():
        f1 = util.evaluate(Y_test
                          ,Y_pred
                          ,f1_only = True)
        
        print(f'illicit-f1 for {name} = {f1}')
        
        
    # Test a balanced model to get baseline
    fit_model,Y_pred_bal = util.train(X_train_ns
                             ,Y_train_ns
                             ,X_test
                             ,model = BalancedRandomForestClassifier())
    
    util.evaluate(Y_test,Y_pred_bal,model = fit_model)
    
    
"""
Unsampled df is 46564 rows
RUS df in 7332 rows
ROS df in 57856 rows

SMOTEEEN df in 55342 rows
illicit-f1 for RUS = 0.7605451005840362
illicit-f1 for ROS = 0.7288378766140602
illicit-f1 for SMOTEEEN = 0.7348377997179125

Printing classification report for BalancedRandomForestClassifier 

              precision    recall  f1-score   support

         0.0       0.98      0.99      0.99     13091
         1.0       0.88      0.67      0.76       879

    accuracy                           0.97     13970
   macro avg       0.93      0.83      0.87     13970
weighted avg       0.97      0.97      0.97     13970


"""


    






























    
    


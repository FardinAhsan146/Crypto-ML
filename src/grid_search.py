import time
import supervised_baseline as util # src/supervised_baseline.py used for utils
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    
     # use this as training data
    df = pd.read_csv('../data/normalized.csv') 
        
    # Will train models with over sampled and under sampled data
    # But test with original temporal split
    X_train_ns, X_test, Y_train_ns, Y_test = util.temporal_test_split(df)
    
    # Parameter dictionary for combinations with feasibile values
    params = { 'max_depth': np.linspace(3,10,4).astype(int),
               'learning_rate': np.linspace(0.01,1,4),
               'n_estimators': np.linspace(100,1000,4).astype(int),
               'colsample_bytree': np.linspace(0.01,1,4)}
    
    
    rf = RandomForestClassifier()
    
    # THIS IS GOING TO TAKE SO LONG!!! I should benchmark it !
    
    start = time.perf_counter()
    
    clf = GridSearchCV(estimator = xgb_gs, 
                       param_grid = params,
                       scoring = 'neg_mean_squared_error', 
                       verbose = 1)
    clf.fit(X, y)
    
    end = time.perf_counter()
    
    print(f'Took {(start-end)//60} minutes \n')
    print('Best parameters: {clf.best_params_}')
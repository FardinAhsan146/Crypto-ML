import time
import supervised_baseline as util # src/supervised_baseline.py used for utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    
    # splits
    X_train, X_test, Y_train, Y_test = util.load_splits()
    
    # #Params
    n_estimators = np.linspace(200,2000,5).astype(int)
    max_depth = np.linspace(10, 110, 5).astype(int)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]

    
    # params grid
    grid = {'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}
    
    # model to search
    rf = RandomForestClassifier()
    
    # scorer
    scorer = make_scorer(f1_score)

    start = time.perf_counter()
    clf = GridSearchCV(estimator = rf, 
                        param_grid = grid,
                        scoring = scorer, 
                        verbose = 10,
                        n_jobs = 6)
    clf.fit(X_train, Y_train)
    end = time.perf_counter()
    
    print(f'Took {(start-end)//60} minutes \n')
    print(f'Best parameters: {clf.best_params_}')
    
    # """
    # Best parameters: {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
    
    # """
    # # Model
    # model = RandomForestClassifier(**{'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200})
    
    # # train
    # model = model.fit(X_train, Y_train)
    
    # # preds
    # y_preds = model.predict(X_test)
    
    
    # print(classification_report(Y_test,y_preds))
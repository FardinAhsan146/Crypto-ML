import supervised_baseline as util 
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
"""
Going to extend/modify random forest to use 
custom loss functions 

"""


if __name__ == '__main__':
        
    X_train_ns, X_test, Y_train_ns, Y_test = load_splits()
    
    fit_model,Y_pred = util.train(X_train,Y_train,X_test, model = RandomForestClassifier())
        
    util.evaluate(Y_test,Y_pred)
        

"""
# Deafult loss function
Printing classification report for RandomForestClassifier 

              precision    recall  f1-score   support

         0.0       0.98      1.00      0.99     13091
         1.0       0.99      0.65      0.79       879

    accuracy                           0.98     13970
   macro avg       0.98      0.83      0.89     13970
weighted avg       0.98      0.98      0.98     13970

#recall loss function




"""
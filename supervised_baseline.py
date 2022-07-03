import re
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

"""
WARNING: If you just downloaded the Elliptic dataset and placed it in the data directory
RUN THE data_cleaning.ipynb once before proceeding, It modifies the files. 
Everything else will be self contained after that. 

This file is basically testing the simple supervised models
in order to get a baseline idea of how the models perform with minimal preprocessing

Techinques to improve performance will be done later.
There will be some simple feature engineering applied. 
But the purpose is to run some baseline tests for now.

And building some functions that can be used later
"""

def read_data():
    """    
    Read the two csvs (classes and features)
    Edges will not be used for non graph based models
    """
    df_classes = pd.read_csv("data/elliptic_txs_classes.csv")
    df_features = pd.read_csv("data/elliptic_txs_features.csv")
    
    return df_classes,df_features
    
def preprocess_data(classes,features):
    """
    Cleans the data and preprocesses as required
    """
    # Merge two datasets
    df = pd.merge(classes, features)
    
    # Getrid of unknown columns
    df = df.loc[(df['class'] != 'unknown')] 
    
    # Turning class into is_illicit
    df['class'] = df['class'].apply(lambda row: 1 if (row == 'illicit') else 0)
    df = df.rename(columns={'class': 'is_illicit'})
    
    return df

def feature_engineer(node_classes, save = False):
    """
    Going to apply standard normalization
    """
    # Normalize df
    transformed = MinMaxScaler().fit_transform(node_classes.values)
    transformed_df = pd.DataFrame(transformed,columns = node_classes.columns, index = node_classes.index)
    
    if save:
        transformed_df.to_csv('data/normalized.csv', index = False)
    
    return transformed_df
    
def temporal_test_split(node_classes):
    """
    Create a 70% temporal test train splot
    This must be maintained throughout the experiments
    The exact same split will be used other. 
    
    """
    # Split into X and Y
    X = node_classes.loc[:, node_classes.columns != 'is_illicit']
    Y = node_classes['is_illicit']
    
    # Create 30/70 temporal split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, shuffle = False)
    
    return X_train, X_test, Y_train, Y_test

    
def train(model,train_input,train_output,test_input,predict = False):
    """
    Pass in a model object
    Use default hyperparameters
    
    Will return a trained model
    
    """
    fitted_model = model.fit(train_input,train_output)
    
    if predict:
        prediction = fitted_model.predict(test_input) 
        return fitted_model, prediction
  
    return fitted_model
    
def evaluate(model,test_output,predicted_output,plot = False):
    """
    Print out simple classification report for now
    Will add more functionality later
    
    """
    model_name = re.findall(r'(?<=\.)[a-zA-Z]+(?=\')',str(model.__class__))[0]
    print(f'Printing classification report for {model_name} \n')
    
    print(classification_report(test_output,predicted_output))
    
    print("\n")
    
    if plot:
        pass
        
        
    
if __name__ == "__main__":
    
    df_classes,df_features = read_data()

    df_processed = preprocess_data(df_classes,df_features)
    
    df = feature_engineer(df_processed, save = True)
    
    X_train, X_test, Y_train, Y_test = temporal_test_split(df)
    
    models = [RandomForestClassifier()
             ,DecisionTreeClassifier()
             ,xgb.XGBClassifier()]
    
    for model in models:
        fit_model,Y_pred = train( model
                                 ,X_train
                                 ,Y_train
                                 ,X_test
                                 ,predict = True)
        
        evaluate(fit_model
                 ,Y_test
                 ,Y_pred)
        
"""

Printing classification report for RandomForestClassifier 

              precision    recall  f1-score   support

         0.0       0.98      1.00      0.99     13091
         1.0       0.99      0.66      0.79       879

    accuracy                           0.98     13970
   macro avg       0.98      0.83      0.89     13970
weighted avg       0.98      0.98      0.98     13970



Printing classification report for DecisionTreeClassifier 

              precision    recall  f1-score   support

         0.0       0.98      0.96      0.97     13091
         1.0       0.54      0.67      0.60       879

    accuracy                           0.94     13970
   macro avg       0.76      0.82      0.78     13970
weighted avg       0.95      0.94      0.95     13970


"""
    
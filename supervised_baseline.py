import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

def feature_engineer(node_classes):
    """
    Going to apply standard normalization
    """
    # Normalize df
    transformed = MinMaxScaler().fit_transform(node_classes.values)
    return pd.DataFrame(transformed,columns = node_classes.columns)
    
def temporal_test_split(node_classes):
    """
    Create a 70% temporal test train splot
    This must be maintained throughout the experiments
    The exact same split will be used other. 
    
    """
    pass

    
def train_and_evaluate(model):
    """
    Pass in a model object
    Use default hyperparameters
    
    Will use 10fold CV later
    """
    pass
    
if __name__ == "__main__":
    
    df_classes,df_features = read_data()

    df = preprocess_data(df_classes,df_features)
    
    df = feature_engineer(df)
    
    # train_X,test_X,train_Y,test_Y = temporal_test_split(df)
    
    # models = []
    
    # for model in models:
    #     train_and_evaluate(model)
        
    
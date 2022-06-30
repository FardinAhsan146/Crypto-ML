import os
import pandas as pd
import numpy as np
from collections import namedtuple

"""
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
    raise NotImplementedError
    
def preprocess_data(nodes,classes):
    """
    Cleans the data and preprocesses as required
    """
    raise NotImplementedError
    
def feature_engineer(node_classes):
    """
    Going to apply standard normalization
    """
    raise NotImplementedError
    
def temporal_test_split(node_classes):
    """
    Create a 70% temporal test train splot
    This must be maintained throughout the experiments
    The exact same split will be used other. 
    
    """
    raise NotImplementedError
    
def train_and_evaluate(model):
    """
    Pass in a model object
    Use default parameters
    """
    raise NotImplementedError
    
if __name__ == "__main__":

    nodes_df,classes_df = read_data()

    df = preprocess_data(nodes_df,classes_df)
    
    df = feature_engineer(df)
    
    train_X,test_X,train_Y,test_Y = temporal_test_split(df)
    
    models = []
    
    for model in models:
        train_and_evaluate(model)
        
    
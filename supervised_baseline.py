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
    
def temporal_test_train(node_classes):
    """
    Create a 70% temporal test train splot
    This must be maintained throughout the experiments
    The exact same split will be used other. 
    
    """
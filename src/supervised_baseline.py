import re
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier

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
    df_classes = pd.read_csv("../data/elliptic_txs_classes.csv")
    df_features = pd.read_csv("../data/elliptic_txs_features.csv")
    
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


def feature_engineer(X_train,X_test,Y_train,Y_test, save = False):
    """
    Going to apply standard normalization
    """
    splits = []
    
    for split in (X_train,X_test):
        # Normalize splits
        transformed = MinMaxScaler().fit_transform(split.values)
        transformed_df = pd.DataFrame(transformed,columns = split.columns, index = split.index)
        splits.append(transformed_df)
        
    for split in (Y_train,Y_test):
        # Normalize splits
        transformed = MinMaxScaler().fit_transform(split.values.reshape(-1,1))
        transformed_df = pd.Series(transformed.ravel(),name = 'is_illicit', index = split.index)
        splits.append(transformed_df)
        
    if save:
        for split,split_name in zip(splits,('X_train','X_test','Y_train','Y_test')):
            split.to_csv(f'../data/normalized_{split_name}.csv', index = False)
    
    return splits
    
def load_splits():
    """
    Load feature engineered splits

    """
    splits = []
    for split in ('X_train','X_test','Y_train','Y_test'):
        df = pd.read_csv(f'../data/normalized_{split}.csv')
        splits.append(df)
        
    return splits

def train(train_input
          ,train_output
          ,test_input
          ,predict = True
          , model = RandomForestClassifier()):
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
    
def evaluate(test_output
             ,predicted_output
             ,f1_only = False
             ,model = RandomForestClassifier()):
    """
    Print out simple classification report for now
    Will add more functionality later
    
    """
    if f1_only:
        
        # compute illicit f1-score per timestamp
        report_dict = classification_report(test_output,
                                            predicted_output,
                                            output_dict = True,
                                            zero_division = 0)
        
        return report_dict['1.0']['f1-score']
    
    else:
        model_name = re.findall(r'(?<=\.)[a-zA-Z]+(?=\')',str(model.__class__))[0]
        print(f'Printing classification report for {model_name} \n')
        
        print(classification_report(test_output,predicted_output))
        
        print("\n")
    
def plot_ts_f1(model,unp_df,test_output,predicted_output):
    """
    plots illicit f1-score per timestep
    All papers on this topic have a plot like this
    
    """
    model_name = re.findall(r'(?<=\.)[a-zA-Z]+(?=\')',str(model.__class__))[0]
    
    # create a time stamp df
    df_ts = pd.concat([unp_df['time_step'],test_output], axis = 1).dropna()
    df_ts['is_illicit_pred'] = predicted_output
    
    # Seperate dfs by time steps
    ts_dict = dict()
        
    for ts in df_ts['time_step'].unique():
        ts_dict[ts] = df_ts.loc[df_ts['time_step'] == ts]
        
    # Find rwos of df per time step
    # ts_counts = {ts:df.shape[0] for ts,df in ts_dict.items()}
     
    # Pull out f1 scores
    f1_dict = dict()
    
    for ts,df in ts_dict.items():
        
        # compute illicit f1-score per timestamp
        report_dict = classification_report(df['is_illicit'],
                                            df['is_illicit_pred'],
                                            output_dict = True,
                                            zero_division = 0)
        f1score = report_dict['1.0']['f1-score']
        
        # Add to dict
        f1_dict[ts] = f1score
        
    # Make plots
    sns.lineplot(x = list(f1_dict.keys()), y = list(f1_dict.values())) # f1-score plot
    plt.title(f'TS illicit f1-score for {model_name}')
    plt.show()
    
    
if __name__ == "__main__":
    
    df_classes,df_features = read_data()

    df_processed = preprocess_data(df_classes,df_features)
    
    X_train, X_test, Y_train, Y_test = temporal_test_split(df_processed)
    
    X_train, X_test, Y_train, Y_test = feature_engineer(X_train,X_test,Y_train,Y_test, save = True)
    
    models = [RandomForestClassifier()
             ,xgb.XGBClassifier()]
    
    for model_class in models:
        fit_model,Y_pred = train(X_train
                                 ,Y_train
                                 ,X_test
                                 ,model = model_class)
        
        evaluate(Y_test
                ,Y_pred
                ,model = model_class)
        

"""
Printing classification report for RandomForestClassifier 

              precision    recall  f1-score   support

         0.0       0.97      1.00      0.99     13091
         1.0       0.99      0.59      0.74       879

    accuracy                           0.97     13970
   macro avg       0.98      0.80      0.86     13970
weighted avg       0.97      0.97      0.97     13970



Printing classification report for XGBClassifier 

              precision    recall  f1-score   support

         0.0       0.97      1.00      0.99     13091
         1.0       0.99      0.57      0.72       879

    accuracy                           0.97     13970
   macro avg       0.98      0.78      0.85     13970
weighted avg       0.97      0.97      0.97     13970
"""
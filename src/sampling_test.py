import supervised_baseline as util 

"""
Going to test different sampling techniques and its effects on
illicit f1 on elliptic dataset with RandomForest baseline
"""

def undersample(df):
    """
    Undersample majority class 
    to same freq as mincority class

    """
    pass 

def oversample(df):
    """
    oversample minority class
    to same size as majority class

    """
    pass 

def dynamicsample(df):
    """
    Apply some sort of sampling technique
    to over+under sample a dataset
    """
    pass 


if __name__ == '__main__':
    
    df = pd.read_csv('../data/normalized.csv') 
        

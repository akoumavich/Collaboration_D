import pandas as pd
import numpy as np


def feature_engineering(df):
    """
    Perform feature engineering operations on the input dataframe
    and create a new dataframe with only the features required for training the model.
    """

    # Perform calculations to derive features from the DataFrame
    # and store the results in new columns
        
    # Example: Derive a new feature by adding two existing features
    df['derived_feature'] = df['PRI_lep_pt'] + df['PRI_had_pt']
    
    
    # Perform feature engineering operations here and create 
    # a list of column names required for training the model

    new_columns = df.columns.tolist()
    
    df_new = pd.DataFrame(df,
        columns=new_columns
    )  # Create a new dataframe with only the required columns

    # Return the modified dataframe
    return df_new

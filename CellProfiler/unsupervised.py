import numpy as np
import pandas as pd

# Define a function to extract numeric columns and process them
def numeric_columns(df_cellline, df_merged):
    # Select numeric columns from the DataFrame
    df_numeric = df_cellline.select_dtypes(include=np.number)
    
    # Print the head of the numeric DataFrame and its columns
    print("Head of numeric columns:")
    print(df_numeric.head())
    print("\nColumns in df_numeric:", df_numeric.columns.tolist())

    numeric_columns = df_numeric.columns.values.tolist()
 
    # Remove the last 9 numeric columns
    numeric_columns = numeric_columns[:-9]

    # Filter feature columns based on specific criteria
    feature_columns = [fc for fc in numeric_columns if ('Metadata' not in fc) &
                       ('Number' not in fc) & ('Outlier' not in fc) &
                       ('ImageQuality' not in fc) & ('concentration' not in fc) &
                       ('Total' not in fc)]
    
    print(f'Excluded columns that are "Metadata", etc.: remaining {len(feature_columns)}')

    # Subset the DataFrame to keep only feature columns
    X = df_merged.loc[:, feature_columns]

    # Drop columns with missing values
    X.dropna(axis=1, inplace=True)
    print(f'Removed features with missing values: remaining {X.shape[1]}')

    # Exclude features with low standard deviation
    X = X.loc[:, (X.std() > 0.0001)]
    print(f'Excluded features with SD < 0.0001: remaining {X.shape[1]}')

    # Create a list of remaining varying features
    varying_features = list(X.columns)
    
    return X, varying_features


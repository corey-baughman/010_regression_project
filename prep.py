import pandas as pd
import numpy as np
import sklearn.preprocessing



def prep_train(df):
    '''
    applies preprocessing steps to train df for Zillow dataset
    
    Arguments: train
    Returns: X_train, y_train
    '''
    dummy_df = pd.get_dummies(df.fips, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['zip', 'year_built', 'fips', 'tax_amount'])
    X_train = df.drop('tax_value', axis=1)
    y_train = df.tax_value
    return X_train, y_train



def prep_val(df):
    '''
    applies preprocessing steps to val df for Zillow dataset
    
    Arguments: None
    Returns: X_val, y_val
    '''
    dummy_df = pd.get_dummies(df.fips, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['zip', 'year_built', 'fips', 'tax_amount'])
    X_val = df.drop('tax_value', axis=1)
    y_val = df.tax_value
    return X_val, y_val



def prep_test(df): 
    '''
    applies preprocessing steps to test df for Zillow dataset
    
    Arguments: None
    Returns: X_test, y_test
    '''
    dummy_df = pd.get_dummies(df.fips, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    df = df.drop(columns=['zip', 'year_built', 'fips', 'tax_amount'])
    X_test = df.drop('tax_value', axis=1)
    y_test = df.tax_value
    return X_test, y_test



def scale_zillow(X_train, X_val, X_test):
    '''
    Scales features in X_train, X_val, and X_test. 
    I have removed outliers and don't need standardization
    so using MinMaxScaler.
    
    Arguments: X_train, X_val, X_test
    Returns: X_train_scaled, X_val_scaled, X_test_scaled
    '''
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Call .fit only with the training data,
    # use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    ### Apply to train, validate, and test
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_val_scaled = pd.DataFrame(scaler.transform(X_val))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))

    return X_train_scaled, X_val_scaled, X_test_scaled

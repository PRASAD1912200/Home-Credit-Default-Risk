from sklearn.preprocessing import LabelEncoder
import pandas as pd

def encode_categorical(train, test):
    """Encode categorical variables"""
    
    categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
    
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(pd.concat([train[col], test[col]]).astype(str))
        
        train[col] = le.transform(train[col].astype(str))
        test[col] = le.transform(test[col].astype(str))
        
        le_dict[col] = le
        
    return train, test, {}

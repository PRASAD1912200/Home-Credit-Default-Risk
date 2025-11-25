import pandas as pd
import gc

def process_previous(df):
    """Process previous applications with advanced features"""
    
    print("\nProcessing previous applications...")
    
    try:
        prev = pd.read_csv('Data/previous_application.csv')
        
        # Feature engineering
        prev['APP_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
        prev['CREDIT_GOODS_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']
        prev['DOWN_PAYMENT_RATIO'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']
        prev['INTEREST_RATE'] = prev['RATE_DOWN_PAYMENT'] / (prev['CNT_PAYMENT'] + 1)
        
        # Days features
        prev['DAYS_FIRST_DRAWING_RELATIVE'] = prev['DAYS_FIRST_DRAWING'] - prev['DAYS_DECISION']
        prev['DAYS_FIRST_DUE_RELATIVE'] = prev['DAYS_FIRST_DUE'] - prev['DAYS_DECISION']
        prev['DAYS_LAST_DUE_RELATIVE'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_DECISION']
        prev['DAYS_TERMINATION_RELATIVE'] = prev['DAYS_TERMINATION'] - prev['DAYS_DECISION']
        
        # Categorical features
        prev['NAME_CONTRACT_STATUS_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
        prev['NAME_CONTRACT_STATUS_REFUSED'] = (prev['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
        
        # General aggregations
        prev_agg = prev.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
            'AMT_APPLICATION': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT': ['min', 'max', 'mean', 'sum'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean', 'sum'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'APP_CREDIT_RATIO': ['min', 'max', 'mean'],
            'CREDIT_GOODS_RATIO': ['min', 'max', 'mean'],
            'DOWN_PAYMENT_RATIO': ['min', 'max', 'mean'],
            'INTEREST_RATE': ['min', 'max', 'mean'],
            'NAME_CONTRACT_STATUS_APPROVED': 'sum',
            'NAME_CONTRACT_STATUS_REFUSED': 'sum'
        })
        
        prev_agg.columns = ['PREV_' + '_'.join(col).strip() for col in prev_agg.columns.values]
        prev_agg.reset_index(inplace=True)
        
        # Approval rate
        prev_agg['PREV_APPROVAL_RATE'] = prev_agg['PREV_NAME_CONTRACT_STATUS_APPROVED_sum'] / prev_agg['PREV_SK_ID_PREV_count']
        prev_agg['PREV_REFUSED_RATE'] = prev_agg['PREV_NAME_CONTRACT_STATUS_REFUSED_sum'] / prev_agg['PREV_SK_ID_PREV_count']
        
        # Approved applications only
        approved = prev[prev['NAME_CONTRACT_STATUS'] == 'Approved']
        if len(approved) > 0:
            approved_agg = approved.groupby('SK_ID_CURR').agg({
                'AMT_CREDIT': ['mean', 'sum'],
                'AMT_APPLICATION': ['mean', 'sum'],
                'CNT_PAYMENT': ['mean', 'sum']
            })
            approved_agg.columns = ['PREV_APPROVED_' + '_'.join(col).strip() for col in approved_agg.columns.values]
            approved_agg.reset_index(inplace=True)
            prev_agg = prev_agg.merge(approved_agg, on='SK_ID_CURR', how='left')
        
        # Refused applications only
        refused = prev[prev['NAME_CONTRACT_STATUS'] == 'Refused']
        if len(refused) > 0:
            refused_agg = refused.groupby('SK_ID_CURR').agg({
                'AMT_APPLICATION': 'count'
            })
            refused_agg.columns = ['PREV_REFUSED_COUNT']
            refused_agg.reset_index(inplace=True)
            prev_agg = prev_agg.merge(refused_agg, on='SK_ID_CURR', how='left')
        
        df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
        
        print(f"Previous application features added. New shape: {df.shape}")
        
        del prev, prev_agg
        gc.collect()
        
    except FileNotFoundError:
        print("Previous application file not found")
        
    return df


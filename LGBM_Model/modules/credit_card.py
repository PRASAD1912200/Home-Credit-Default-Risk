import pandas as pd
import gc

def process_credit_card(df):
    """Process credit card balance with spending patterns"""
    
    print("\nProcessing credit card data...")
    
    try:
        cc = pd.read_csv('Data/credit_card_balance.csv')
        
        # Utilization features
        cc['BALANCE_LIMIT_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
        cc['MIN_PAYMENT_RATIO'] = cc['AMT_PAYMENT_CURRENT'] / (cc['AMT_INST_MIN_REGULARITY'] + 1)
        cc['PAYMENT_MIN_DIFF'] = cc['AMT_PAYMENT_CURRENT'] - cc['AMT_INST_MIN_REGULARITY']
        cc['ATM_DRAWINGS_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / (cc['AMT_DRAWINGS_CURRENT'] + 1)
        cc['POS_DRAWINGS_RATIO'] = cc['AMT_DRAWINGS_POS_CURRENT'] / (cc['AMT_DRAWINGS_CURRENT'] + 1)
        
        # Aggregations
        cc_agg = cc.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
            'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
            'AMT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum'],
            'AMT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],
            'AMT_INST_MIN_REGULARITY': ['max', 'mean', 'sum'],
            'AMT_PAYMENT_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean', 'sum'],
            'AMT_RECEIVABLE_PRINCIPAL': ['max', 'mean', 'sum'],
            'AMT_RECIVABLE': ['max', 'mean', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_ATM_CURRENT': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_CURRENT': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['max', 'mean', 'sum'],
            'CNT_DRAWINGS_POS_CURRENT': ['max', 'mean', 'sum'],
            'CNT_INSTALMENT_MATURE_CUM': ['max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'mean', 'sum'],
            'BALANCE_LIMIT_RATIO': ['min', 'max', 'mean'],
            'MIN_PAYMENT_RATIO': ['min', 'max', 'mean'],
            'ATM_DRAWINGS_RATIO': ['mean'],
            'POS_DRAWINGS_RATIO': ['mean']
        })
        
        cc_agg.columns = ['CC_' + '_'.join(col).strip() for col in cc_agg.columns.values]
        cc_agg.reset_index(inplace=True)
        
        # Additional features
        cc_agg['CC_AVG_UTILIZATION'] = cc_agg['CC_AMT_BALANCE_sum'] / (cc_agg['CC_AMT_CREDIT_LIMIT_ACTUAL_sum'] + 1)
        cc_agg['CC_TOTAL_OVERDUE'] = cc_agg['CC_SK_DPD_sum'] + cc_agg['CC_SK_DPD_DEF_sum']
        
        df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
        
        print(f"Credit card features added. New shape: {df.shape}")
        
        del cc, cc_agg
        gc.collect()
        
    except FileNotFoundError:
        print("Credit card file not found")
        
    return df

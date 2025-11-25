import pandas as pd
import gc

def process_pos_cash(df):
    """Process POS cash balance"""
    
    print("\nProcessing POS cash data...")
    
    try:
        pos = pd.read_csv('Data/POS_CASH_balance.csv')
        
        # Features
        pos['REMAINING_INSTALMENTS'] = pos['CNT_INSTALMENT'] - pos['CNT_INSTALMENT_FUTURE']
        pos['INSTALMENT_COMPLETION'] = pos['REMAINING_INSTALMENTS'] / (pos['CNT_INSTALMENT'] + 1)
        
        # Aggregations
        pos_agg = pos.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
            'CNT_INSTALMENT': ['max', 'mean', 'sum'],
            'CNT_INSTALMENT_FUTURE': ['max', 'mean', 'sum'],
            'SK_DPD': ['max', 'mean', 'sum'],
            'SK_DPD_DEF': ['max', 'mean', 'sum'],
            'REMAINING_INSTALMENTS': ['max', 'mean', 'sum'],
            'INSTALMENT_COMPLETION': ['mean']
        })
        
        pos_agg.columns = ['POS_' + '_'.join(col).strip() for col in pos_agg.columns.values]
        pos_agg.reset_index(inplace=True)
        
        # Additional features
        pos_agg['POS_TOTAL_DPD'] = pos_agg['POS_SK_DPD_sum'] + pos_agg['POS_SK_DPD_DEF_sum']
        pos_agg['POS_DPD_RATIO'] = pos_agg['POS_TOTAL_DPD'] / (pos_agg['POS_SK_ID_PREV_count'] + 1)
        
        df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
        
        print(f"POS cash features added. New shape: {df.shape}")
        
        del pos, pos_agg
        gc.collect()
        
    except FileNotFoundError:
        print("POS cash file not found")
        
    return df

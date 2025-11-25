import pandas as pd
import gc

def process_installments(df):
    """Process installments with payment behavior features"""
    
    print("\nProcessing installments...")
    
    try:
        inst = pd.read_csv('Data/installments_payments.csv')
        
        # Payment differences
        inst['PAYMENT_DIFF'] = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
        inst['PAYMENT_RATIO'] = inst['AMT_PAYMENT'] / inst['AMT_INSTALMENT']
        inst['DAYS_BEFORE_DUE'] = inst['DAYS_INSTALMENT'] - inst['DAYS_ENTRY_PAYMENT']
        inst['LATE_PAYMENT'] = (inst['DAYS_BEFORE_DUE'] < 0).astype(int)
        inst['SIGNIFICANT_LATE'] = (inst['DAYS_BEFORE_DUE'] < -5).astype(int)
        inst['PAID_OVER'] = (inst['AMT_PAYMENT'] > inst['AMT_INSTALMENT']).astype(int)
        inst['PAID_UNDER'] = (inst['AMT_PAYMENT'] < inst['AMT_INSTALMENT']).astype(int)
        
        # Aggregations
        inst_agg = inst.groupby('SK_ID_CURR').agg({
            'SK_ID_PREV': 'count',
            'NUM_INSTALMENT_VERSION': ['max', 'mean'],
            'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
            'DAYS_INSTALMENT': ['min', 'max', 'mean'],
            'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
            'AMT_INSTALMENT': ['min', 'max', 'mean', 'sum'],
            'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
            'PAYMENT_DIFF': ['min', 'max', 'mean', 'sum', 'var'],
            'PAYMENT_RATIO': ['min', 'max', 'mean', 'var'],
            'DAYS_BEFORE_DUE': ['min', 'max', 'mean', 'var'],
            'LATE_PAYMENT': ['sum', 'mean'],
            'SIGNIFICANT_LATE': ['sum', 'mean'],
            'PAID_OVER': ['sum', 'mean'],
            'PAID_UNDER': ['sum', 'mean']
        })
        
        inst_agg.columns = ['INST_' + '_'.join(col).strip() for col in inst_agg.columns.values]
        inst_agg.reset_index(inplace=True)
        
        # Additional features
        inst_agg['INST_TOTAL_LATE_RATIO'] = inst_agg['INST_LATE_PAYMENT_sum'] / inst_agg['INST_SK_ID_PREV_count']
        inst_agg['INST_PAYMENT_DIFF_RATIO'] = inst_agg['INST_PAYMENT_DIFF_sum'] / (inst_agg['INST_AMT_INSTALMENT_sum'] + 1)
        
        df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
        
        print(f"Installments features added. New shape: {df.shape}")
        
        del inst, inst_agg
        gc.collect()
        
    except FileNotFoundError:
        print("Installments file not found")
        
    return df

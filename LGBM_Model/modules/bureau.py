import pandas as pd
import gc

def process_bureau(df):
    """Process bureau and bureau_balance data with advanced features"""
    
    print("\nProcessing bureau data...")
    
    try:
        bureau = pd.read_csv('Data/bureau.csv')
        
        # Create bureau features
        bureau['CREDIT_ACTIVE_BINARY'] = (bureau['CREDIT_ACTIVE'] == 'Active').astype(int)
        bureau['CREDIT_ENDDATE_BINARY'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
        
        # Process bureau_balance
        try:
            bb = pd.read_csv('Data/bureau_balance.csv')
            
            # Status features
            bb['STATUS_0'] = (bb['STATUS'] == '0').astype(int)
            bb['STATUS_1'] = (bb['STATUS'] == '1').astype(int)
            bb['STATUS_2'] = (bb['STATUS'] == '2').astype(int)
            bb['STATUS_3'] = (bb['STATUS'] == '3').astype(int)
            bb['STATUS_4'] = (bb['STATUS'] == '4').astype(int)
            bb['STATUS_5'] = (bb['STATUS'] == '5').astype(int)
            bb['STATUS_C'] = (bb['STATUS'] == 'C').astype(int)
            bb['STATUS_X'] = (bb['STATUS'] == 'X').astype(int)
            
            bb_agg = bb.groupby('SK_ID_BUREAU').agg({
                'MONTHS_BALANCE': ['min', 'max', 'mean', 'size'],
                'STATUS_0': 'sum',
                'STATUS_1': 'sum',
                'STATUS_2': 'sum',
                'STATUS_3': 'sum',
                'STATUS_4': 'sum',
                'STATUS_5': 'sum',
                'STATUS_C': 'sum',
                'STATUS_X': 'sum'
            })
            bb_agg.columns = ['BB_' + '_'.join(col).strip() for col in bb_agg.columns.values]
            bb_agg.reset_index(inplace=True)
            
            bureau = bureau.merge(bb_agg, on='SK_ID_BUREAU', how='left')
            
            del bb, bb_agg
            gc.collect()
            
        except FileNotFoundError:
            print("Bureau balance not found")
        
        # Aggregate bureau by SK_ID_CURR
        bureau_agg = bureau.groupby('SK_ID_CURR').agg({
            'SK_ID_BUREAU': 'count',
            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_LIMIT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['max', 'mean', 'sum'],
            'CREDIT_TYPE': ['nunique'],
            'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
            'AMT_ANNUITY': ['max', 'mean', 'sum'],
            'CREDIT_ACTIVE_BINARY': ['sum', 'mean'],
            'CREDIT_ENDDATE_BINARY': ['sum', 'mean']
        })
        
        bureau_agg.columns = ['BUREAU_' + '_'.join(col).strip() for col in bureau_agg.columns.values]
        bureau_agg.reset_index(inplace=True)
        
        # Additional bureau features
        bureau_agg['BUREAU_DEBT_CREDIT_RATIO'] = bureau_agg['BUREAU_AMT_CREDIT_SUM_DEBT_sum'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_sum'] + 1)
        bureau_agg['BUREAU_OVERDUE_RATIO'] = bureau_agg['BUREAU_AMT_CREDIT_SUM_OVERDUE_sum'] / (bureau_agg['BUREAU_AMT_CREDIT_SUM_sum'] + 1)
        bureau_agg['BUREAU_ACTIVE_LOANS_RATIO'] = bureau_agg['BUREAU_CREDIT_ACTIVE_BINARY_sum'] / bureau_agg['BUREAU_SK_ID_BUREAU_count']
        
        # Active credits only
        active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
        if len(active) > 0:
            active_agg = active.groupby('SK_ID_CURR').agg({
                'AMT_CREDIT_SUM': ['sum', 'mean'],
                'AMT_CREDIT_SUM_DEBT': ['sum', 'mean']
            })
            active_agg.columns = ['BUREAU_ACTIVE_' + '_'.join(col).strip() for col in active_agg.columns.values]
            active_agg.reset_index(inplace=True)
            bureau_agg = bureau_agg.merge(active_agg, on='SK_ID_CURR', how='left')
        
        # Closed credits only
        closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
        if len(closed) > 0:
            closed_agg = closed.groupby('SK_ID_CURR').agg({
                'DAYS_CREDIT': 'count',
                'CREDIT_DAY_OVERDUE': ['max', 'mean']
            })
            closed_agg.columns = ['BUREAU_CLOSED_' + '_'.join(col).strip() for col in closed_agg.columns.values]
            closed_agg.reset_index(inplace=True)
            bureau_agg = bureau_agg.merge(closed_agg, on='SK_ID_CURR', how='left')
        
        df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
        
        print(f"Bureau features added. New shape: {df.shape}")
        
        del bureau, bureau_agg
        gc.collect()
        
    except FileNotFoundError:
        print("Bureau file not found")
    
    return df

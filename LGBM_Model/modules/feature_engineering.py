import numpy as np
import pandas as pd
import gc

def advanced_feature_engineering(df):
    print("Creating advanced features...")
    
    # Fix anomalies
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    df['CODE_GENDER'].replace('XNA', np.nan, inplace=True)
    df['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
    
    # =============================================================================
    # INTERACTION FEATURES
    # =============================================================================
    
    # Credit/Income ratios
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['GOODS_PRICE_INCOME_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    # Credit term calculations
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    
    # Age features (in years)
    df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365
    df['EMPLOYMENT_YEARS'] = -df['DAYS_EMPLOYED'] / 365
    df['REGISTRATION_YEARS'] = -df['DAYS_REGISTRATION'] / 365
    df['ID_PUBLISH_YEARS'] = -df['DAYS_ID_PUBLISH'] / 365
    
    # Age ratios
    df['DAYS_EMPLOYED_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_AGE'] = df['AMT_INCOME_TOTAL'] / df['AGE_YEARS']
    df['INCOME_PER_EMPLOYED_YEAR'] = df['AMT_INCOME_TOTAL'] / (df['EMPLOYMENT_YEARS'] + 1)
    
    # Family features
    df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_PERSON'] = df['AMT_CREDIT'] / df['CNT_FAM_MEMBERS']
    df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])
    
    # =============================================================================
    # EXTERNAL SOURCES - ADVANCED COMBINATIONS
    # =============================================================================
    
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    
    # Basic stats
    df['EXT_SOURCE_MEAN'] = df[ext_sources].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_sources].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_sources].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_sources].max(axis=1)
    df['EXT_SOURCE_RANGE'] = df['EXT_SOURCE_MAX'] - df['EXT_SOURCE_MIN']
    
    # Products and interactions
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_1_2'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2']
    df['EXT_SOURCE_1_3'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_2_3'] = df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    
    # Weighted combinations
    df['EXT_WEIGHTED_1'] = df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 1 + df['EXT_SOURCE_3'] * 3
    df['EXT_WEIGHTED_2'] = df['EXT_SOURCE_1'] * 1 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 2
    
    # External sources with other features
    df['EXT_INCOME_1'] = df['EXT_SOURCE_1'] * df['AMT_INCOME_TOTAL']
    df['EXT_INCOME_2'] = df['EXT_SOURCE_2'] * df['AMT_INCOME_TOTAL']
    df['EXT_INCOME_3'] = df['EXT_SOURCE_3'] * df['AMT_INCOME_TOTAL']
    df['EXT_CREDIT_1'] = df['EXT_SOURCE_1'] * df['AMT_CREDIT']
    df['EXT_CREDIT_2'] = df['EXT_SOURCE_2'] * df['AMT_CREDIT']
    df['EXT_CREDIT_3'] = df['EXT_SOURCE_3'] * df['AMT_CREDIT']
    
    # =============================================================================
    # DOCUMENT AND FLAG FEATURES
    # =============================================================================
    
    # Document flags
    doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT' in col]
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
    df['DOCUMENT_RATIO'] = df['DOCUMENT_COUNT'] / len(doc_cols)
    
    # Enquiry flags
    enquiry_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 
                    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']
    df['TOTAL_ENQUIRIES'] = df[enquiry_cols].sum(axis=1)
    df['RECENT_ENQUIRIES'] = df['AMT_REQ_CREDIT_BUREAU_HOUR'] + df['AMT_REQ_CREDIT_BUREAU_DAY'] + df['AMT_REQ_CREDIT_BUREAU_WEEK']
    
    # Contact flags
    df['CONTACT_INFO_COUNT'] = (df[['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 
                                      'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']].sum(axis=1))
    
    # Region ratings
    df['REGION_RATING_MEAN'] = df[['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']].mean(axis=1)
    df['REGION_RATING_DIFF'] = df['REGION_RATING_CLIENT'] - df['REGION_RATING_CLIENT_W_CITY']
    
    # =============================================================================
    # BUILDING AND SOCIAL FEATURES
    # =============================================================================
    
    # Building features
    building_cols = ['APARTMENTS_AVG', 'BASEMENTAREA_AVG', 'YEARS_BEGINEXPLUATATION_AVG',
                     'YEARS_BUILD_AVG', 'COMMONAREA_AVG', 'ELEVATORS_AVG', 'ENTRANCES_AVG',
                     'FLOORSMAX_AVG', 'FLOORSMIN_AVG', 'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG',
                     'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG']
    
    for col in building_cols:
        if col in df.columns:
            df[f'{col}_MODE_DIFF'] = df[col] - df[col.replace('AVG', 'MODE')]
            df[f'{col}_MEDI_DIFF'] = df[col] - df[col.replace('AVG', 'MEDI')]
    
    # Social circle defaults
    df['SOCIAL_CIRCLE_DEFAULT_RATIO'] = (df['DEF_30_CNT_SOCIAL_CIRCLE'] + df['DEF_60_CNT_SOCIAL_CIRCLE']) / (df['OBS_30_CNT_SOCIAL_CIRCLE'] + df['OBS_60_CNT_SOCIAL_CIRCLE'] + 1)
    df['SOCIAL_CIRCLE_TOTAL_DEFAULTS'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] + df['DEF_60_CNT_SOCIAL_CIRCLE']
    df['SOCIAL_CIRCLE_TOTAL_OBS'] = df['OBS_30_CNT_SOCIAL_CIRCLE'] + df['OBS_60_CNT_SOCIAL_CIRCLE']
    
    # =============================================================================
    # POLYNOMIAL FEATURES FOR KEY VARIABLES
    # =============================================================================
    
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[f'{col}_SQUARED'] = df[col] ** 2
        df[f'{col}_CUBED'] = df[col] ** 3
        df[f'{col}_SQRT'] = np.sqrt(df[col])
    
    df['CREDIT_SQUARED'] = df['AMT_CREDIT'] ** 2
    df['INCOME_SQUARED'] = df['AMT_INCOME_TOTAL'] ** 2
    
    print(f"Advanced features created. New shape: {df.shape}")
    
    return df

# =============================================================================
# BUREAU DATA PROCESSING
# =============================================================================

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

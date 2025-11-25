from modules.data_loader import load_data
from modules.feature_engineering import advanced_feature_engineering
from modules.bureau import process_bureau
from modules.previous import process_previous
from modules.installments import process_installments
from modules.credit_card import process_credit_card
from modules.pos_cash import process_pos_cash
from modules.encoder import encode_categorical
from modules.optuna_tuner import optimize_hyperparameters
from modules.model_pipeline import train_pipeline

import pandas as pd
import numpy as np
import gc
import os

def main(optimize_params=False, n_trials=30):

    train, test = load_data()

    train_ids = train['SK_ID_CURR']
    test_ids = test['SK_ID_CURR']
    target = train['TARGET']

    print("\n===== FEATURE ENGINEERING =====")
    train = advanced_feature_engineering(train)
    test = advanced_feature_engineering(test)

    print("\n===== ADDITIONAL TABLES =====")
    train = process_bureau(train)
    test = process_bureau(test)

    train = process_previous(train)
    test = process_previous(test)

    train = process_installments(train)
    test = process_installments(test)

    train = process_credit_card(train)
    test = process_credit_card(test)

    train = process_pos_cash(train)
    test = process_pos_cash(test)

    # Drop ID + target
    train = train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    test = test.drop(['SK_ID_CURR'], axis=1)

    print("\n===== ENCODING =====")
    train, test, _ = encode_categorical(train, test)

    train, test = train.align(test, join='left', axis=1, fill_value=0)

    # Params (same)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        'learning_rate': 0.02,
        'num_leaves': 40,
        'max_depth': 8,
        'min_child_samples': 30,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'min_split_gain': 0.01
    }

    print("\n===== TRAINING MODEL =====")
    oof, test_preds, fi, fold_scores, cv_score = train_pipeline(train, test, target, params)

    # Save submission
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_preds})
    submission.to_csv('Submissions/submission_lightGBM.csv', index=False)

    # Save OOF
    oof_df = pd.DataFrame({'SK_ID_CURR': train_ids, 'TARGET': target, 'PREDICTION': oof})
    oof_df.to_csv('outputs/oof_predictions.csv', index=False)

    # Save feature importance
    fi.groupby("feature")["importance"].mean().sort_values(ascending=False).to_csv(
        "outputs/feature_importance_advanced.csv"
    )

    print("\n===== PIPELINE COMPLETED =====")
    print("CV AUC:", cv_score)

if __name__ == "__main__":
    main()

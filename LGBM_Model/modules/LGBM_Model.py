import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import joblib
import gc

def train_pipeline(train, test, target, params):
    n_folds = 5
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros(train.shape[0])
    test_preds = np.zeros(test.shape[0])
    feature_importance_df = pd.DataFrame()
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train, target)):
        print(f"\n========== Fold {fold+1}/{n_folds} ==========")

        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=15000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(500)
            ]
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += model.predict(test, num_iteration=model.best_iteration) / n_folds

        fold_importance = pd.DataFrame({
            'feature': train.columns,
            'importance': model.feature_importance(importance_type='gain'),
            'fold': fold + 1
        })
        feature_importance_df = pd.concat([feature_importance_df, fold_importance], axis=0)

        score = roc_auc_score(y_val, oof_preds[val_idx])
        fold_scores.append(score)
        print(f"Fold {fold+1} AUC: {score:.6f}")

        gc.collect()

    cv_score = roc_auc_score(target, oof_preds)
    print(f"\nFINAL CV AUC: {cv_score:.6f}")

    # ✔ Save model
    joblib.dump(model, "models/final_model.pkl")
    print("Saved model → models/final_model.pkl")

    return oof_preds, test_preds, feature_importance_df, fold_scores, cv_score

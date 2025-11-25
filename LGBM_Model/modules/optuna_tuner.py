import optuna
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def objective(trial, X, y):
    """Optuna objective function for hyperparameter tuning"""
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'random_state': 42,
        
        # Hyperparameters to tune
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
    }
    
    # 3-fold CV for faster optimization
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=5000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(0)]
        )
        
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        score = roc_auc_score(y_val, preds)
        scores.append(score)
    
    return np.mean(scores)
    

def optimize_hyperparameters(X, y, n_trials=50):
    """Run Optuna optimization"""
    
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    study = optuna.create_study(direction='maximize', study_name='lgb_optimization')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest AUC: {study.best_value:.6f}")
    print("\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params    

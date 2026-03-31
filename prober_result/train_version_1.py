"""Train a LightGBM baseline for IEEE-CIS fraud detection.

Run:
    python3 train.py
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from data_process import (
    load_and_merge_tables,
    preprocess_features,
    restrict_to_naive_features,
    time_based_validation_split,
)
from prober import log_round_metrics


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = SCRIPT_DIR / 'result'
PROBER_DIR = SCRIPT_DIR / 'prober_result'

RANDOM_STATE = 42
DEFAULT_N_ESTIMATORS = 150
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_NUM_LEAVES = 2
DEFAULT_MAX_DEPTH = 2
DEFAULT_MIN_CHILD_SAMPLES = 2048
DEFAULT_SUBSAMPLE = 0.2
DEFAULT_COLSAMPLE_BYTREE = 0.1
DEFAULT_REG_ALPHA = 15.0
DEFAULT_REG_LAMBDA = 30.0
N_JOBS = -1
VALID_TIME_QUANTILE = 0.8
NAIVE_FEATURES = [
    'TransactionAmt',
    'TransactionDT',
    'ProductCD',
    'card1',
    'card2',
    'card3',
    'card4',
    'card5',
    'card6',
    'addr1',
    'addr2',
    'dist1',
    'dist2',
    'P_emaildomain',
    'R_emaildomain',
]
LOG_EVAL_PERIOD = 500
MAX_TRAIN_ROWS = 0  # potential_improvement3 (addressed): use all available rows by default; override via CLI to debug faster.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train LightGBM baseline with naive settings.')
    parser.add_argument(
        '--round-index',
        default='0',
        help='Identifier used for saving prober outputs and train version snapshots.',
    )
    parser.add_argument('--n-estimators', type=int, default=DEFAULT_N_ESTIMATORS, help='Number of boosting rounds.')
    parser.add_argument('--learning-rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate for boosting.')
    parser.add_argument('--num-leaves', type=int, default=DEFAULT_NUM_LEAVES, help='Maximum number of leaves per tree.')
    parser.add_argument('--max-depth', type=int, default=DEFAULT_MAX_DEPTH, help='Maximum tree depth (-1 for unlimited).')
    parser.add_argument('--min-child-samples', type=int, default=DEFAULT_MIN_CHILD_SAMPLES, help='Minimum data points per leaf.')
    parser.add_argument('--subsample', type=float, default=DEFAULT_SUBSAMPLE, help='Row subsampling fraction.')
    parser.add_argument('--colsample-bytree', type=float, default=DEFAULT_COLSAMPLE_BYTREE, help='Column subsampling fraction.')
    parser.add_argument('--reg-alpha', type=float, default=DEFAULT_REG_ALPHA, help='L1 regularization term.')
    parser.add_argument('--reg-lambda', type=float, default=DEFAULT_REG_LAMBDA, help='L2 regularization term.')
    parser.add_argument(
        '--valid-time-quantile',
        type=float,
        default=VALID_TIME_QUANTILE,
        help='Quantile for the TransactionDT cutoff used when building the validation set.',
    )
    parser.add_argument(
        '--max-train-rows',
        type=int,
        default=MAX_TRAIN_ROWS,
        help='Maximum number of rows to keep in training (0 means use all available).',
    )
   
    return parser.parse_args()


def save_train_version(round_index: str) -> Path:
    PROBER_DIR.mkdir(parents=True, exist_ok=True)
    version_path = PROBER_DIR / f'train_version_{round_index}.py'
    shutil.copy2(SCRIPT_DIR / 'train.py', version_path)
    logger.info('Saved train snapshot to %s', version_path)
   
    return version_path


def truncate_training_rows(
    features: pd.DataFrame,
    target: pd.Series,
    max_rows: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows <= 0 or len(features) <= max_rows:
        return features, target

    logger.info('Truncating training data to the first %d rows for naive baseline.', max_rows)
    limited_features = features.iloc[:max_rows].copy()
    limited_target = target.iloc[:max_rows].copy()
   
    return limited_features, limited_target


def engineer_simple_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add light engineered versions of TransactionAmt for additional signal."""

    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    if 'TransactionAmt' not in train_copy.columns:
        return train_copy, test_copy

    train_amt = train_copy['TransactionAmt'].astype('float32')
    test_amt = test_copy['TransactionAmt'].astype('float32')
    train_copy['TransactionAmt_log1p'] = np.log1p(np.clip(train_amt, a_min=0.0, a_max=None))
    test_copy['TransactionAmt_log1p'] = np.log1p(np.clip(test_amt, a_min=0.0, a_max=None))

    amt_mean = float(train_amt.mean())
    amt_std = float(train_amt.std())
    denom_std = amt_std if amt_std > 1e-3 else 1.0
    train_copy['TransactionAmt_zscore'] = (train_amt - amt_mean) / denom_std
    test_copy['TransactionAmt_zscore'] = (test_amt - amt_mean) / denom_std

    for key in ('card1', 'card2', 'addr1'):
        if key not in train_copy.columns or key not in test_copy.columns:
            continue

        train_group_mean = train_copy.groupby(key)['TransactionAmt'].transform('mean')
        train_copy[f'TransactionAmt_to_{key}_mean'] = train_amt / (train_group_mean + 1e-3)

        group_mean_reference = train_copy.groupby(key)['TransactionAmt'].mean()
        fallback = float(group_mean_reference.mean()) if not group_mean_reference.empty else amt_mean
        test_group_mean = test_copy[key].map(group_mean_reference).fillna(fallback)
        test_copy[f'TransactionAmt_to_{key}_mean'] = test_amt / (test_group_mean + 1e-3)

    return train_copy, test_copy


def train(args: argparse.Namespace) -> None:
    round_index = str(args.round_index)
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency `lightgbm`. Install required packages first: '
            '`pip install lightgbm pandas scikit-learn`'
        ) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_merged, test_merged = load_and_merge_tables(DATA_DIR)
   

    y = train_merged['isFraud'].astype(int)
    test_ids = test_merged['TransactionID'].copy()

    train_features = train_merged.drop(columns=['isFraud'])
    test_features = test_merged.copy()

    logger.info('Preprocessing features ...')
    train_features, test_features = preprocess_features(train_features, test_features)
    # potential_improvement4 (addressed): add leakage-safe feature engineering, scaling, and anomaly handling beyond the bare minimum preprocessing shown here.
    split_series = (
        train_features['TransactionDT'].copy()
        if 'TransactionDT' in train_features.columns
        else None
    )
    train_features, test_features = restrict_to_naive_features(
        train_features,
        test_features,
        NAIVE_FEATURES,
        modulo=2,
    )
    # potential_improvement5: use the full engineered feature space (and avoid coarse modulo hashing) to retain discriminative transaction signals.
    train_features, test_features = engineer_simple_features(train_features, test_features)
    
    X = train_features.drop(columns=['TransactionID'], errors='ignore')
    X_test = test_features.drop(columns=['TransactionID'], errors='ignore')
    X = X.round(0).astype('float32')
    X_test = X_test.round(0).astype('float32')
    # potential_improvement6: keep high-precision continuous values or normalized variants instead of rounding everything to whole numbers.

    max_train_rows = max(0, int(args.max_train_rows))
    X, y = truncate_training_rows(X, y, max_train_rows)
    # potential_improvement7: retain more chronological data or use stratified/time-aware sampling rather than taking only the very first rows.

    split_ready = X.copy()
    if split_series is not None:
        split_ready['TransactionDT'] = split_series.loc[split_ready.index].astype('float32')
    else:
        split_ready['TransactionDT'] = 0.0

    valid_time_quantile = float(args.valid_time_quantile)
    if not 0.0 < valid_time_quantile < 1.0:
        logger.warning('Validation quantile %.3f out of bounds, using default %.3f', valid_time_quantile, VALID_TIME_QUANTILE)
        valid_time_quantile = VALID_TIME_QUANTILE
    logger.info('Using validation TransactionDT quantile %.3f for time-based split.', valid_time_quantile)

    X_train, X_valid, y_train, y_valid, time_threshold = time_based_validation_split(
        split_ready,
        y,
        valid_time_quantile,
        RANDOM_STATE,
    )
    X_train = X_train.drop(columns=['TransactionDT'], errors='ignore')
    X_valid = X_valid.drop(columns=['TransactionDT'], errors='ignore')
    X = X.drop(columns=['TransactionDT'], errors='ignore')
    X_test = X_test.drop(columns=['TransactionDT'], errors='ignore')
    # potential_improvement8: feed temporal predictors such as TransactionDT trends or recency features into the model instead of stripping them out post split.
    

    logger.info('Training naive LightGBM baseline ...')
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    # potential_improvement9: tune the LightGBM configuration (depth, learning rate schedule, class weights, feature subsampling) to better fit the data distribution.
   

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.log_evaluation(LOG_EVAL_PERIOD)],
    )
    # potential_improvement10: apply early stopping, train longer, or ensemble multiple models to stabilize validation AUC before generating final submissions.
   

    val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = float(roc_auc_score(y_valid, val_pred))
    logger.info('Validation ROC-AUC: %.6f', val_auc)

    logger.info('Generating test predictions ...')
    test_pred = model.predict_proba(X_test)[:, 1]

    sample_submission_path = DATA_DIR / 'sample_submission.csv'
    if sample_submission_path.exists():
        submission = pd.read_csv(sample_submission_path)
        if 'TransactionID' not in submission.columns:
            submission.insert(0, 'TransactionID', test_ids.values)
        submission['isFraud'] = test_pred
    else:
        submission = pd.DataFrame({'TransactionID': test_ids.values, 'isFraud': test_pred})

    submission_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'n_valid_rows': int(X_valid.shape[0]),
        'best_iteration': int(model.best_iteration_ or 0),
        'time_split_quantile': valid_time_quantile,
        'time_split_threshold': time_threshold,
        'round_index': round_index,
    }
    metrics_path = OUTPUT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
  
    log_round_metrics(PROBER_DIR, round_index, metrics)

    feature_importance = pd.DataFrame(
        {
            'feature': X.columns,
            'importance_gain': model.booster_.feature_importance(importance_type='gain'),
            'importance_split': model.booster_.feature_importance(importance_type='split'),
        }
    ).sort_values('importance_gain', ascending=False)
    feature_importance_path = OUTPUT_DIR / 'feature_importance.csv'
    feature_importance.to_csv(feature_importance_path, index=False)

    logger.info('Saved submission: %s', submission_path)
    logger.info('Saved metrics: %s', metrics_path)
    logger.info('Saved feature importance: %s', feature_importance_path)


def main() -> None:
    args = parse_args()
    round_index = str(args.round_index)
    save_train_version(round_index)
    train(args)


if __name__ == '__main__':
    main()

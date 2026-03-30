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
DEFAULT_N_ESTIMATORS = 6000
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_NUM_LEAVES = 4
DEFAULT_MAX_DEPTH = 3
DEFAULT_MIN_CHILD_SAMPLES = 512
DEFAULT_SUBSAMPLE = 0.5
DEFAULT_COLSAMPLE_BYTREE = 0.2
DEFAULT_REG_ALPHA = 5.0
DEFAULT_REG_LAMBDA = 10.0
N_JOBS = -1
VALID_TIME_QUANTILE = 0.8
NAIVE_FEATURES = ['TransactionAmt']
# potential_improvement_1: surface these defaults via a config file or Aim experiment tracker so sweeps are reproducible.
LOG_EVAL_PERIOD = 500
MAX_TRAIN_ROWS = 5000


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
    # potential_improvement_2: expose additional knobs (feature groups, truncation size, validation split) or load from a YAML/CLI profile for quicker iteration.
    return parser.parse_args()


def save_train_version(round_index: str) -> Path:
    PROBER_DIR.mkdir(parents=True, exist_ok=True)
    version_path = PROBER_DIR / f'train_version_{round_index}.py'
    shutil.copy2(SCRIPT_DIR / 'train.py', version_path)
    logger.info('Saved train snapshot to %s', version_path)
    # potential_improvement_3: capture a git commit hash or diff instead of copying the whole script to reduce clutter and ensure reproducibility.
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
    # potential_improvement_4: sample rows stratified by target or TransactionDT to avoid time leakage while still limiting volume.
    return limited_features, limited_target


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
    # potential_improvement_5: preload cached/parquet versions of the merged datasets to avoid repeated heavy CSV parsing.

    y = train_merged['isFraud'].astype(int)
    test_ids = test_merged['TransactionID'].copy()

    train_features = train_merged.drop(columns=['isFraud'])
    test_features = test_merged.copy()

    logger.info('Preprocessing features ...')
    train_features, test_features = preprocess_features(train_features, test_features)
    train_features, test_features = restrict_to_naive_features(
        train_features,
        test_features,
        NAIVE_FEATURES,
        modulo=3,
    )
    # potential_improvement_6: retain richer engineered features or configurable feature groups so model iterations can improve beyond this naive subset.

    X = train_features.drop(columns=['TransactionID'], errors='ignore')
    X_test = test_features.drop(columns=['TransactionID'], errors='ignore')

    X, y = truncate_training_rows(X, y, MAX_TRAIN_ROWS)

    X_train, X_valid, y_train, y_valid, time_threshold = time_based_validation_split(
        X,
        y,
        VALID_TIME_QUANTILE,
        RANDOM_STATE,
    )
    # potential_improvement_7: support cross-validation folds or repeated temporal splits to get more stable validation metrics.

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
    # potential_improvement_8: tune LightGBM hyperparameters (class weights, regularization, learning schedule) programmatically or via optuna to boost AUROC.

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.log_evaluation(LOG_EVAL_PERIOD)],
    )
    # potential_improvement_9: add early stopping and custom callbacks (e.g., Aim tracking) so bad runs terminate sooner and diagnostics are richer.

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
        'time_split_quantile': VALID_TIME_QUANTILE,
        'time_split_threshold': time_threshold,
        'round_index': round_index,
    }
    metrics_path = OUTPUT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    # potential_improvement_10: enrich saved metrics with additional diagnostics (loss curves, feature stats, inference timing) for deeper analysis.
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

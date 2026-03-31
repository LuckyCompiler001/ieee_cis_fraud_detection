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
DEFAULT_N_ESTIMATORS = 4000
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_NUM_LEAVES = 64
DEFAULT_MAX_DEPTH = -1
DEFAULT_MIN_CHILD_SAMPLES = 256
DEFAULT_SUBSAMPLE = 0.8
DEFAULT_COLSAMPLE_BYTREE = 0.7
DEFAULT_REG_ALPHA = 1.0
DEFAULT_REG_LAMBDA = 5.0
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
EARLY_STOPPING_ROUNDS = 200


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
    parser.add_argument(
        '--restrict-naive-features',
        action='store_true',
        help='Apply the reduced naive feature set with modulo hashing instead of using the full engineered space.',
    )
    parser.add_argument(
        '--bagging-rounds',
        type=int,
        default=1,
        help='Number of bagging (multi-seed) models to average for validation/test predictions.',
    )
    parser.add_argument(
        '--bagging-seed-step',
        type=int,
        default=13,
        help='Step added to the random seed for each additional bagging model.',
    )
    parser.add_argument(
        '--refit-full',
        action='store_true',
        help='After validation, refit a final model on train+valid using the averaged best iteration.',
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

    logger.info('Truncating training data to %d evenly spaced rows for faster experimentation.', max_rows)
    row_indices = np.linspace(0, len(features) - 1, num=max_rows, dtype=int)
    row_indices = np.unique(row_indices)
    limited_features = features.iloc[row_indices].copy()
    limited_target = target.iloc[row_indices].copy()
   
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


def engineer_temporal_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create simple temporal signals derived from TransactionDT."""

    if 'TransactionDT' not in train_frame.columns or 'TransactionDT' not in test_frame.columns:
        return train_frame, test_frame

    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    train_dt = train_copy['TransactionDT'].astype('float32')
    test_dt = test_copy['TransactionDT'].astype('float32')

    dt_min = float(train_dt.min())
    dt_range = max(float(train_dt.max() - dt_min), 1.0)
    train_copy['TransactionDT_norm'] = (train_dt - dt_min) / dt_range
    test_copy['TransactionDT_norm'] = (test_dt - dt_min) / dt_range

    train_copy['TransactionDT_log1p'] = np.log1p(np.clip(train_dt - dt_min, a_min=0.0, a_max=None))
    test_copy['TransactionDT_log1p'] = np.log1p(np.clip(test_dt - dt_min, a_min=0.0, a_max=None))

    seconds_in_day = float(24 * 60 * 60)
    train_day_frac = ((train_dt - dt_min) % seconds_in_day) / seconds_in_day
    test_day_frac = ((test_dt - dt_min) % seconds_in_day) / seconds_in_day
    train_copy['TransactionDT_sin_day'] = np.sin(2 * np.pi * train_day_frac)
    train_copy['TransactionDT_cos_day'] = np.cos(2 * np.pi * train_day_frac)
    test_copy['TransactionDT_sin_day'] = np.sin(2 * np.pi * test_day_frac)
    test_copy['TransactionDT_cos_day'] = np.cos(2 * np.pi * test_day_frac)

    train_copy['TransactionDT_rank'] = train_dt.rank(pct=True).astype('float32')
    test_copy['TransactionDT_rank'] = test_dt.rank(pct=True).astype('float32')

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
    if args.restrict_naive_features:
        train_features, test_features = restrict_to_naive_features(
            train_features,
            test_features,
            NAIVE_FEATURES,
            modulo=2,
        )
        logger.info('Restricting to predefined naive features (modulo hashing active).')
    else:
        logger.info('Using the full engineered feature space without modulo hashing.')
    # potential_improvement5 (addressed): use the full engineered feature space (and avoid coarse modulo hashing) to retain discriminative transaction signals.
    train_features, test_features = engineer_simple_features(train_features, test_features)
    train_features, test_features = engineer_temporal_features(train_features, test_features)
    
    X = train_features.drop(columns=['TransactionID', 'TransactionDT'], errors='ignore')
    X_test = test_features.drop(columns=['TransactionID', 'TransactionDT'], errors='ignore')
    X = X.astype('float32')
    X_test = X_test.astype('float32')
    # potential_improvement6 (addressed): keep high-precision continuous values or normalized variants instead of rounding everything to whole numbers.

    max_train_rows = max(0, int(args.max_train_rows))
    X, y = truncate_training_rows(X, y, max_train_rows)
    # potential_improvement7 (addressed): retain more chronological data or use stratified/time-aware sampling rather than taking only the very first rows.

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
    # potential_improvement8 (addressed): feed temporal predictors such as TransactionDT trends or recency features into the model instead of stripping them out post split.
    

    logger.info('Training naive LightGBM baseline ...')
    n_bagging_models = max(1, int(args.bagging_rounds))
    bagging_seed_step = max(1, int(args.bagging_seed_step))
    logger.info('Running %d bagging model(s) with seed step %d.', n_bagging_models, bagging_seed_step)
    # potential_improvement11 (addressed): ensemble multiple random seeds to stabilize validation and test predictions.

    feature_names = X_train.columns
    val_pred_accum = np.zeros(X_valid.shape[0], dtype=np.float64)
    test_pred_accum = np.zeros(X_test.shape[0], dtype=np.float64)
    gain_importances: list[np.ndarray] = []
    split_importances: list[np.ndarray] = []
    best_iterations: list[int] = []

    for bag_index in range(n_bagging_models):
        bag_seed = RANDOM_STATE + bag_index * bagging_seed_step
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
            class_weight='balanced',
            random_state=bag_seed,
            n_jobs=N_JOBS,
        )
        # potential_improvement9 (addressed): tune the LightGBM configuration (depth, learning rate schedule, class weights, feature subsampling) to better fit the data distribution.

        callbacks = [
            lgb.log_evaluation(LOG_EVAL_PERIOD),
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
        ]
        # potential_improvement10 (addressed): apply early stopping to halt training once the validation AUC stops improving.
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='auc',
            callbacks=callbacks,
        )
        best_iter = int(model.best_iteration_ or args.n_estimators)
        best_iterations.append(best_iter)

        val_pred_single = model.predict_proba(X_valid)[:, 1]
        test_pred_single = model.predict_proba(X_test)[:, 1]
        bag_auc = float(roc_auc_score(y_valid, val_pred_single))
        logger.info(
            'Bag %d/%d validation ROC-AUC: %.6f (best_iteration=%d)',
            bag_index + 1,
            n_bagging_models,
            bag_auc,
            best_iter,
        )
        val_pred_accum += val_pred_single
        test_pred_accum += test_pred_single
        gain_importances.append(model.booster_.feature_importance(importance_type='gain'))
        split_importances.append(model.booster_.feature_importance(importance_type='split'))

    val_pred = val_pred_accum / n_bagging_models
    test_pred = test_pred_accum / n_bagging_models
    val_auc = float(roc_auc_score(y_valid, val_pred))
    logger.info('Averaged validation ROC-AUC: %.6f', val_auc)

    refit_iterations = 0
    refit_train_rows = 0
    if args.refit_full:
        refit_iterations = int(np.mean(best_iterations)) if best_iterations else args.n_estimators
        refit_iterations = max(1, refit_iterations)
        logger.info(
            'Refitting final model on train+valid using %d boosting rounds for submission.',
            refit_iterations,
        )
        refit_model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            n_estimators=refit_iterations,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            min_child_samples=args.min_child_samples,
            subsample=args.subsample,
            subsample_freq=1,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            class_weight='balanced',
            random_state=RANDOM_STATE + n_bagging_models * bagging_seed_step + 1,
            n_jobs=N_JOBS,
        )
        full_train = pd.concat([X_train, X_valid], axis=0)
        full_target = pd.concat([y_train, y_valid], axis=0)
        refit_model.fit(full_train, full_target)
        test_pred = refit_model.predict_proba(X_test)[:, 1]
        gain_importances.append(refit_model.booster_.feature_importance(importance_type='gain'))
        split_importances.append(refit_model.booster_.feature_importance(importance_type='split'))
        refit_train_rows = int(full_train.shape[0])
        # potential_improvement12 (addressed): refit on the combined dataset using the averaged best iteration to squeeze extra performance for submissions.

    logger.info(
        'Generating test predictions using %s.',
        'refit model' if args.refit_full else 'bagging ensemble',
    )

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

    avg_best_iteration = int(np.round(np.mean(best_iterations))) if best_iterations else 0
    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'n_valid_rows': int(X_valid.shape[0]),
        'best_iteration': avg_best_iteration,
        'bagging_rounds': n_bagging_models,
        'bagging_seed_step': bagging_seed_step,
        'refit_full': bool(args.refit_full),
        'refit_iterations': int(refit_iterations),
        'refit_train_rows': int(refit_train_rows),
        'time_split_quantile': valid_time_quantile,
        'time_split_threshold': time_threshold,
        'round_index': round_index,
    }
    metrics_path = OUTPUT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
  
    log_round_metrics(PROBER_DIR, round_index, metrics)

    if gain_importances:
        avg_gain = np.mean(np.stack(gain_importances), axis=0)
        avg_split = np.mean(np.stack(split_importances), axis=0)
    else:
        avg_gain = np.zeros(len(feature_names), dtype=np.float32)
        avg_split = np.zeros(len(feature_names), dtype=np.float32)
    feature_importance = pd.DataFrame(
        {
            'feature': feature_names,
            'importance_gain': avg_gain,
            'importance_split': avg_split,
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

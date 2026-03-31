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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

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
    parser.add_argument(
        '--calibrate-probabilities',
        action='store_true',
        help='Apply Platt scaling on validation predictions before exporting test probabilities.',
    )
    parser.add_argument(
        '--optimize-threshold',
        action='store_true',
        help='Search for a validation decision threshold (F1-based) for downstream use.',
    )
    parser.add_argument(
        '--variance-threshold',
        type=float,
        default=0.0,
        help='Drop numeric features whose variance falls below this threshold (0 disables).',
    )
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.0,
        help='Drop numeric features that are too correlated (absolute Pearson correlation above this value).',
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


def engineer_frequency_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    freq_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add simple frequency encodings for selected categorical columns."""

    if freq_columns is None:
        freq_columns = ['card1', 'card2', 'addr1', 'ProductCD', 'P_emaildomain']

    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    combined = pd.concat([train_copy, test_copy], axis=0, ignore_index=True)
    for column in freq_columns:
        if column not in train_copy.columns or column not in test_copy.columns:
            continue

        counts = combined[column].value_counts(dropna=False)
        train_copy[f'{column}_freq'] = train_copy[column].map(counts).fillna(0).astype(np.float32)
        test_copy[f'{column}_freq'] = test_copy[column].map(counts).fillna(0).astype(np.float32)

    return train_copy, test_copy


def engineer_group_statistics(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add aggregated TransactionAmt statistics per grouping key."""

    if 'TransactionAmt' not in train_frame.columns:
        return train_frame, test_frame

    if group_columns is None:
        group_columns = ['card1', 'card2', 'addr1', 'P_emaildomain', 'R_emaildomain']

    train_copy = train_frame.copy()
    test_copy = test_frame.copy()
    train_amt = train_copy['TransactionAmt'].astype('float32')
    global_mean = float(train_amt.mean())
    global_std = float(train_amt.std())
    global_median = float(train_amt.median())

    for column in group_columns:
        if column not in train_copy.columns or column not in test_copy.columns:
            continue

        grouped_stats = train_copy.groupby(column)['TransactionAmt'].agg(['mean', 'std', 'median', 'count'])
        mean_map = grouped_stats['mean']
        std_map = grouped_stats['std'].fillna(0.0)
        median_map = grouped_stats['median']
        count_map = grouped_stats['count']

        train_copy[f'TransactionAmt_{column}_mean'] = (
            train_copy[column].map(mean_map).fillna(global_mean).astype('float32')
        )
        test_copy[f'TransactionAmt_{column}_mean'] = (
            test_copy[column].map(mean_map).fillna(global_mean).astype('float32')
        )

        train_copy[f'TransactionAmt_{column}_std'] = (
            train_copy[column].map(std_map).fillna(global_std).astype('float32')
        )
        test_copy[f'TransactionAmt_{column}_std'] = (
            test_copy[column].map(std_map).fillna(global_std).astype('float32')
        )

        train_copy[f'TransactionAmt_{column}_median'] = (
            train_copy[column].map(median_map).fillna(global_median).astype('float32')
        )
        test_copy[f'TransactionAmt_{column}_median'] = (
            test_copy[column].map(median_map).fillna(global_median).astype('float32')
        )

        train_copy[f'{column}_TransactionAmt_count'] = (
            train_copy[column].map(count_map).fillna(0).astype('float32')
        )
        test_copy[f'{column}_TransactionAmt_count'] = (
            test_copy[column].map(count_map).fillna(0).astype('float32')
        )

    return train_copy, test_copy


def engineer_interaction_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create lightweight interaction terms between numeric signals."""

    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    if 'TransactionAmt' in train_copy.columns:
        amt = train_copy['TransactionAmt'].astype('float32')
        amt_test = test_copy['TransactionAmt'].astype('float32')
        freq_candidates = [
            column
            for column in ('card1_freq', 'card2_freq', 'addr1_freq')
            if column in train_copy.columns and column in test_copy.columns
        ]
        for freq_column in freq_candidates:
            freq_train = train_copy[freq_column].astype('float32')
            freq_test = test_copy[freq_column].astype('float32')
            train_copy[f'TransactionAmt_x_{freq_column}'] = amt * freq_train
            test_copy[f'TransactionAmt_x_{freq_column}'] = amt_test * freq_test
            train_copy[f'TransactionAmt_div_{freq_column}'] = amt / (freq_train + 1.0)
            test_copy[f'TransactionAmt_div_{freq_column}'] = amt_test / (freq_test + 1.0)

        if 'TransactionDT_norm' in train_copy.columns and 'TransactionDT_norm' in test_copy.columns:
            dt_train = train_copy['TransactionDT_norm'].astype('float32')
            dt_test = test_copy['TransactionDT_norm'].astype('float32')
            train_copy['TransactionAmt_x_TransactionDT_norm'] = amt * dt_train
            test_copy['TransactionAmt_x_TransactionDT_norm'] = amt_test * dt_test

    if 'TransactionDT_rank' in train_copy.columns and 'card1_freq' in train_copy.columns:
        train_copy['TransactionDT_rank_x_card1_freq'] = (
            train_copy['TransactionDT_rank'].astype('float32') * train_copy['card1_freq'].astype('float32')
        )
        test_copy['TransactionDT_rank_x_card1_freq'] = (
            test_copy['TransactionDT_rank'].astype('float32') * test_copy['card1_freq'].astype('float32')
        )

    return train_copy, test_copy


def engineer_missing_indicators(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    sentinel: float = -999.0,
    max_indicator_columns: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append binary columns denoting whether the value was originally missing."""

    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    candidate_columns = [
        column
        for column in train_copy.columns
        if (train_copy[column] == sentinel).any() or (test_copy[column] == sentinel).any()
    ]
    if max_indicator_columns > 0 and len(candidate_columns) > max_indicator_columns:
        candidate_columns = candidate_columns[:max_indicator_columns]

    for column in candidate_columns:
        train_copy[f'{column}_is_missing'] = (train_copy[column] == sentinel).astype(np.int8)
        test_copy[f'{column}_is_missing'] = (test_copy[column] == sentinel).astype(np.int8)

    return train_copy, test_copy


def apply_variance_threshold(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop columns whose variance falls below the requested threshold."""

    if threshold <= 0.0:
        return train_frame, test_frame, []

    variances = train_frame.var(axis=0, numeric_only=True)
    low_variance_columns = variances[variances < threshold].index.tolist()
    if not low_variance_columns:
        return train_frame, test_frame, []

    logger.info(
        'Dropping %d low-variance feature(s) below threshold %.6f.',
        len(low_variance_columns),
        threshold,
    )
    filtered_train = train_frame.drop(columns=low_variance_columns, errors='ignore')
    filtered_test = test_frame.drop(columns=low_variance_columns, errors='ignore')

    return filtered_train, filtered_test, low_variance_columns


def apply_correlation_threshold(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop highly correlated columns to reduce redundancy."""

    if threshold <= 0.0 or threshold >= 1.0:
        return train_frame, test_frame, []
    numeric_train = train_frame.select_dtypes(include=[np.number])
    if numeric_train.empty:
        return train_frame, test_frame, []

    corr_matrix = numeric_train.corr(method='pearson').abs()
    upper_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(upper_mask)
    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]
    if not to_drop:
        return train_frame, test_frame, []

    logger.info(
        'Dropping %d highly correlated feature(s) above threshold %.4f.',
        len(to_drop),
        threshold,
    )

    filtered_train = train_frame.drop(columns=to_drop, errors='ignore')
    filtered_test = test_frame.drop(columns=to_drop, errors='ignore')

    return filtered_train, filtered_test, to_drop


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
    train_features, test_features = engineer_frequency_features(train_features, test_features)
    # potential_improvement15 (addressed): append frequency encodings for key categorical fields to capture prevalence trends.
    train_features, test_features = engineer_group_statistics(train_features, test_features)
    # potential_improvement17 (addressed): aggregate TransactionAmt statistics per grouping key to expose customer spending behavior.
    train_features, test_features = engineer_missing_indicators(train_features, test_features)
    # potential_improvement18 (addressed): add binary missing-value indicators so the model can learn from absence patterns.
    train_features, test_features = engineer_interaction_features(train_features, test_features)
    # potential_improvement19 (addressed): inject key interaction terms to capture non-linear relationships between engineered signals.
    
    X = train_features.drop(columns=['TransactionID', 'TransactionDT'], errors='ignore')
    X_test = test_features.drop(columns=['TransactionID', 'TransactionDT'], errors='ignore')
    X = X.astype('float32')
    X_test = X_test.astype('float32')
    # potential_improvement6 (addressed): keep high-precision continuous values or normalized variants instead of rounding everything to whole numbers.

    variance_threshold = max(0.0, float(args.variance_threshold))
    X, X_test, dropped_low_variance_columns = apply_variance_threshold(X, X_test, variance_threshold)
    # potential_improvement16 (addressed): remove near-constant signals using a variance threshold to reduce noise and overfitting.

    correlation_threshold = min(max(0.0, float(args.correlation_threshold)), 0.999)
    X, X_test, dropped_correlated_columns = apply_correlation_threshold(X, X_test, correlation_threshold)
    # potential_improvement20 (addressed): prune highly correlated features to reduce redundancy and improve generalization.

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

    calibration_model: LogisticRegression | None = None
    if args.calibrate_probabilities:
        calibration_model = LogisticRegression(max_iter=1000, solver='lbfgs')
        calibration_model.fit(val_pred.reshape(-1, 1), y_valid.values)
        val_pred = calibration_model.predict_proba(val_pred.reshape(-1, 1))[:, 1]
        test_pred = calibration_model.predict_proba(test_pred.reshape(-1, 1))[:, 1]
        logger.info('Applied Platt scaling calibration on validation predictions.')
        # potential_improvement13 (addressed): calibrate predicted probabilities before generating submissions.

    best_threshold = None
    best_threshold_metric = None
    best_threshold_score = None
    if args.optimize_threshold:
        threshold_grid = np.linspace(0.05, 0.95, 181)
        best_threshold = 0.5
        best_threshold_metric = 'f1'
        best_threshold_score = -1.0
        for threshold in threshold_grid:
            threshold_pred = (val_pred >= threshold).astype(int)
            threshold_score = f1_score(y_valid, threshold_pred)
            if threshold_score > best_threshold_score:
                best_threshold = float(threshold)
                best_threshold_score = float(threshold_score)
        logger.info(
            'Optimized validation threshold %.4f with F1 score %.4f.',
            best_threshold,
            best_threshold_score,
        )
        # potential_improvement14 (addressed): tune a downstream classification threshold using validation predictions.

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
        'probability_calibration': bool(args.calibrate_probabilities),
        'best_threshold': best_threshold,
        'best_threshold_metric': best_threshold_metric,
        'best_threshold_score': best_threshold_score,
        'variance_threshold': variance_threshold,
        'dropped_low_variance_columns': dropped_low_variance_columns,
        'correlation_threshold': correlation_threshold,
        'dropped_correlated_columns': dropped_correlated_columns,
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

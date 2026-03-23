"""Train a LightGBM baseline for IEEE-CIS fraud detection.

Run:
    python3 train.py
"""

from __future__ import annotations

import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = SCRIPT_DIR / 'result'

RANDOM_STATE = 42
N_ESTIMATORS = 2000
LEARNING_RATE = 0.03
NUM_LEAVES = 64
N_JOBS = -1
VALID_TIME_QUANTILE = 0.8


def load_and_merge_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info('Loading transaction and identity tables ...')
    train_transaction = pd.read_csv(data_dir / 'train_transaction.csv')
    test_transaction = pd.read_csv(data_dir / 'test_transaction.csv')
    train_identity = pd.read_csv(data_dir / 'train_identity.csv')
    test_identity = pd.read_csv(data_dir / 'test_identity.csv')

    logger.info('Merging train tables on TransactionID ...')
    train_merged = train_transaction.merge(train_identity, on='TransactionID', how='left')

    logger.info('Merging test tables on TransactionID ...')
    test_merged = test_transaction.merge(test_identity, on='TransactionID', how='left')

    return train_merged, test_merged


def encode_categorical_columns(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = sorted(set(train_frame.columns) | set(test_frame.columns))
    train_frame = train_frame.reindex(columns=all_columns)
    test_frame = test_frame.reindex(columns=all_columns)

    categorical_columns = [
        column
        for column in all_columns
        if pd.api.types.is_object_dtype(train_frame[column].dtype)
        or pd.api.types.is_string_dtype(train_frame[column].dtype)
        or isinstance(train_frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(test_frame[column].dtype)
        or pd.api.types.is_string_dtype(test_frame[column].dtype)
        or isinstance(test_frame[column].dtype, pd.CategoricalDtype)
    ]

    logger.info('Encoding %d categorical columns ...', len(categorical_columns))
    for column in categorical_columns:
        combined = pd.concat([train_frame[column], test_frame[column]], axis=0)
        combined = combined.astype('string').fillna('__MISSING__')

        codes, _ = pd.factorize(combined, sort=False)
        train_frame[column] = codes[: len(train_frame)].astype(np.int32)
        test_frame[column] = codes[len(train_frame) :].astype(np.int32)

    return train_frame, test_frame


def preprocess_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    train_copy, test_copy = encode_categorical_columns(train_copy, test_copy)

    train_copy = train_copy.replace([np.inf, -np.inf], np.nan).fillna(-999)
    test_copy = test_copy.replace([np.inf, -np.inf], np.nan).fillna(-999)

    for column in train_copy.columns:
        if pd.api.types.is_float_dtype(train_copy[column]):
            train_copy[column] = train_copy[column].astype(np.float32)
            test_copy[column] = test_copy[column].astype(np.float32)
        elif pd.api.types.is_integer_dtype(train_copy[column]):
            train_copy[column] = train_copy[column].astype(np.int32)
            test_copy[column] = test_copy[column].astype(np.int32)

    return train_copy, test_copy


def time_based_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    time_threshold = float(features['TransactionDT'].quantile(VALID_TIME_QUANTILE))
    train_mask = features['TransactionDT'] <= time_threshold
    valid_mask = features['TransactionDT'] > time_threshold

    if valid_mask.sum() == 0 or train_mask.sum() == 0:
        logger.warning('Time split failed (empty side), falling back to stratified random split.')
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        return X_train, X_valid, y_train, y_valid, float('nan')

    X_train = features.loc[train_mask].copy()
    y_train = target.loc[train_mask].copy()
    X_valid = features.loc[valid_mask].copy()
    y_valid = target.loc[valid_mask].copy()

    return X_train, X_valid, y_train, y_valid, time_threshold


def train() -> None:
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

    X = train_features.drop(columns=['TransactionID'], errors='ignore')
    X_test = test_features.drop(columns=['TransactionID'], errors='ignore')

    X_train, X_valid, y_train, y_valid, time_threshold = time_based_validation_split(X, y)

    pos_count = int(y_train.sum())
    neg_count = int((1 - y_train).sum())
    scale_pos_weight = neg_count / max(pos_count, 1)

    logger.info('Training LightGBM baseline ...')
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(200, first_metric_only=True), lgb.log_evaluation(100)],
    )

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
    }
    metrics_path = OUTPUT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

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
    train()


if __name__ == '__main__':
    main()

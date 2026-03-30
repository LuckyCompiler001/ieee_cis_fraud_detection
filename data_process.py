"""Data processing utilities for the IEEE-CIS fraud detection baseline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def load_and_merge_tables(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load transaction and identity tables and return merged train/test frames."""

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def restrict_to_naive_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    naive_features: Sequence[str],
    modulo: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    naive_columns = [column for column in naive_features if column in train_frame.columns]
    if not naive_columns:
        numeric_columns = train_frame.select_dtypes(include=[np.number]).columns.tolist()
        naive_columns = numeric_columns[:1] if numeric_columns else [train_frame.columns[0]]

    logger.info('Restricting to naive feature set: %s', ', '.join(naive_columns))

    train_simple = train_frame.loc[:, naive_columns].copy()
    test_simple = test_frame.loc[:, naive_columns].copy()

    for column in naive_columns:
        train_simple[column] = (train_simple[column] % modulo).astype(np.float32)
        test_simple[column] = (test_simple[column] % modulo).astype(np.float32)

    return train_simple, test_simple


def time_based_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
    valid_time_quantile: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    time_threshold = float(features['TransactionDT'].quantile(valid_time_quantile))
    train_mask = features['TransactionDT'] <= time_threshold
    valid_mask = features['TransactionDT'] > time_threshold

    if valid_mask.sum() == 0 or train_mask.sum() == 0:
        logger.warning('Time split failed (empty side), falling back to stratified random split.')
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=random_state,
            stratify=target,
        )
        return X_train, X_valid, y_train, y_valid, float('nan')

    X_train = features.loc[train_mask].copy()
    y_train = target.loc[train_mask].copy()
    X_valid = features.loc[valid_mask].copy()
    y_valid = target.loc[valid_mask].copy()

    return X_train, X_valid, y_train, y_valid, time_threshold

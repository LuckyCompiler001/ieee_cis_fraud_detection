# IEEE-CIS Fraud Detection

## 1) Task Definition

This project focuses on fraud detection in online transactions.

Objective: build a machine learning model to predict whether a transaction is fraudulent.

- Problem type: binary classification
- Target variable:
  - `isFraud = 1` → fraudulent transaction
  - `isFraud = 0` → legitimate transaction

The dataset comes from real-world e-commerce transactions provided by Vesta Corporation and is widely used as a benchmark for fraud detection systems.

## 2) Dataset Structure

The dataset consists of two complementary tables.

### (1) Transaction table (core data)

- `data/train_transaction.csv`
- `data/test_transaction.csv`

Each row represents a single transaction.

Key properties:

- ~590,000 training samples
- ~430+ features
- includes target `isFraud` (train only)

### (2) Identity table (auxiliary data)

- `data/train_identity.csv`
- `data/test_identity.csv`

Contains:

- device information
- browser / OS
- user-related metadata

### (3) Table relationship

Join key: `TransactionID`

`train = train_transaction.merge(train_identity, on="TransactionID", how="left")`

Not all transactions have identity data.

## 3) Feature Overview

The dataset contains multiple feature groups.

### Transaction features

- `TransactionDT` → relative time (not absolute timestamp)
- `TransactionAmt` → transaction amount
- `ProductCD` → product category
- `card1`–`card6` → card information
- `addr1`, `addr2` → address

### Behavioral / engineered features

- `C1`–`C14` → count-style features
- `D1`–`D15` → time gaps / delays
- `dist1`, `dist2` → distance metrics

### High-dimensional anonymized features

- `V1`–`V339`

These are transformed / engineered features that are often highly predictive but not directly interpretable.

### Identity features

- device type
- browser
- operating system

## 4) Machine Learning Formulation

`f(X) -> P(fraud)`

Where:

- input `X` = transaction + identity features
- output = probability of fraud

## 5) Model Selection

### Baseline model (recommended)

Use LightGBM (Gradient Boosting Decision Tree).

Why:

- strong performance on tabular data
- handles missing values natively
- scalable to 400+ features

### Alternative models

- XGBoost
- CatBoost
- Logistic Regression (baseline check)

Tree-based boosting models are widely used on this dataset due to strong performance with heterogeneous features.

## 6) Input to the Model

After preprocessing:

- each row = one transaction
- features include transaction, identity, and anonymized `V` features

Preprocessing steps:

- handle missing values
- encode categorical variables
- optionally normalize numeric features

## 7) Output of the Model

`y_pred = model.predict_proba(X)[:, 1]`

Meaning:

- output = probability transaction is fraud
- range = `[0, 1]`

## 8) Evaluation Metric

Primary metric: ROC-AUC.

Reason:

- dataset is highly imbalanced (~3–4% fraud)
- AUC evaluates ranking quality better than raw accuracy for this setup

## 9) Key Challenges

1. **Class imbalance**
   - ~3.5% fraud cases, naive models bias toward non-fraud
2. **Temporal structure**
   - `TransactionDT` encodes time and splits are chronological
   - improper validation can cause leakage
3. **High dimensionality**
   - 400+ features, many correlated or redundant
4. **Missing data**
   - identity table is sparse and many columns have high missing rates

## 10) End-to-End Pipeline

Raw CSVs  
↓  
Merge transaction + identity  
↓  
Preprocessing (missing values, encoding)  
↓  
Train model (LightGBM)  
↓  
Predict probability of fraud  
↓  
Evaluate with ROC-AUC

## 11) First Milestone (Baseline)

Before advanced work:

1. Merge transaction + identity
2. Train LightGBM on raw features
3. Use simple preprocessing (`fillna` + encoding)
4. Evaluate AUC

## 12) Summary

- Task: binary classification (fraud detection)
- Input: transaction + identity features
- Output: fraud probability
- Model: LightGBM baseline
- Metric: ROC-AUC
- Core difficulty: imbalance + temporal validation


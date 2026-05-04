# Bank Churn Analysis

A machine learning project analyzing customer attrition for a bank, built to identify at-risk customers and support targeted retention strategies.

## Project Overview

This project takes a real-world bank customer dataset through a full data science workflow. The goal is to classify customers as churned or retained, and to identify the behavioral signals that most strongly predict attrition.

## Workflow

**Data Cleaning & Validation** — Removed logically invalid records (e.g. unrealistic ages, revolving balances exceeding credit limits), handled missing values, and standardized categorical features to ensure analysis integrity.

**Exploratory Data Analysis** — Investigated key variables including customer inactivity, credit utilization, tenure, and transaction behavior. Used correlation plots, distribution charts, and statistical tests to surface patterns separating churned from retained customers.

**Statistical Testing** — Applied t-tests, point-biserial correlation, ANOVA, chi-square, and Fisher's exact tests to validate the significance of identified patterns.

**Machine Learning Models** — Built and compared two classifiers:
- *Logistic Regression* — baseline and threshold-optimized versions, with GridSearchCV tuning and class imbalance handling
- *Random Forest* — tuned via GridSearchCV across 96 parameter combinations with 5-fold cross-validation

## Results

| Model | Test Accuracy | AUC | Churn Recall |
|---|---|---|---|
| Logistic Regression (baseline) | 85.3% | 0.917 | — |
| Logistic Regression (recall-optimized) | 62.6% | 0.866 | 90.5% |
| **Random Forest (tuned)** | **94.9%** | **0.985** | **75.7%** |

The tuned Random Forest was the strongest overall performer, achieving 94.9% accuracy and a 0.985 AUC. Analysis confirmed that customer inactivity, transaction frequency, and credit utilization are the strongest predictors of churn.

## Tech Stack

Python · Pandas · NumPy · scikit-learn · SciPy · Seaborn · Matplotlib · Jupyter Notebook

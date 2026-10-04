
# Bias-Aware and Explainable Mortgage Approval Prediction Using HMDA Regulatory Data

## Project Overview

This capstone project develops a bias-aware and explainable machine learning framework for mortgage approval prediction using publicly available Home Mortgage Disclosure Act (HMDA) regulatory data. The project focuses on predicting whether mortgage applications are approved/originated or denied while evaluating whether model outcomes differ across demographic groups.

The goal is not only to build an accurate predictive model, but also to assess fairness, explain model behavior, and demonstrate how responsible artificial intelligence techniques can be applied in a regulated financial services context.

## Research Problem

Mortgage lenders increasingly use analytics and machine learning to support credit risk assessment and application decisioning. However, historical lending data may reflect structural and socioeconomic disparities. If machine learning models are trained only to maximize predictive accuracy, they may unintentionally reproduce or amplify disparities across protected demographic groups.

This project addresses the problem by building a predictive framework that explicitly evaluates both model performance and fairness.

## Research Questions

The project is guided by the following research questions:

1. Can machine learning models accurately predict mortgage approval decisions using applicant and loan characteristics?
2. Do statistically significant disparities exist in mortgage approval outcomes across demographic groups?
3. Which applicant and loan-related factors have the greatest influence on mortgage approval decisions?
4. Can fairness-aware machine learning techniques reduce demographic bias while maintaining acceptable predictive accuracy?

## Data Source

The dataset is derived from the public HMDA loan-level data available through the Consumer Financial Protection Bureau / FFIEC HMDA Data Browser.

Source: https://ffiec.cfpb.gov/data-browser/

The project will use a filtered state-level sample from the 2023 HMDA public dataset. The analysis will focus on home-purchase mortgage applications where the action taken represents either:

- Loan originated / approved
- Application denied

Public HMDA data are modified to protect applicant and borrower privacy. Therefore, this project uses only publicly available fields and does not attempt to infer confidential variables such as exact credit scores.

## Dataset Scope

The final modeling dataset will include selected applicant, loan, demographic, institutional, and geographic variables relevant to mortgage approval prediction and fairness analysis.

Recommended fields include:

| Category | Variables |
|---|---|
| Target Variable | `action_taken` |
| Applicant Characteristics | `income`, `race`, `ethnicity`, `sex`, `age` |
| Loan Characteristics | `loan_amount`, `loan_type`, `loan_purpose`, `loan_term`, `interest_rate`, `property_value` |
| Risk-Related Variables | `debt_to_income_ratio`, `combined_loan_to_value_ratio`, `lien_status` |
| Property / Occupancy | `occupancy_type`, `total_units`, `business_or_commercial_purpose` |
| Geography | `state_code`, `county_code`, `census_tract` |
| Institution | `lei` |

## Repository Structure

```text
bias-aware-hmda-mortgage-approval/
│
├── data/
│   ├── raw/
│   │   └── state_MI.csv
│   ├── processed/
│   │   └── hmda_2023_modeling_dataset.csv
│   └── data_dictionary.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_model_development.ipynb
│   ├── 04_explainability_analysis.ipynb
│   └── 05_fairness_and_bias_mitigation.ipynb
│
├── outputs/
│   ├── figures/
│   ├── model_metrics/
│   └── fairness_metrics/
│
├── README.md
└── requirements.txt

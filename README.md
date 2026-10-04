# Bias-Aware and Explainable Mortgage Application Outcome Prediction Using HMDA Regulatory Data

## Project Overview

This capstone project develops a bias-aware and explainable machine learning framework for mortgage application outcome prediction using publicly available Home Mortgage Disclosure Act (HMDA) regulatory data. The project focuses on predicting whether mortgage applications result in loan origination or denial while evaluating whether model outcomes differ across demographic groups.

The goal is not only to build an accurate predictive model, but also to assess fairness, explain model behavior, and demonstrate how responsible artificial intelligence techniques can be applied in a regulated financial services context.

## Research Problem

Mortgage lenders increasingly use analytics and machine learning to support credit risk assessment and mortgage application decisioning. However, historical lending data may reflect structural and socioeconomic disparities. If machine learning models are trained only to maximize predictive accuracy, they may unintentionally reproduce or amplify disparities across protected demographic groups.

This project addresses the problem by building a predictive framework that explicitly evaluates both model performance and fairness.

## Research Questions

The project is guided by the following research questions:

1. Can machine learning models accurately predict mortgage application outcomes using applicant and loan characteristics?
2. Do statistically significant disparities exist in mortgage application outcomes across demographic groups?
3. Which applicant and loan-related factors have the greatest influence on mortgage application outcomes?
4. Can fairness-aware machine learning techniques reduce demographic bias while maintaining acceptable predictive accuracy?

## Data Source

The dataset is derived from the public HMDA loan-level data available through the Consumer Financial Protection Bureau / FFIEC HMDA Data Browser.

Source: https://ffiec.cfpb.gov/data-browser/

The project will use a filtered state-level sample from the 2023 HMDA public dataset. The analysis will focus on home-purchase mortgage applications where the action taken represents either:

- Loan originated
- Application denied

For modeling, the target variable will be recoded into a binary outcome using selected HMDA action codes, such as:

- `1 = Loan originated`
- `3 = Application denied`

Public HMDA data are modified to protect applicant and borrower privacy. Therefore, this project uses only publicly available fields and does not attempt to infer confidential variables such as exact credit scores.

## Dataset Scope

The final modeling dataset will include selected applicant, loan, demographic, institutional, geographic, underwriting, property, and census-tract variables relevant to mortgage application outcome prediction and fairness analysis.

The raw dataset may retain a broader set of selected HMDA fields. The processed modeling dataset will exclude post-decision variables, high-missingness fields, and fields that may create target leakage.

## Data Dictionary

| Variable Group | Fields | Role in Study | Use in Modeling |
|---|---|---|---|
| Target Variable | `action_taken` | Dependent variable used to classify mortgage applications as loan originated or denied. | Model target. Retain selected binary outcome values, such as `1 = loan originated` and `3 = application denied`, for classification. |
| Application and Institution Identifiers | `activity_year`, `lei` | Identify reporting year and lending institution. | Retain for traceability and exploratory analysis. Exclude or encode carefully in predictive modeling. |
| Geographic Identifiers | `derived_msa-md`, `state_code`, `county_code`, `census_tract` | Represent property location and market geography. | Use as contextual predictors or grouping variables after encoding. |
| Loan Product and Structure | `conforming_loan_limit`, `derived_loan_product_type`, `derived_dwelling_category`, `loan_type`, `loan_purpose`, `lien_status`, `preapproval`, `reverse_mortgage`, `open-end_line_of_credit`, `business_or_commercial_purpose` | Describe the loan product, application type, lien position, and purpose of borrowing. | Core independent variables for model development and exploratory analysis. |
| Loan Amount and Pricing Characteristics | `loan_amount`, `loan_to_value_ratio`, `interest_rate`, `loan_term`, `property_value` | Capture requested credit amount, collateral value, pricing, and repayment term. | Use as core predictors, subject to missing-value treatment and outlier review. |
| Applicant Financial Characteristics | `income`, `debt_to_income_ratio`, `applicant_credit_score_type`, `co-applicant_credit_score_type` | Provide financial capacity and underwriting-related information available in public HMDA data. | Use `income`, `debt_to_income_ratio`, and `applicant_credit_score_type` as primary predictors. Treat `co-applicant_credit_score_type` as optional. |
| Applicant Demographic Audit Fields | `derived_ethnicity`, `derived_race`, `derived_sex`, `applicant_ethnicity-1`, `applicant_race-1`, `applicant_sex`, `applicant_age`, `applicant_age_above_62` | Support fairness auditing and subgroup outcome analysis. | Retain primarily for fairness metrics and bias analysis. Exclude from the primary operational prediction model unless used in controlled fairness experiments. |
| Co-applicant Demographic Fields | `co-applicant_ethnicity-1`, `co-applicant_race-1`, `co-applicant_sex`, `co-applicant_age`, `co-applicant_age_above_62` | Support optional analysis of joint applications. | Retain in raw data. Exclude from initial modeling unless joint-application analysis is added. |
| Multi-select Demographic Fields | `applicant_ethnicity-2` to `applicant_ethnicity-5`, `applicant_race-2` to `applicant_race-5`, `co-applicant_ethnicity-2` to `co-applicant_ethnicity-5`, `co-applicant_race-2` to `co-applicant_race-5` | Capture additional reported ethnicity and race categories. | Retain in raw data. Generally exclude from first modeling version to reduce sparsity and complexity. |
| Demographic Observation Fields | `applicant_ethnicity_observed`, `co-applicant_ethnicity_observed`, `applicant_race_observed`, `co-applicant_race_observed`, `applicant_sex_observed`, `co-applicant_sex_observed` | Indicate whether selected demographic information was visually observed or otherwise recorded. | Exclude from predictive modeling. May be referenced only in data-quality discussion. |
| Underwriting and Application Channel | `submission_of_application`, `initially_payable_to_institution`, `aus-1`, `aus-2`, `aus-3`, `aus-4`, `aus-5` | Describe the submission channel, payee relationship, and automated underwriting system information. | Use `submission_of_application`, `initially_payable_to_institution`, and `aus-1` as candidate predictors. Retain additional AUS fields only if coverage is sufficient. |

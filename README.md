## Data Source

The dataset is derived from the public HMDA loan-level data available through the Consumer Financial Protection Bureau / FFIEC HMDA Data Browser.

Source: https://ffiec.cfpb.gov/data-browser/

The project uses a filtered Michigan state-level sample from the 2023 HMDA public dataset. The analysis focuses on home-purchase mortgage applications where the action taken represents either:

- Loan originated
- Application denied

For modeling, the target variable will be recoded into a binary outcome using selected HMDA action codes, such as:

- `1 = Loan originated`
- `3 = Application denied`

The raw selected dataset used for this project is available here:

CSV file: https://github.com/royjoseblr/bias-aware-hmda-mortgage-approval/blob/main/data/raw/state_MI.csv

Direct raw download: https://raw.githubusercontent.com/royjoseblr/bias-aware-hmda-mortgage-approval/main/data/raw/state_MI.csv

Public HMDA data are modified to protect applicant and borrower privacy. Therefore, this project uses only publicly available fields and does not attempt to infer confidential variables such as exact credit scores.

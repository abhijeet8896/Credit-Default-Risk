# Credit Default Prediction using Machine Learning

## Project Overview

This project predicts whether a credit card customer will default on their payment in the following month using historical repayment, billing, and demographic information.

The objective is to support better **credit risk assessment** by identifying borrower patterns associated with default behaviour.

The project includes:

- Data preprocessing and quality checks
- Exploratory Data Analysis (EDA)
- Feature engineering
- Predictive modeling
- Model comparison
- SQL-based business analysis

Final modeling was performed using **XGBoost**, with **Logistic Regression** as a baseline model.

---

## Problem Statement

Credit card defaults can result in significant financial losses for lending institutions.

The goal of this project is to build a machine learning model that can:

> predict the probability of borrower default in the next month

based on previous repayment history, billing amounts, payment behaviour, and customer profile information.

---

## Dataset Source

Dataset used:

**Default of Credit Card Clients Dataset**

Source:

:contentReference[oaicite:0]{index=0}

The dataset contains information on:

- Credit limit
- Gender
- Education
- Marital status
- Age
- Repayment history
- Bill statements
- Previous payments

### Target Variable

```text
default.payment.next.month
```

- `1` → Default
- `0` → No Default

---

## Project Structure

```text
project/
│── Credit Risk Default Analysis.ipynb
│── credit_default.csv
│── README.md
│── writeup.pdf
```

---

## Setup Guide

### 1. Clone or Download the Project

Download the project files or clone the repository.

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost sqlite3 jupyter
```

### 4. Run Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebook.ipynb
```

Run all cells sequentially.

---

## Methodology Summary

The project followed the following workflow:

### 1. Data Understanding

Initial inspection included:

- Dataset shape
- Data types
- Missing values
- Class balance
- Data quality issues

### 2. Data Preprocessing

Special handling was applied to assignment edge cases:

#### EDUCATION

Values:

```text
0, 5, 6
```

were merged into:

```text
Other
```

#### MARRIAGE

Value:

```text
0
```

was merged into:

```text
Other
```

Negative `BILL_AMT` values were checked and documented as possible overpayments or billing adjustments.

### 3. Exploratory Data Analysis (EDA)

EDA focused on:

- Credit limit distribution
- Age distribution
- Repayment delay behaviour
- Default rate across demographic groups
- Repayment trends by default outcome
- Correlation analysis

### 4. Feature Engineering

Additional behavioural features were created to improve prediction quality.

### 5. Model Development

Two models were trained:

#### Logistic Regression
Used as an interpretable baseline.

#### XGBoost
Used as the final tree-based model for improved predictive performance.

Stratified **5-fold cross-validation** was used for evaluation.

Primary evaluation metric:

> **AUC-ROC**

Additional metrics:

- Precision
- Recall
- F1-score

---

## Engineered Features

The following features were created:

### AVG_UTIL_RATE

Average credit utilization over six months.

Formula:

```text
mean(BILL_AMT / LIMIT_BAL)
```

Purpose:

> measure credit utilization behaviour

---

### AVG_PAY_RATIO

Average repayment consistency.

Formula:

```text
mean(PAY_AMT / BILL_AMT)
```

(only where `BILL_AMT > 0`)

Purpose:

> measure repayment discipline

---

### TOTAL_DELAY_MONTHS

Count of repayment delay months.

Formula:

```text
count(PAY_x > 0)
```

Purpose:

> cumulative delinquency signal

---

## Model Performance

### Baseline Model

**Logistic Regression**

Used for benchmark comparison.

Handled class imbalance using:

```python
class_weight='balanced'
```

---

### Final Model

**XGBoost**

Selected because it better captures:

- Nonlinear repayment behaviour
- Feature interactions
- Financial risk patterns

Final performance:

```text
AUC-ROC ≈ 0.75
```

---

## SQL Analysis

As an additional enhancement, the dataset was loaded into **SQLite** to answer business-oriented questions.

Example analyses:

- Default rate by education level
- Default rate by age group
- Average credit limit by default status

---

## Key Findings

### 1. Repayment Behaviour Is the Strongest Signal

Borrowers with repeated payment delays were significantly more likely to default.

Repayment history consistently showed stronger predictive power than demographic characteristics.

---

### 2. Financial Behaviour Matters More Than Demographics

Variables such as:

- Education
- Marital status
- Age

showed weaker predictive impact compared to repayment-related variables.

---

### 3. Delinquency Persistence Is Important

Customers with multiple delayed repayment months demonstrated higher default probability.

This suggests repayment consistency is an important lending signal.

---

### 4. XGBoost Outperformed Logistic Regression

XGBoost produced stronger predictive performance and better captured borrower risk complexity.

---

## Future Improvements

Given additional time, the following improvements could be implemented:

- SHAP explainability for prediction interpretation
- Fairness evaluation across demographic groups
- Hyperparameter optimization
- Streamlit deployment for real-time prediction

---

## Author
Abhijeet Bagal

# Predicting Change in Credit Score

## Objective

Build a machine learning model to predict whether a customer's credit score will:

- Increase
- Decrease
- Remain Stable

in the next 3 months, based on their current financial behavior using realistic synthetic data.

## Project Structure

- `jupiter_credit_dataset.csv`: Synthetic dataset with 50,000 rows.
- `notebook.ipynb`: Full EDA, modeling, SHAP analysis, and validation.
- `model_comparison.csv`: Model performance results.
- `final_predictions.csv`: Final test set predictions.
- `README.md`: This documentation file.
- `plots/`: Directory containing SHAP, ROC-AUC, correlation, and other visualizations.

## Assumptions

To simulate real-world credit movement trends without historical bureau data, the following assumptions were made:

1. **Single-Month Snapshot as Future Proxy**  
   - Each row represents one customer for one month.
   - Current financial features are used to predict expected credit movement in the next 3 months.

2. **Synthetic Risk-Based Labeling**  
   - Labels (`increase`, `stable`, `decrease`) are assigned using a risk score.
   - The risk score is a weighted combination of:
     - `dpd_last_3_months`
     - `credit_utilization_ratio`
     - `emi_income_ratio`
     - `repayment_history_score`
     - `num_hard_inquiries_last_6m`

3. **Injecting Realistic Risk Cases**  
   - 5% of customers are deliberately set as high-risk to simulate default-like behavior:
     - EMI > 65% of income
     - Utilization > 90%
     - DPD > 30 days
     - Recent defaults within 6 months

4. **No Temporal Features**  
   - This version assumes static monthly data.
   - In a real deployment, this could evolve into time-series modeling (e.g., using LSTMs).

## Dataset Overview

| Feature                        | Description                                |
|--------------------------------|--------------------------------------------|
| `customer_id`                 | Unique ID (repeated over 5 months)         |
| `age`, `gender`, `location`   | Demographic features                       |
| `monthly_income`              | Monthly income in ₹                        |
| `monthly_emi_outflow`         | EMI burden                                 |
| `current_outstanding`         | Current dues                               |
| `credit_utilization_ratio`    | Outstanding / Credit Limit                 |
| `repayment_history_score`     | Score from 0 to 100                        |
| `dpd_last_3_months`           | Days Past Due                              |
| `num_hard_inquiries_last_6m`  | Number of credit pulls                     |
| `months_since_last_default`   | Recency of last default                    |
| `target_credit_score_movement`| `increase` / `stable` / `decrease` (target)|

- **Dataset Size**: 50,000 rows
- **Target Balance (after injection)**:
  - `increase`: 80.15% (40,073 rows)
  - `stable`: 14.85% (7,427 rows)
  - `decrease`: 5.00% (2,500 rows)

## EDA Insights

Refer to `notebook folder` for full plots and images.

### 1. Target Distribution
- Highly imbalanced target with `increase` dominating at 80.15% (40,073 rows), followed by `stable` at 14.85% (7,427 rows), and `decrease` at 5.00% (2,500 rows).
- The `decrease` class, though a small minority, is business-critical as it represents customers at risk of credit score decline.

**Action**: Emphasize high recall for the `decrease` class to ensure at-risk customers are identified, using techniques like SMOTE to address the imbalance.

### 2. Risk Score Distribution
- Risk score peaks near 0.3 (low risk), with a tail above 0.7 (high risk).
- Labels are assigned as follows:
  - `risk_score < 0.25`: `increase`
  - `0.25–0.45`: `stable`
  - `> 0.45`: `decrease`

### 3. Feature Behavior by Class

| Feature                   | `increase` Class | `decrease` Class |
|---------------------------|------------------|------------------|
| `dpd_last_3_months`       | 0–10             | 30–90            |
| `credit_utilization_ratio`| <0.4             | >0.9             |
| `repayment_history_score` | >80              | 20–40            |
| `emi_income_ratio`        | <0.3             | >0.65            |

**Insight**: Confirmed risk logic:  
- High DPD, high EMI, and low repayment scores correlate with `decrease`.  
- Low DPD and good repayment scores correlate with `increase`.

### 4. Demographics
- Low-income customers (<₹30,000) are more likely to have `decrease`.
- Young (<30) and older (>50) customers show a slight skew toward `decrease`.
- Tier-1 cities (e.g., Mumbai, Bangalore) exhibit higher risk, likely due to increased credit usage.

**Action**: Consider city-specific interventions to manage risk in Tier-1 cities.

## Modeling Approach

### Preprocessing
- **Categorical Features**: `gender`, `location`, `credit_utilization_binned` (encoded).
- **Numerical Features**: Scaled using `StandardScaler`.
- **Class Imbalance**: SMOTE applied only to training data to prevent leakage.

### Train/Test Split
- Performed a customer-level split to prevent data leakage (ensuring the same customer does not appear in both train and test sets).

### Models Trained
**Model Comparison Results**:

| Model              | CV F1 Mean | CV F1 Std | Test F1 Macro |
|--------------------|------------|-----------|---------------|
| Logistic Regression| 0.997478   | 0.000650  | 0.992735      |
| XGBoost           | 0.992869   | 0.000866  | 0.976916      |
| Random Forest     | 0.964171   | 0.001453  | 0.909388      |

**Best Model Selected**: Random Forest  
- Chosen for high performance, explainability, and robustness to feature scaling.

## SHAP Explainability

**Top Features by SHAP Importance**:

1. `credit_utilization_ratio`
2. `dpd_last_3_months`
3. `repayment_history_score`
4. `emi_income_ratio`
5. `num_hard_inquiries_last_6m`

**Insight**: The model aligns with credit bureau heuristics, prioritizing features like credit utilization and payment delinquency.

## Business Recommendations

### High-Risk Segments
- Customers with:
  - `dpd_last_3_months > 30`
  - `credit_utilization_ratio > 0.9`
  - `emi_income_ratio > 0.6`

**Policy Interventions**:
- Offer repayment counseling.
- Trigger automated payment reminders.
- Freeze credit limit increases.

### High-Opportunity Segments
- Customers with:
  - `repayment_history_score > 80`
  - `credit_utilization_ratio < 0.4`

**Product Offers**:
- Pre-approved credit limit increases.
- Upsell to premium credit cards.
- Invite to loyalty programs.

### Segment-Based Strategies

| Segment                  | Strategy                              |
|--------------------------|---------------------------------------|
| Low-income customers     | Limit EMI exposure, educate on credit |
| Tier-1 cities (e.g., Mumbai) | Monitor closely, stricter approvals |
| `stable` score users     | Early monitoring to prevent decline   |

## Availability

I confirm I am available to join onsite at Jupiter HQ in Bangalore starting June 23, 2025, if selected.

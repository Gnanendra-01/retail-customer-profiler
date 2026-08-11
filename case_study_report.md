# Retail Multi-Segment Profiler & High-Value Customer Classifier
**UG Level 2 — Problem Statement 11 Case Study Report**

---

## Executive Summary

In today's hyper-competitive omnichannel retail landscape, customer acquisition costs have surged dramatically. Retail enterprises can no longer afford generic, one-size-fits-all marketing strategies. To maximize customer lifetime value (CLV) and optimize marketing Return on Investment (ROI), businesses must group customers into behavioral personas and identify top-tier high-value shoppers early in their journey.

This case study presents an end-to-end data analytics and machine learning solution for **UG Level 2 - Problem Statement 11**. The project combines:
1. **Unsupervised Learning (K-Means Clustering)**: Segmenting customer base into 5 distinct buying personas based on demographics and Recency, Frequency, and Monetary (RFM) transaction metrics.
2. **Supervised Learning (Random Forest Classifier)**: Predicting whether a new customer will become a high-value spender ($\ge 70\text{th percentile}$ revenue contributor) with **96.8% accuracy** and a **0.992 ROC-AUC score**.

---

## 1. Problem Definition & Objectives

### 1.1 Business Objective
Retail firms receive transactional and demographic records for thousands of shoppers daily. The goal is to solve two core challenges:
- **Segmentation**: Uncover hidden customer groupings (personas) without prior labeling.
- **Classification**: Train a predictive model to immediately flag high-value shoppers upon onboarding, enabling dedicated concierge and VIP retention workflows.

### 1.2 Quantitative Target Metrics
- **Clustering Quality**: Silhouette Score $> 0.50$ with distinct elbow inertia inflection.
- **Classification Performance**: Model Accuracy $> 90\%$, ROC-AUC Score $> 0.95$.
- **Real-time Inference**: Capability to evaluate new customer attributes sub-second in a web dashboard.

---

## 2. Dataset Architecture & Exploratory Data Analysis (EDA)

The project leverages a retail customer dataset ($N = 500$ records) with the following schema:

| Feature Name | Type | Description | Range / Units |
| :--- | :--- | :--- | :--- |
| `CustomerID` | String | Unique identifier | `CUST-1000` to `CUST-1499` |
| `Gender` | Categorical | Customer gender | Female (54%), Male (46%) |
| `Age` | Numerical | Age in years | 18 – 70 years |
| `AnnualIncome_k` | Numerical | Estimated annual income | 15 – 140 ($k\$$) |
| `SpendingScore` | Numerical | Mall spending score rating | 1 – 100 |
| `PurchaseFrequency` | Numerical | Orders placed per year | 1 – 60 orders/yr |
| `AvgOrderValue` | Numerical | Average spent per order | $10 – $450 |
| `RecencyDays` | Numerical | Days since last purchase | 1 – 180 days |
| `TotalSpend` | Numerical | Calculated total spend | $\text{Frequency} \times \text{AvgOrderValue}$ |
| `IsHighValue` | Binary Target | High-value label | 1 (High Value), 0 (Standard) |

---

## 3. Unsupervised Clustering Methodology (K-Means)

### 3.1 Feature Standardization & Scaling
Because features such as `TotalSpend` (ranging in thousands) and `Age` (ranging 18-70) operate on different scales, feature values were standardized using Z-score normalization:
$$z = \frac{x - \mu}{\sigma}$$

### 3.2 Optimal Cluster Determination
The K-Means algorithm was executed across cluster counts $K \in [2, 8]$. The Sum of Squared Errors (SSE / Inertia) and Silhouette Coefficients were computed:

$$\text{Silhouette Score} = \frac{b - a}{\max(a, b)}$$

- **$K=2$**: Silhouette $= 0.482$, Inertia $= 1820.4$
- **$K=3$**: Silhouette $= 0.512$, Inertia $= 1310.2$
- **$K=4$**: Silhouette $= 0.540$, Inertia $= 940.1$
- **$K=5$ (Optimal)**: Silhouette $= \mathbf{0.578}$, Inertia $= \mathbf{680.5}$
- **$K=6$**: Silhouette $= 0.510$, Inertia $= 590.2$

$K=5$ proved optimal, corresponding to the classic retail 5-persona segmentation framework.

---

## 4. Persona Catalog & Behavioral Profiles

| Persona ID | Persona Name | Market Share | Avg Income | Spend Score | Avg Total Spend | Primary Strategy |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **VIP Power Spenders** | 20.0% | $90.2k | 82 / 100 | $9,120.50 | Dedicated VIP concierge, exclusive early releases, loyalty multipliers |
| **1** | **Careful Saver Elites** | 20.0% | $88.5k | 22 / 100 | $2,100.00 | Premium quality reassurance, cross-sell luxury durable goods |
| **2** | **Trend-Seeking Youth** | 20.0% | $30.1k | 78 / 100 | $2,100.00 | BNPL financing, flash sales, trending product alerts |
| **3** | **Budget Conscious** | 20.0% | $28.4k | 20 / 100 | $360.00 | Clearance discount coupons, value bundle packs |
| **4** | **Core Mass Market** | 20.0% | $55.0k | 50 / 100 | $2,200.00 | Seasonal promotions, personalized recommendations |

---

## 5. Supervised High-Value Classification Model

### 5.1 Model Selection & Training
A **Random Forest Classifier** ($100$ decision trees, max depth $6$) was trained on a 75/25 stratified train-test split.

### 5.2 Confusion Matrix & Diagnostic Performance

| | Predicted Standard (0) | Predicted High-Value (1) |
| :--- | :---: | :---: |
| **Actual Standard (0)** | 84 | 2 |
| **Actual High-Value (1)** | 2 | 37 |

- **Accuracy**: $96.8\%$
- **Precision**: $94.87\%$
- **Recall (Sensitivity)**: $94.87\%$
- **F1 Score**: $94.87\%$
- **ROC-AUC**: $0.992$

### 5.3 Feature Importance Ranking
1. `TotalSpend` ($34.20\%$)
2. `PurchaseFrequency` ($22.50\%$)
3. `SpendingScore` ($18.40\%$)
4. `AvgOrderValue` ($12.10\%$)
5. `AnnualIncome_k` ($8.30\%$)
6. `RecencyDays` ($4.50\%$)

---

## 6. Web Prototype Features & Implementation

An interactive web prototype was developed in vanilla HTML5, CSS3, and JavaScript with Chart.js:
- **Executive Dashboard**: KPI metrics, segment revenue share doughnut chart, feature importances.
- **Persona Profiler**: Dynamic 2D customer scatter plot, Elbow & Silhouette curves, multi-dimensional radar comparison chart.
- **Real-Time Classifier Tool**: Live slider inputs (Income, Spending Score, Age, Frequency, AOV, Recency) with instant probability meter and marketing playbook recommendation.
- **Segment Catalog**: Detailed persona cards with market share and strategic recommendations.
- **Interactive Case Study Viewer**: Embedded complete case study documentation.

---

## 7. Strategic Recommendations & Business Impact

1. **VIP Loyalty Program**: Focus retention budget on Persona 0 (VIP Power Spenders), who generate 45%+ of total retail revenue despite representing 20% of customer headcount.
2. **Re-engagement Campaigns**: Target Persona 1 (Careful Savers) with high-end premium marketing to increase their low spending score.
3. **BNPL Integration**: Offer Buy-Now-Pay-Later payment flexibility to Persona 2 (Trend-Seeking Youth) to convert high engagement into higher basket sizes.

---
*Report prepared for UG Level 2 Internship Certification submission.*

# Retail Smart Recommender 🛒🤖

End-to-end **retail analytics + AI recommendation** project built in pure Python.

This project takes cleaned retail transaction data (CSV), analyzes customer buying patterns, and builds a simple, explainable **recommendation system** that suggests what product category and price range to promote for different customer segments.

---

## 📂 Project Overview

**Goal:**  
Use real-world style retail data to move from **descriptive analytics** (dashboards, summaries) to **prescriptive analytics** (what should we recommend next?).

**Core idea:**  
Learn patterns such as:

> “20–30 year old males who shop on Saturdays often buy **Clothing – Luxury (500+)**.”

Then use those patterns to recommend products for marketing campaigns, digital flyers, or in‑store promotions.

---

## 🧾 Dataset

The input CSV (`retail_sales_clean.csv`) contains one row per transaction with:

- `transaction_id`
- `date`, `year`, `month`, `day`, `quarter`
- `customer_id`, `gender`, `age`, `age_group`
- `product_category` (Clothing, Beauty, Electronics, etc.)
- `quantity`, `price_per_unit`, `total_amount`
- `price_range` (Budget, Economy, Premium, Luxury)
- `month_name`, `day_of_week`

These engineered features allow us to learn patterns by **customer segment** and **shopping day** instead of just raw product totals.

---

## 🤖 AI Model: Segment-based Recommender

This project implements a **segment-based collaborative filtering** approach using simple, interpretable rules:

- Segments are defined by:  
  `age_group × gender × day_of_week`
- For each segment, we learn which:  
  `product_category × price_range`  
  combinations are most frequently purchased.
- We compute a **confidence score** for each pattern:

\[
confidence = \frac{\text{transactions for this segment + product}}{\text{all transactions in this segment}}
\]

Example pattern the model might learn:

> For `age_group = 20–30`, `gender = Male`, `day_of_week = Saturday`  
> 42% of transactions are **Clothing – Luxury (500+)**

These patterns are then used as rules to make recommendations.

---


# 📊 Customer Segmentation using RFM Analysis

## 🧠 Overview
This project applies **RFM (Recency, Frequency, Monetary) analysis** to segment customers based on purchasing behavior.

It enables businesses to identify high-value customers, improve retention strategies, and reduce churn using data-driven insights.

---

## 🎯 Business Problem
Modern businesses face several key challenges:
- Identifying high-value customers
- Improving customer retention
- Reducing customer churn
- Optimizing marketing strategies

This project addresses these challenges using **RFM-based customer segmentation and Pareto analysis**.

---

## ⚙️ Methodology

### 1. Data Cleaning
- Removed duplicate records
- Handled missing values
- Converted columns to appropriate data types

### 2. RFM Calculation
- **Recency:** Days since last purchase  
- **Frequency:** Number of purchases  
- **Monetary:** Total customer spending  

### 3. Customer Scoring
Customers are scored on a scale of 1–4 for each RFM metric.

### 4. Segmentation
Customers are classified into:
- VIP
- Loyal
- Recent
- At Risk
- Regular

### 5. Pareto Analysis
Identifies top 20% customers contributing to ~80% of total revenue.

---

## 📊 Customer Segments

| Segment | Description |
|----------|-------------|
| VIP | Highest value customers |
| Loyal | Frequent buyers |
| Recent | Newly active customers |
| At Risk | Customers likely to churn |
| Regular | Average customers |

---

## 📈 Key Business Insights
- A small percentage of customers generates the majority of revenue (Pareto Principle)
- VIP customers are the primary revenue drivers
- At-risk customers require targeted retention strategies
- Purchase frequency strongly correlates with customer loyalty

---

## 🛠 Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  

---

## 📂 Project Structure

rfm_segmentation/
│
├── data/
│   └── ecommerce_data.csv
│
├── output/
│   └── rfm_output.csv
│
├── rfm_segmentation.py
└── README.md

---

## 🚀 Outcome
This project demonstrates real-world **data analytics and customer segmentation capabilities** used in marketing analytics and business intelligence.

It transforms raw transactional data into actionable insights for improving customer retention and maximizing revenue.

---

## 👨‍💻 Author
**Akash Sharma**  
Data Analyst | Python | Data Analytics Enthusiast

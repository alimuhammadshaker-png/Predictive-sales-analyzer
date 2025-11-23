# 💰 Predictive Risk & Opportunity Detector

## 🚀 Live Application


This application is live on Streamlit Community Cloud!
**👉 [https://predictive-sales-analyzer-n8wzytezagmzxgqe9mukzb.streamlit.app]** (e.g., https://predictive-sales-analyzer-n8wzytezagmzxgqe9mukzb.streamlit.app)

---

## ✨ Overview

The **Predictive Risk & Opportunity Detector** is a novel, feature-rich web tool designed for small business owners, e-commerce managers, and analysts. It goes beyond simple forecasting to provide **actionable alerts** based on future sales predictions.

* **Core Technology:** Time Series Forecasting using the **Meta Prophet** model.
* **Purpose:** To help users anticipate sales surges (Opportunities) or potential dips (Risks) well in advance.

## 🎯 Key Features

* **Actionable Alerts:** Generates clear, high-level alerts if the next 7 days' forecast shows a significant deviation from the recent 30-day average.
* **Universal Holiday Support (New!):** Users can integrate **built-in country holidays** (US, UK, DE, etc.) or **upload a custom CSV file** of holidays/events for *any* global market.
* **External Factors:** Easily incorporate external regressors like **Marketing Spend** or seasonal promotions to improve model accuracy.
* **Detailed Visualization:** Provides the full forecast plot and separate model component plots (Trend, Seasonality, Holidays) for full transparency.

## 💻 Tech Stack

* **Frontend/Web App:** [Streamlit](https://streamlit.io/)
* **Forecasting Model:** [Meta Prophet](https://facebook.github.io/prophet/)
* **Language:** Python
* **Data Handling:** Pandas, NumPy

# E-Commerce Customer Churn Prediction

## 1. Project Overview

### Problem Statement
The primary objective of this project is to develop a machine learning model capable of **predicting which e-commerce customers are likely to churn**. Identifying these high-risk customers will enable the business to implement proactive retention strategies, such as personalized offers, improved customer support, or targeted marketing campaigns.

### Business Value
This project offers significant business value by directly impacting key performance indicators (KPIs).
- **Increased Customer Retention:** By identifying customers at risk, the company can actively work to keep them, thereby increasing customer lifetime value.
- **Optimized Marketing Spend:** Resources can be focused on customers who are most likely to churn, improving the return on investment for retention efforts.
- **Actionable Insights:** The model's findings can reveal key behavioral patterns that correlate with churn, providing insights for overall business strategy and product development.

***

## 2. Data Sources & Acquisition

### Data Source
The project will use the **Online Retail Dataset**(https://www.kaggle.com/datasets/vijayuv/onlineretail/data) from Kaggle, a widely used and publicly available dataset for this type of problem.

### Data Description
The dataset contains transactional data, including:
- `InvoiceNo`: Invoice number.
- `StockCode`: Product code.
- `Description`: Product name.
- `Quantity`: Number of units sold.
- `InvoiceDate`: Date and time of the transaction.
- `UnitPrice`: Product price per unit.
- `CustomerID`: Unique identifier for each customer.
- `Country`: Country where the customer resides.

### Data Acquisition Method
The dataset will be downloaded as a `.csv` file and will be the single source of truth for the project.

***

## 3. Data Engineering & Feature Creation

The data engineering phase is crucial for transforming raw transactional data into meaningful features. A clear data pipeline will be developed to ensure a repeatable process.

### Churn Definition
A customer will be labeled as "churned" if they have not made a purchase within the last **90 days** relative to the most recent transaction date in the dataset. This will serve as the binary target variable for the model.

### Feature Engineering
The following features will be engineered from the raw data:

- **RFM (Recency, Frequency, Monetary) Metrics:**
    - **Recency:** The number of days since a customer's last purchase.
    - **Frequency:** The total number of unique purchases made by a customer.
    - **Monetary:** The total amount of money a customer has spent.

- **Time-Based Features:**
    - **Customer Lifetime:** The number of days between a customer's first and last purchase.
    - **Average Inter-Purchase Time:** The average number of days between consecutive purchases.

- **Behavioral Features:**
    - **Average Order Value:** The average monetary value of a customer's orders.
    - **Unique Products:** The total number of unique products a customer has purchased.

***

## 4. Modeling & Machine Learning

### Model Selection
- **Baseline Model:** A **Logistic Regression** model will be used to provide a performance benchmark.
- **Primary Model:** A **Gradient Boosting** algorithm, such as **LightGBM** or **XGBoost**, will be trained. These models are well-suited for structured data, are computationally efficient, and generally provide high performance.

### Evaluation Metrics
Since churn datasets are typically imbalanced, standard accuracy is not a reliable metric. The following metrics will be prioritized:
- **Precision:** Of all customers predicted to churn, what percentage actually churned?
- **Recall:** Of all customers who actually churned, what percentage did the model correctly identify?
- **F1-Score:** The harmonic mean of precision and recall.
- **ROC-AUC:** A measure of the model's ability to discriminate between the two classes.

***

## 5. Deployment & Infrastructure

### MLOps and Model Serving
The final, trained model will be exposed as a **RESTful API** using the **FastAPI** framework. This will allow for real-time predictions and integration with other applications.

### Technical Stack
The entire project will leverage **free and open-source tools** to ensure accessibility and reproducibility.
- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **ETL/Orchestration:** Dagster or Apache Airflow (running in Docker containers)
- **Machine Learning:** Scikit-learn, LightGBM, XGBoost
- **API Framework:** FastAPI
- **Containerization:** Docker
- **Demonstration:** Streamlit or Gradio for a simple web interface.

### Project Architecture
The project will follow a micro-service architecture with a clear data flow: 

[Image of machine learning workflow]

1.  **Data Ingestion:** Raw `.csv` data is loaded.
2.  **Data Processing:** Python scripts transform the data into a feature set.
3.  **Model Training:** The processed data is used to train the selected model.
4.  **Model Serving:** The trained model is saved and deployed as an API.
5.  **Prediction:** The API takes new customer data and returns a churn prediction.

***

## 6. Project Deliverables

- **GitHub Repository:** A public repository containing all code, a `README.md`, and this design document.
- **Jupyter Notebooks:** Notebooks detailing the EDA, feature engineering, and modeling process.
- **API Demo:** A live, functional API or web app showcasing the model's predictions.
- **Documentation:** A blog post or detailed `README.md` explaining the project from end-to-end, including challenges and key findings.
# Telecom Customer Churn Prediction & Retention Strategy

## Table of Contents
* [Business Problem](#business-problem)
* [Data Source](#data-source)
* [Project Workflow](#project-workflow)
* [Key Findings from EDA](#key-findings-from-exploratory-data-analysis)
* [Model Selection & Performance](#model-selection--performance)
* [Final Model & Business Recommendations](#final-model--business-recommendations)
* [Final Model Assets](#final-model-assets)
* [How to Run This Project](#how-to-run-this-project)
* [Tools & Libraries Used](#tools--libraries-used)

---

## Business Problem
For subscription-based businesses like telecom companies, customer retention is a primary driver of long-term profitability. Acquiring new customers is significantly more expensive than retaining existing ones. This project addresses the critical business problem of **customer churn**.

The goal is not only to build a machine learning model that can accurately predict *which* customers are likely to churn, but also to uncover the key *drivers* behind their decision to leave. By understanding the "why," the business can move from a reactive to a proactive retention strategy, targeting at-risk customers with specific, data-driven interventions to improve loyalty and reduce revenue loss.

---

## Data Source
The dataset used for this project is the **IBM Telco Customer Churn** dataset, sourced from Kaggle. It contains data on 7,043 customers, with 21 features describing their demographics, subscribed services, and account information.

---

## Project Workflow
This project followed a structured, end-to-end machine learning workflow to ensure robust and reliable results:

1.  **Data Cleaning & Preprocessing:** Handled missing values using median imputation and prepared the data for analysis.
2.  **Exploratory Data Analysis (EDA):** Performed in-depth visual analysis to uncover patterns, relationships, and key factors influencing customer churn.
3.  **Feature Engineering & Scaling:** Converted categorical features into a numerical format using one-hot encoding and scaled numerical features to prevent model bias.
4.  **Handling Class Imbalance:** Addressed the significant class imbalance in the dataset using the **SMOTE (Synthetic Minority Over-sampling Technique)** to ensure the model could effectively learn the patterns of the minority "Churn" class.
5.  **Comparative Model Building:** Trained and evaluated three different baseline models (Logistic Regression, Random Forest, XGBoost) to compare their performance.
6.  **Hyperparameter Tuning:** Systematically optimized all three models using `GridSearchCV` to find the best possible settings for each, with a focus on maximizing the **Recall** score.
7.  **Ensemble Modeling:** Built a final `VotingClassifier` ensemble to test if combining the tuned models could yield superior performance.
8.  **Final Model Selection & Recommendations:** Analyzed the performance of all models and selected the ultimate champion based on the primary business objective.

---

## Key Findings from Exploratory Data Analysis
Our visual analysis revealed several powerful insights into the drivers of churn:

*   **Contract Type is the Strongest Predictor:** Customers with **Month-to-Month contracts** churn at a dramatically higher rate than those on one or two-year contracts.
*   **New Customers are a High-Risk Segment:** A majority of churn occurs within the first 12 months of a customer's tenure.
*   **Service Pain Points:** Customers with **Fiber optic** internet service and those **without Tech Support** are significantly more likely to leave.
*   **Payment Friction:** Customers who pay via **Electronic check** churn far more often than those using automatic payment methods.

---

## Model Selection & Performance
After a comprehensive comparison and tuning process, a surprising and insightful result emerged. The final performance of our top three tuned models and the ensemble on the key metric of **Churn Recall** was as follows:

| Model                       | Churn Recall | Churn Precision | Churn F1-Score |
|-----------------------------|--------------|-----------------|----------------|
| **Tuned Logistic Regression** | **0.75**     | 0.53            | 0.62           |
| Tuned XGBoost               | 0.71         | 0.54            | 0.61           |
| Final Ensemble Model        | 0.71         | 0.55            | 0.62           |
| Tuned Random Forest         | 0.66         | 0.55            | 0.60           |

**The analysis concluded that the Tuned Logistic Regression model was the optimal solution.** Despite being the simplest model, its ability to capture the strong linear relationships in the data, combined with our robust preprocessing, allowed it to outperform more complex models.

---

## Final Model & Business Recommendations
Our final recommended model, the **Tuned Logistic Regression**, can successfully identify **75%** of all customers who are at risk of churning. This high recall rate is ideal for a proactive retention strategy.

Based on the model's insights, we recommend the following actions:

1.  **Deploy the Model for Proactive Outreach:** Use the model to generate weekly lists of at-risk customers and target them with tailored retention offers.
2.  **Rethink the Month-to-Month Offering:** Create incentive programs to encourage month-to-month customers to switch to longer-term contracts.
3.  **Invest in New Customer Onboarding:** Develop a "First 90 Days" program to support new customers and address early-stage issues.
4.  **Audit High-Churn Services:** Investigate and resolve the underlying problems with the Fiber optic service and promote the value of the Tech Support add-on.

---

## Final Model Assets
The final, trained model and the data scaler have been saved as `.joblib` files in this repository:
*   `churn_prediction_model.joblib`: The final Tuned Logistic Regression model object.
*   `data_scaler.joblib`: The `StandardScaler` object fitted on the training data.

These can be loaded into any script or application to make predictions on new, unseen customer data.

---

## How to Run This Project
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/imSyedMoinUddin/Telecom-Churn-Prediction.git
    ```
2.  **Navigate to the Directory:**
    ```bash
    cd Telecom-Churn-Prediction
    ```
3.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open the `.ipynb` file to view and run the project.

---

## Tools & Libraries Used
*   **Python 3.12.10**
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical operations.
*   **Matplotlib & Seaborn:** For data visualization.
*   **Scikit-learn:** For data preprocessing, modeling, and evaluation.
*   **Imbalanced-learn:** For handling class imbalance with SMOTE.
*   **XGBoost:** For building the Gradient Boosting model.
*   **Joblib:** For saving the final model.
*   **Jupyter Notebook:** As the development environment.

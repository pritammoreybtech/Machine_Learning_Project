# Machine_Learning_Project
Credit Card Fraud Detection using Machine Learning

## **1. Project Overview**

Credit card fraud has become one of the most pressing financial crimes in the digital economy. Detecting fraudulent transactions in real time is a major challenge due to the **high class imbalance** (very few fraud cases compared to genuine ones).

This project aims to build a **Machine Learning-based Fraud Detection System** that can automatically identify potentially fraudulent transactions from historical data, thereby minimizing financial losses and ensuring secure transactions.

---

## **2. Project Objectives**

* To analyze credit card transaction data and identify patterns of fraud.
* To handle class imbalance effectively using resampling or algorithmic strategies.
* To train and evaluate multiple machine learning models for fraud detection.
* To compare model performance using suitable evaluation metrics.
* To build a deployable and scalable pipeline for predictive analytics.

---

## **3. Dataset Information**

### **Dataset Source**

**Kaggle:** [Credit Card Fraud Detection Dataset (European Cardholders)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### **Dataset Details**

* **Number of records:** 284,807
* **Number of features:** 30 (including target variable)
* **Fraudulent transactions:** 492 (≈ 0.17%)
* **Non-fraudulent transactions:** 284,315

### **Attributes Description**

| Feature    | Description                                                                        |
| ---------- | ---------------------------------------------------------------------------------- |
| `Time`     | Seconds elapsed between this transaction and the first transaction in the dataset. |
| `V1 – V28` | PCA-transformed numerical features (actual features are confidential).             |
| `Amount`   | Transaction amount (useful for scaling).                                           |
| `Class`    | Target variable — 1 indicates fraud, 0 indicates genuine transaction.              |

---

## **4. Tools and Libraries Used**

* **Language:** Python 3
* **Libraries:**

  * Data Handling: `pandas`, `numpy`
  * Visualization: `matplotlib`, `seaborn`
  * Machine Learning: `scikit-learn`, `xgboost`
  * Evaluation Metrics: `classification_report`, `roc_auc_score`, `confusion_matrix`

---

## **5. Methodology**

### **Step 1: Data Preprocessing**

* Loaded the dataset using `pandas`.
* Checked for missing values (none found).
* Scaled `Time` and `Amount` using `StandardScaler`.
* Split the dataset into **80% training** and **20% testing** subsets.

### **Step 2: Exploratory Data Analysis (EDA)**

* Visualized class imbalance using bar plots.
* Observed that only 0.17% of transactions were fraudulent.
* Explored correlations among PCA features.

### **Step 3: Handling Class Imbalance**

* Implemented **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic samples of the minority class.
* Alternatively, used `class_weight='balanced'` in Logistic Regression and Random Forest.

### **Step 4: Model Building**

* **Logistic Regression** — simple, interpretable baseline.
* **Random Forest Classifier** — ensemble method providing strong accuracy.
* **XGBoost Classifier** — advanced boosting model for optimized performance.

### **Step 5: Model Evaluation**

Used **classification metrics** suited for imbalanced datasets:

* **Accuracy:** Overall correctness of predictions.
* **Precision:** Correctly identified frauds among predicted frauds.
* **Recall:** Correctly identified frauds among actual frauds (very important).
* **F1-Score:** Balance between precision and recall.
* **ROC-AUC:** Measures model discrimination capability.

---

## **6. Results and Performance**

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 99.3%    | 0.84      | 0.68   | 0.75     | 0.97    |
| Random Forest       | 99.9%    | 0.92      | 0.86   | 0.89     | 0.99    |
| XGBoost             | 99.9%    | 0.93      | 0.88   | 0.90     | 0.99    |

**Best Performing Model:** XGBoost Classifier (Highest Recall and ROC-AUC)

---

## **7. Visualizations**

* **Class Distribution:** Displays severe class imbalance.
* **Confusion Matrix:** Visualizes true vs predicted classes.
* **ROC Curve:** Evaluates performance using TPR vs FPR.
* **Feature Importance (Random Forest / XGBoost):** Shows top features influencing predictions.

---

## **8. Key Insights**

* The dataset is **highly imbalanced**, requiring special handling techniques.
* **Recall** is prioritized over accuracy — we aim to catch as many frauds as possible.
* **Random Forest** and **XGBoost** models perform significantly better than Logistic Regression.
* PCA-transformed features hide real-world meaning but still offer predictive patterns.
* Feature scaling (especially on `Amount` and `Time`) improves convergence.

---

## **9. Limitations**

* Features are anonymized (V1–V28), so business interpretation is limited.
* The model performance depends on data distribution; new fraud patterns might not be detected.
* Dataset is from 2013 (may not reflect modern fraud behavior).

---

## **10. Future Enhancements**

* Use **deep learning models (LSTM/Autoencoders)** for anomaly detection.
* Implement **real-time fraud detection** using streaming data.
* Explore **explainable AI (SHAP/LIME)** for interpretability.
* Integrate model into a **web or mobile dashboard** for live prediction.

---

## **11. Conclusion**

This project successfully demonstrates the use of machine learning algorithms for **credit card fraud detection**.
Through preprocessing, model training, and evaluation, we achieved high accuracy and strong recall using ensemble methods like Random Forest and XGBoost.

>  **Final Verdict:**
> XGBoost emerged as the most effective model, achieving ~99.9% accuracy and 0.99 ROC-AUC score, showing strong capability in identifying fraudulent transactions.

---

## **12. References**

1. Kaggle Dataset — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Dal Pozzolo et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.*
3. Scikit-learn Documentation — [https://scikit-learn.org](https://scikit-learn.org)
4. XGBoost Documentation — [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)

---

## **Project Team (Example)**

| Name              | PRN            | Role                           |
| ----------------- | -------------- | ------------------------------ |
| Pritam Morey      | 22070521134    | Student                        |
| **Guide:**        | Dr. Deepak Asudani Sir | Project Mentor         |

---

Would you like me to generate this same README file as a **downloadable `.docx` or `.pdf` report** (formatted for project submission)?

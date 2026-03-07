# 🏎️ Formula 1 Podium Prediction using Machine Learning

## 📌 Overview
This project builds an end-to-end **Machine Learning pipeline** to predict whether a Formula 1 driver will finish on the **podium (Top 3)** in a race.  
The model uses historical race data collected from the **Ergast F1 API** and applies multiple regression and classification algorithms to analyze driver and constructor performance.

The project demonstrates the complete workflow of a real-world ML system including:

- API data collection  
- Data preprocessing and feature engineering  
- Training multiple ML models  
- Model evaluation and comparison  
- Data visualization

---

# 🎯 Project Objectives

The main objectives of this project are:

- Collect historical Formula 1 race data using the **Ergast F1 API**
- Build a machine learning dataset from the API response
- Engineer useful racing features
- Train multiple ML models
- Predict whether a driver will finish in the **Top 3 (podium)**
- Predict **finishing position** using regression models
- Evaluate model performance using standard ML metrics

---

# 📊 Dataset

The dataset is collected dynamically using the **Ergast Developer API**, which provides historical Formula 1 race data.

### API Endpoint Example
https://api.jolpi.ca/ergast/f1/


The collected dataset includes information such as:

- Season
- Race name
- Circuit
- Driver name
- Constructor (team)
- Grid position
- Finishing position
- Points scored
- Number of laps

This data is converted into a structured dataset using **Python and Pandas**.

---

# ⚙️ Feature Engineering

Several features were created to improve the predictive power of the models.

### Driver Performance
Average finishing position of a driver across races.
driver_avg_fin


### Constructor Performance
Average finishing position of the team.
constructor_avg_finish


### Grid Position
Starting position of the driver in the race.
grid


---

# 🎯 Target Variables

Two prediction tasks were implemented.

### 1️⃣ Classification Task
Predict whether a driver finishes on the **podium (Top 3)**.
podium = 1 if position <= 3
podium = 0 otherwise


### 2️⃣ Regression Task
Predict the **exact finishing position** of the driver.
position = 1 to 20


---

# 🤖 Machine Learning Models

The following models were implemented using **Scikit-learn**.

### Classification Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Regression Models
- Linear Regression
- Polynomial Regression

---

# 📈 Model Evaluation

Different metrics were used for evaluating classification and regression models.

## Classification Metrics

- Accuracy
- F1 Score

Example Results:

| Model | Accuracy |
|------|------|
| Logistic Regression | ~92% |
| Decision Tree | ~88% |
| Random Forest | ~91% |

Random Forest achieved an **F1 Score ≈ 0.81**, indicating strong performance in identifying podium finishers.

---

## Regression Metrics

Regression models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Example Results:

| Metric | Value |
|------|------|
| MAE | ~10 |
| RMSE | ~11 |

These results indicate that predicting exact finishing positions is significantly harder than predicting podium finishes.

---

# 🧠 Key Insights

- **Grid position is one of the strongest predictors of race results.**
- Drivers starting near the front have a significantly higher chance of finishing on the podium.
- Constructor performance also strongly influences race outcomes.
- Random Forest performs better than simple linear models due to its ability to capture nonlinear relationships.

---

# 🚀 Technologies Used

- Python  
- Pandas  
- NumPy  
- Requests  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

More advanced models such as **Gradient Boosting, XGBoost, or Neural Networks** could also improve predictive performance.

---

# 📌 Conclusion

This project demonstrates how machine learning can be applied to motorsport analytics by combining **API data collection, feature engineering, and predictive modeling**.

The **Random Forest classifier** achieved strong performance in predicting podium finishes, making this a practical and realistic machine learning application in sports analytics.

---

# 📚 References

Ergast Developer API  
https://ergast.com/mrd/

Scikit-learn Documentation  
https://scikit-learn.org/

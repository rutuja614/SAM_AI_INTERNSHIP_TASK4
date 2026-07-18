# 🍽️ Task 4: Restaurant Rating Prediction

## 📌 Project Overview

This project is part of my **Data Analyst Internship at SAM AI Technologies**.

The objective of this task is to build a **Machine Learning Regression Model** that predicts restaurant ratings using various restaurant features such as votes, price range, online delivery availability, and cuisines.

The project includes data preprocessing, feature engineering, model training, evaluation, and model saving.

---

# 🎯 Objectives

- Predict restaurant ratings using Machine Learning.
- Perform data preprocessing and feature engineering.
- Train a Random Forest Regression model.
- Evaluate the model using regression metrics.
- Save the trained model for future predictions.

---

# 📂 Dataset

**Dataset Used:** Zomato Restaurant Dataset

The dataset contains information such as:

- Restaurant Name
- City
- Cuisines
- Votes
- Price Range
- Online Delivery
- Aggregate Rating
- Average Cost for Two
- Currency
- Table Booking
- And other restaurant details.

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- VS Code

---

# 📚 Machine Learning Workflow

## 1. Import Libraries

Imported the required Python libraries:

- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## 2. Load Dataset

Loaded the Zomato dataset using Pandas.

```python
df = pd.read_csv("zomato.csv", encoding="latin1")
```

---

## 3. Data Preprocessing

Selected the following features:

- Votes
- Price Range
- Has Online Delivery
- Cuisines

Target Variable:

- Aggregate Rating

Performed:

- Missing value removal
- Label Encoding
- Cuisine preprocessing
- Feature selection

---

## 4. Feature Engineering

Converted:

- Has Online Delivery

into numerical values.

Extracted only the primary cuisine from the Cuisines column and encoded it using LabelEncoder.

---

## 5. Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

using:

```python
train_test_split()
```

---

## 6. Model Training

Machine Learning Algorithm Used:

**Random Forest Regressor**

The model was trained using the training dataset.

---

## 7. Model Prediction

The trained model predicted restaurant ratings on the testing dataset.

---

## 8. Model Evaluation

The model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 📊 Model Performance

| Metric | Value |
|---------|--------|
| Mean Absolute Error (MAE) | **0.239** |
| Root Mean Squared Error (RMSE) | **0.363** |
| R² Score | **0.942** |

### Interpretation

- Low MAE indicates high prediction accuracy.
- Low RMSE shows minimal prediction error.
- R² Score of **94.2%** indicates that the model explains most of the variance in restaurant ratings.

---



# 📦 Output

The project generates:

- Restaurant Rating Predictions
- Model Evaluation Metrics
- Saved Machine Learning Model

Generated file:

```
restaurant_rating_model.pkl
```

---

# 📈 Learning Outcomes

Through this project, I learned:

- Data Cleaning
- Feature Engineering
- Label Encoding
- Regression Models
- Random Forest Regression
- Model Evaluation
- Machine Learning Workflow
- Model Serialization using Joblib

---

# 🚀 Future Improvements

- Try XGBoost and LightGBM models.
- Hyperparameter tuning using GridSearchCV.
- Add more restaurant features.
- Deploy the model as a web application using Flask or Streamlit.

---

# 👩‍💻 Author

**Rutuja Desai**

Data Analyst Intern – SAM AI Technologies

---

# ⭐ Internship Task

**Task 4 – Restaurant Rating Prediction**

Completed as part of the **Data Analyst Internship at SAM AI Technologies**.

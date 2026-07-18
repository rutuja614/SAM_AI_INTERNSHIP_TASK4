import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

print("="*60)
print("Restaurant Rating Prediction")
print("="*60)

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("zomato.csv", encoding="latin1")

print("\nDataset Loaded Successfully")
print(df.head())

# ----------------------------
# Select Features
# ----------------------------

data = df[[
    "Votes",
    "Price range",
    "Has Online delivery",
    "Cuisines",
    "Aggregate rating"
]]

print("\nSelected Columns")
print(data.head())

# ----------------------------
# Remove Missing Values
# ----------------------------

data = data.dropna()

# ----------------------------
# Encode Online Delivery
# ----------------------------

delivery_encoder = LabelEncoder()

data["Has Online delivery"] = delivery_encoder.fit_transform(
    data["Has Online delivery"]
)

# ----------------------------
# Take Only Main Cuisine
# ----------------------------

data["Main Cuisine"] = data["Cuisines"].apply(
    lambda x: x.split(",")[0]
)

data.drop("Cuisines", axis=1, inplace=True)

# ----------------------------
# Encode Cuisine
# ----------------------------

cuisine_encoder = LabelEncoder()

data["Main Cuisine"] = cuisine_encoder.fit_transform(
    data["Main Cuisine"]
)

# ----------------------------
# Features and Target
# ----------------------------

X = data.drop("Aggregate rating", axis=1)

y = data["Aggregate rating"]

print("\nFeatures Ready")
print(X.head())

# ----------------------------
# Train Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42

)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ----------------------------
# Build Model
# ----------------------------

print("\nTraining Random Forest Model...")

model = RandomForestRegressor(

    n_estimators=100,

    random_state=42

)

model.fit(

    X_train,

    y_train

)

print("Model Training Completed")

# ----------------------------
# Prediction
# ----------------------------

y_pred = model.predict(X_test)

# ----------------------------
# Evaluation
# ----------------------------

mae = mean_absolute_error(

    y_test,

    y_pred

)

rmse = np.sqrt(

    mean_squared_error(

        y_test,

        y_pred

    )

)

r2 = r2_score(

    y_test,

    y_pred

)

print("\nModel Evaluation")

print("-"*40)

print("Mean Absolute Error :", round(mae,3))

print("Root Mean Squared Error :", round(rmse,3))

print("R2 Score :", round(r2,3))

# ----------------------------
# Save Model
# ----------------------------

joblib.dump(

    model,

    "restaurant_rating_model.pkl"

)

print("\nModel Saved Successfully")

print("\nTask 4 Completed Successfully")
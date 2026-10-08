import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# ==========================================
# 1. Load dataset
# ==========================================

data = pd.read_csv("used_car_price_regression_dataset.csv")

data.drop(columns=["Year"], inplace=True)


# ==========================================
# 2. Encode Transmission
# ==========================================

#data["Transmission"] = (
 #   data["Transmission"]
 #   .apply(lambda x: 1 if x == "Automatic" else 0)
  #  .astype(int)
#)


# ==========================================
# 3. Separate X and y
# ==========================================

X = data.drop(columns=["Price"])
y = data["Price"]


# ==========================================
# 4. Categorical columns
# ==========================================

categorical_columns = [
    "Brand",
    "Fuel_Type",
    "Transmission",
    "Location"
]


# ==========================================
# 5. Create preprocessor
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 6. Create complete pipeline
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])


# ==========================================
# 7. Train-test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# 8. Train complete pipeline
# ==========================================

pipeline.fit(X_train, y_train)


# ==========================================
# 9. Prediction on test data
# ==========================================

y_pred = pipeline.predict(X_test)

print("First 5 predictions:")
print(y_pred[:5])


joblib.dump(pipeline, "car_predictor.pkl")

print("Model saved successfully!")

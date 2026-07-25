import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.ensemble import RandomForestRegressor

from .train_utils import DATA_FILE_PATH, MODEL_DIR, MODEL_PATH


# ==========================
# Load Dataset
# ==========================

data = (
    pd.read_csv(DATA_FILE_PATH)
    .drop_duplicates()
    .drop(columns=['name', 'model', 'edition'], errors='ignore')
)


# ==========================
# Split Features and Target
# ==========================

X = data.drop(columns=['selling_price'])

y = data['selling_price'].copy()


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================
# Identify Columns
# ==========================

num_cols = X_train.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()


cat_cols = [
    col for col in X_train.columns 
    if col not in num_cols
]


# ==========================
# Numerical Pipeline
# ==========================

num_pipeline = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(strategy='median')
        ),
        (
            'scaler',
            StandardScaler()
        )
    ]
)


# ==========================
# Categorical Pipeline
# ==========================

cat_pipeline = Pipeline(
    steps=[
        (
            'imputer',
            SimpleImputer(
                strategy='constant',
                fill_value='missing'
            )
        ),
        (
            'encoder',
            OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )
        )
    ]
)


# ==========================
# Column Transformer
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            num_pipeline,
            num_cols
        ),
        (
            'cat',
            cat_pipeline,
            cat_cols
        )
    ]
)


# ==========================
# Random Forest Model
# ==========================

regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42
)


# ==========================
# Complete ML Pipeline
# ==========================

model = Pipeline(
    steps=[
        (
            'preprocessor',
            preprocessor
        ),
        (
            'regressor',
            regressor
        )
    ]
)


# ==========================
# Train Model
# ==========================

model.fit(
    X_train,
    y_train
)


# ==========================
# Save Model
# ==========================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


joblib.dump(
    model,
    MODEL_PATH
)


print("✅ Model trained successfully")
print(f"✅ Model saved at: {MODEL_PATH}") 
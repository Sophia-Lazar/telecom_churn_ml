import pandas as pd
import joblib

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# load data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

data_path = BASE_DIR / "data" / "processed" / "cleaned_data.csv"

df = pd.read_csv(data_path)

# drop unnecessary columns
df=df.drop(columns=["customerID"],errors="ignore")

# split feature&target
X=df.drop("Churn",axis=1)
y=df["Churn"]

# train test split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# separete column types
num_cols= ["tenure","MonthlyCharges","TotalCharges"]
cat_cols= [col for col in X.columns if col not in num_cols]

# numeric pipeline
num_pipeline=Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])

# categorical pipeline
cat_pipeline= Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown="ignore"))
])

# combine pipelines
preprocessor= ColumnTransformer([
    ("num",num_pipeline,num_cols),
    ("cat",cat_pipeline,cat_cols)
])

# full pipeline
pipeline= Pipeline([
    ("preprocessing",preprocessor),
    ("model",RandomForestClassifier(random_state=42))
])

# hyperparameter tuning
param_grid=  {
    "model__n_estimators": [100,200],
    "model__max_depth": [10,20,None],
    "model__min_samples_split": [2,5],
    "model__min_samples_leaf": [1,2],
    "model__class_weight": ["balanced"]
}

# grid search
grid_search= GridSearchCV(
    pipeline,
    param_grid,
    scoring="recall",
    cv=3,
    n_jobs=1
)

# train model
grid_search.fit(X_train,y_train)

# save final pipeline
best_pipeline=grid_search.best_estimator_

model_dir = BASE_DIR / "models"
model_dir.mkdir(parents=True, exist_ok=True)  

model_path = model_dir / "churn_pipeline.pkl"

joblib.dump(best_pipeline, model_path)
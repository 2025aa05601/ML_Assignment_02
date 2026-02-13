import pandas as pd
import joblib
import sklearn
import xgboost
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("../data/sample_dataset.csv")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model/saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# -------------------------------
# Missing data elements
# --------------------------------

threshold = 0.30

missing_stats = pd.DataFrame({
    "Feature": df.columns,
    "Missing Count": df.isna().sum(),
    "Missing (%)": (df.isna().mean() * 100).round(2)
})

features_to_drop = missing_stats[
  missing_stats["Missing (%)"] > (threshold * 100)
 ]["Feature"].tolist()

if features_to_drop:
    df = df.drop(columns=features_to_drop)
        
# -------------------------------
# Encode for Target 
# -------------------------------
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# y is the selected target column
if y.dtype == "object" or y.dtype.name == "category":
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

else:
    y_encoded = y

#re-assign the target data
y = y_encoded

# -------------------------------
# Encode for Features 
# -------------------------------

# Separate column types
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Numeric preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Models dictionary
models = {
    "logistic": Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    
    "decision_tree": Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier())
    ]),
    
    "knn": Pipeline([
        ("preprocessor", preprocessor),
        ("model", KNeighborsClassifier())
    ]),
    
    "naive_bayes": Pipeline([
        ("preprocessor", preprocessor),
        ("model", GaussianNB())
    ]),
    
    "random_forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(n_estimators=200))
    ]),
    
    "xgboost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            eval_metric="logloss",
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1
        ))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))
print("All models trained and saved.")


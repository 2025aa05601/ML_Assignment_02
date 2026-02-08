import pandas as pd
import joblib
import sklearn
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("data/sample_dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "decision_tree": DecisionTreeClassifier(),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "xgboost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/saved_models/{name}.pkl")

print("All models trained and saved.")


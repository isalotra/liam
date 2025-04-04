import pickle
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

mlflow.log_param("n_estimators", 100)
mlflow.log_metric("accuracy", model.score(X_test, y_test))

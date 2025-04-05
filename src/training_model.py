import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Configuration de MLflow
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("Classification Iris")

# Chargement des données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    # Enregistrement des paramètres et métriques
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    
    # Sauvegarde du modèle
    mlflow.sklearn.log_model(model, "random_forest_model")

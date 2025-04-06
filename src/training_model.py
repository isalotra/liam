import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import numpy as np
import os
import tempfile

# Configuration de MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Classification Iris")

# Chargement des données
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Meilleur suivi des hyperparamètres
n_estimators_list = [50, 100, 200]
max_depth_list = [5, 10, None]

with mlflow.start_run():
    best_accuracy = 0
    best_model = None

    # Création et entraînement du modèle
    n_estimators = 40
    max_depth = 20
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=2)
    model.fit(X_train, y_train)

    # Prédictions et évaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=data.target_names)

    # Calcul de la validation croisée
    cross_val_accuracy = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))


    # Sauvegarde du rapport dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as tmpfile:
        tmpfile.write(report)
        tmpfile_path = tmpfile.name


    # Suppression du fichier temporaire après l'enregistrement
    # os.remove(tmpfile_path)

    best_model = model
    mlflow.sklearn.log_model(model, "random_forest_model")
    # Enregistrement des meilleurs paramètres et métriques
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("cross_val_accuracy", cross_val_accuracy)
    # Enregistrement du fichier de rapport en tant qu'artefact
    mlflow.log_artifact(tmpfile_path)

    # Affichage des résultats
    print(f"Best Test Accuracy: {accuracy}")
    print(f"Best Cross-Validation Accuracy: {cross_val_accuracy}")
    print(f"Best Model Classification Report:\n{report}")

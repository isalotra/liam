# Utilisation de l'image Python officielle
FROM python:3.9

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de l'application
COPY requirements.txt requirements.txt
COPY src/ src/

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour l'API
EXPOSE 8000

# Commande de démarrage
CMD ["python", "-m", "uvicorn", "src.api_model:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

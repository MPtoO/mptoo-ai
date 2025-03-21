# Utiliser une image Python officielle
FROM python:3.10-slim as builder

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements.txt
COPY requirements.txt .

# Installer les dépendances dans une couche séparée
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Image finale
FROM python:3.10-slim

# Créer un utilisateur non-root
RUN useradd -m appuser

# Définir le répertoire de travail
WORKDIR /app

# Copier les wheels et installer les dépendances
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copier le code source
COPY . .

# Donner les permissions à l'utilisateur
RUN chown -R appuser:appuser /app
USER appuser

# Créer le répertoire data avec les bonnes permissions
RUN mkdir -p /app/data && chmod 755 /app/data

# Exposer le port
EXPOSE 8000

# Commande de démarrage
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "api.wsgi:app"] 
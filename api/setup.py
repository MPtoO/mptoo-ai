"""
Script de configuration pour l'API MPtoO
Ce script prépare l'environnement pour l'API FastAPI.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
import secrets


def create_env_file():
    """Création du fichier .env avec les variables d'environnement"""
    env_path = Path("api/.env")
    
    if not env_path.exists():
        print("Création du fichier .env...")
        
        # Générer une clé secrète
        secret_key = secrets.token_hex(32)
        
        env_content = f"""# Configuration de l'API MPtoO
SECRET_KEY={secret_key}
DEBUG=true
ENVIRONMENT=development

# Configuration de la base de données
DATABASE_URL=sqlite:///./mptoo.db

# Configuration CORS
BACKEND_CORS_ORIGINS=["http://localhost:5173", "http://localhost:8000", "http://localhost:3000"]

# Configuration JWT
ACCESS_TOKEN_EXPIRE_MINUTES=1440
"""
        
        with open(env_path, "w") as f:
            f.write(env_content)
        
        print(f"Fichier .env créé avec succès à {env_path.absolute()}")
    else:
        print(f"Le fichier .env existe déjà à {env_path.absolute()}")


def main():
    """Fonction principale"""
    print("Configuration de l'API MPtoO...")
    
    # Création des dossiers nécessaires
    directories = [
        "api/core",
        "api/ai/agents",
        "api/ai/predictors",
        "api/ai/nlp",
        "api/scrapers",
        "api/indexers",
        "api/services",
        "api/models"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            print(f"Création du dossier {directory}...")
            path.mkdir(parents=True, exist_ok=True)
    
    # Créer le fichier __init__.py dans chaque dossier
    for directory in directories:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            print(f"Création du fichier {init_file}...")
            init_file.touch()
    
    # Installation des dépendances
    requirements_path = Path("api/requirements.txt")
    if requirements_path.exists():
        print("Installation des dépendances...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
            print("Dépendances installées avec succès !")
        except subprocess.CalledProcessError:
            print("Erreur lors de l'installation des dépendances.")
    else:
        print(f"Le fichier {requirements_path} n'existe pas !")
    
    # Création du fichier .env
    create_env_file()
    
    print("Configuration terminée ! Vous pouvez maintenant démarrer l'API avec:")
    print("cd api && python -m uvicorn main:app --reload")


if __name__ == "__main__":
    main() 
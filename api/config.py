#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Configuration de base
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Configuration de la base de données
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Configuration de l'authentification
SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Configuration Elasticsearch
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# Configuration IA
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "llama3")

# Configuration CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",") 
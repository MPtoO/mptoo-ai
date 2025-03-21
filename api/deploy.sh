#!/bin/bash

# Script de déploiement pour l'API MPtoO
# Usage: ./deploy.sh [dev|prod]

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Fonctions pour les messages
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration de l'environnement
setup_env() {
  ENV_FILE=".env"
  
  if [ "$1" == "prod" ]; then
    info "Configuration pour production..."
    
    # Générer une clé secrète
    SECRET_KEY=$(openssl rand -hex 32)
    
    # Créer le fichier .env
    cat > $ENV_FILE << EOF
# Configuration de l'API MPtoO
SECRET_KEY=${SECRET_KEY}
DEBUG=false
ENVIRONMENT=production

# Configuration de la base de données
DATABASE_URL=sqlite:///./data/mptoo.db

# Configuration CORS
BACKEND_CORS_ORIGINS=["http://localhost", "http://localhost:80", "http://127.0.0.1", "http://127.0.0.1:80"]

# Configuration JWT
ACCESS_TOKEN_EXPIRE_MINUTES=1440
EOF
    
    success "Fichier .env créé pour la production"
  else
    info "Configuration pour développement..."
    
    # Créer le fichier .env
    cat > $ENV_FILE << EOF
# Configuration de l'API MPtoO
SECRET_KEY=dev_secret_key_change_in_production
DEBUG=true
ENVIRONMENT=development

# Configuration de la base de données
DATABASE_URL=sqlite:///./data/mptoo.db

# Configuration CORS
BACKEND_CORS_ORIGINS=["http://localhost:5173", "http://localhost:8000", "http://localhost:3000", "http://localhost:8080"]

# Configuration JWT
ACCESS_TOKEN_EXPIRE_MINUTES=1440
EOF
    
    success "Fichier .env créé pour le développement"
  fi
}

# Création des dossiers nécessaires
setup_directories() {
  info "Création des dossiers nécessaires..."
  
  # Liste des dossiers à créer
  directories=(
    "core"
    "ai/agents"
    "ai/predictors"
    "ai/nlp"
    "scrapers"
    "indexers"
    "services"
    "models"
    "data"
  )
  
  # Créer les dossiers et les fichiers __init__.py
  for dir in "${directories[@]}"; do
    mkdir -p $dir
    touch "$dir/__init__.py"
  done
  
  success "Dossiers et fichiers __init__.py créés"
}

# Installation des dépendances
install_dependencies() {
  info "Installation des dépendances..."
  
  if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    success "Dépendances installées"
  else
    error "Fichier requirements.txt introuvable"
    exit 1
  fi
}

# Initialisation des données pour le développement
setup_mock_data() {
  info "Configuration des données de test..."
  
  # Vérifier si le dossier data existe
  if [ ! -d "data" ]; then
    mkdir -p data
  fi
  
  # Créer des fichiers de données de test s'ils n'existent pas déjà
  if [ ! -f "data/mock_analysis.json" ]; then
    cat > data/README.md << EOF
# Données de test pour MPtoO

Ce dossier contient des données de test utilisées en mode développement:

- mock_analysis.json: Exemples d'analyses
- mock_predictions.json: Exemples de prédictions

Ces fichiers sont utilisés par les endpoints de l'API en mode développement.
EOF
    
    cat > data/mock_analysis.json << EOF
{
  "analyses": [
    {
      "id": "ana-001",
      "timestamp": "2023-06-15T09:30:00Z",
      "source": "web_scraping",
      "market": "automobile",
      "sector": "electric_vehicles",
      "sentiment": 0.78,
      "keywords": ["véhicule électrique", "batterie", "autonomie", "recharge rapide"],
      "summary": "Analyse positive du marché des véhicules électriques avec une croissance attendue de 25% sur l'année à venir."
    },
    {
      "id": "ana-002",
      "timestamp": "2023-06-16T14:45:00Z",
      "source": "financial_data",
      "market": "technologie",
      "sector": "semiconducteurs",
      "sentiment": 0.45,
      "keywords": ["puce", "processeur", "silicium", "pénurie"],
      "summary": "Analyse mitigée du secteur des semiconducteurs avec des problèmes persistants de chaîne d'approvisionnement."
    },
    {
      "id": "ana-003",
      "timestamp": "2023-06-17T11:15:00Z",
      "source": "social_media",
      "market": "retail",
      "sector": "mode",
      "sentiment": 0.92,
      "keywords": ["écoresponsable", "durable", "recyclé", "éthique"],
      "summary": "Très forte tendance pour les produits de mode écoresponsables avec un engouement marqué sur les réseaux sociaux."
    }
  ]
}
EOF
    
    cat > data/mock_predictions.json << EOF
{
  "predictions": [
    {
      "id": "pred-001",
      "timestamp": "2023-06-15T10:00:00Z",
      "market": "automobile",
      "sector": "electric_vehicles",
      "horizon": "6_months",
      "confidence": 0.85,
      "trend_direction": "up",
      "trend_strength": 0.75,
      "price_prediction": {
        "current": 145.50,
        "predicted": 189.25,
        "change_percent": 30.1
      },
      "factors": [
        {
          "name": "Politique gouvernementale",
          "impact": 0.8,
          "description": "Nouvelles subventions pour véhicules électriques"
        },
        {
          "name": "Innovation technologique",
          "impact": 0.65,
          "description": "Amélioration de l'autonomie des batteries"
        },
        {
          "name": "Demande consommateurs",
          "impact": 0.7,
          "description": "Hausse de la demande pour transport écologique"
        }
      ]
    },
    {
      "id": "pred-002",
      "timestamp": "2023-06-16T15:30:00Z",
      "market": "technologie",
      "sector": "semiconducteurs",
      "horizon": "3_months",
      "confidence": 0.72,
      "trend_direction": "stable",
      "trend_strength": 0.25,
      "price_prediction": {
        "current": 223.75,
        "predicted": 230.50,
        "change_percent": 3.02
      },
      "factors": [
        {
          "name": "Chaîne d'approvisionnement",
          "impact": 0.6,
          "description": "Amélioration lente des problèmes d'approvisionnement"
        },
        {
          "name": "Demande industrielle",
          "impact": 0.5,
          "description": "Demande stable de l'industrie"
        }
      ]
    }
  ]
}
EOF
    
    success "Données de test créées"
  else
    info "Les données de test existent déjà"
  fi
}

# Fonction principale
main() {
  if [ "$1" == "prod" ]; then
    info "Déploiement de l'API en mode production"
    setup_env "prod"
    setup_directories
    install_dependencies
    success "API MPtoO prête pour la production!"
    info "Démarrez avec 'uvicorn main:app --host 0.0.0.0 --port 8001'"
  else
    info "Configuration de l'API en mode développement"
    setup_env "dev"
    setup_directories
    install_dependencies
    setup_mock_data
    success "API MPtoO prête pour le développement!"
    info "Démarrez avec 'uvicorn main:app --reload --host 0.0.0.0 --port 8001'"
  fi
}

# Exécuter le script
main "$1" 
#!/bin/bash

# Script d'installation MPTOO
# Copyright 2025 Mohammed Amine Taybi
# Licensed under the Apache License, Version 2.0

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Installation de MPTOO ===${NC}"
echo

# Vérification des prérequis
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

if ! command -v pip &> /dev/null; then
    echo -e "${RED}pip n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm n'est pas installé. Veuillez l'installer avant de continuer.${NC}"
    exit 1
fi

# Création de la structure de répertoires
echo -e "${YELLOW}Création de la structure de répertoires...${NC}"
mkdir -p backend/data/models
mkdir -p backend/data/uploads
mkdir -p logs
mkdir -p frontend
mkdir -p docs
mkdir -p scripts
mkdir -p backups

# Configuration de l'environnement
echo -e "${YELLOW}Configuration de l'environnement...${NC}"
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}Fichier .env créé à partir de .env.example${NC}"
        echo -e "${YELLOW}⚠️ N'oubliez pas de modifier le fichier .env avec vos propres valeurs${NC}"
    else
        echo -e "${RED}Fichier .env.example non trouvé. Impossible de créer le fichier .env${NC}"
        exit 1
    fi
fi

# Installation des dépendances backend
echo -e "${YELLOW}Installation des dépendances backend...${NC}"
cd backend || exit 1

# Création de l'environnement virtuel Python
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Création de l'environnement virtuel Python...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Environnement virtuel Python créé${NC}"
fi

# Activation de l'environnement virtuel et installation des dépendances
echo -e "${YELLOW}Installation des dépendances Python...${NC}"
source venv/bin/activate
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dépendances Python installées avec succès${NC}"
else
    echo -e "${RED}Fichier requirements.txt non trouvé${NC}"
    echo -e "${YELLOW}Création d'un fichier requirements.txt minimal...${NC}"
    echo -e "fastapi>=0.95.0\nuvicorn>=0.22.0\npydantic>=2.0.0\nsqlalchemy>=2.0.0\npsycopg2-binary>=2.9.0\npython-dotenv>=1.0.0\npython-jose>=3.3.0\npasslib>=1.7.4\npython-multipart>=0.0.6\nemail-validator>=2.0.0\nredis>=4.5.0\npandas>=2.0.0\npyjwt>=2.6.0\nalembic>=1.11.0" > requirements.txt
    pip install -r requirements.txt
    echo -e "${GREEN}Dépendances Python de base installées avec succès${NC}"
fi
deactivate

cd .. || exit 1

# Installation des dépendances frontend
echo -e "${YELLOW}Installation des dépendances frontend...${NC}"
cd frontend || exit 1

if [ -f "package.json" ]; then
    echo -e "${YELLOW}Installation des dépendances Node.js...${NC}"
    npm install
    echo -e "${GREEN}Dépendances Node.js installées avec succès${NC}"
else
    echo -e "${RED}Fichier package.json non trouvé${NC}"
    echo -e "${YELLOW}Voulez-vous initialiser un nouveau projet Vue.js? [y/N]${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${YELLOW}Initialisation d'un nouveau projet Vue.js...${NC}"
        npm init -y
        npm install vue@next vue-router@next pinia axios
        npm install -D vite @vitejs/plugin-vue typescript vue-tsc
        echo -e "${GREEN}Projet Vue.js initialisé avec succès${NC}"
    else
        echo -e "${YELLOW}Création d'un package.json minimal...${NC}"
        echo '{
  "name": "mptoo-frontend",
  "version": "1.0.0",
  "description": "Frontend pour MPTOO",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "serve": "vite preview"
  },
  "dependencies": {
    "axios": "^1.4.0",
    "pinia": "^2.1.0",
    "vue": "^3.3.0",
    "vue-router": "^4.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-vue": "^4.2.0",
    "typescript": "^5.0.0",
    "vite": "^4.3.0",
    "vue-tsc": "^1.6.0"
  }
}' > package.json
        npm install
        echo -e "${GREEN}Dépendances Node.js de base installées avec succès${NC}"
    fi
fi

cd .. || exit 1

# Rendre les scripts exécutables
echo -e "${YELLOW}Configuration des permissions...${NC}"
chmod +x scripts/*.sh

# Résumé
echo
echo -e "${GREEN}=== Installation terminée ===${NC}"
echo
echo -e "Documentation: ${YELLOW}https://github.com/MPtoO${NC}"
echo -e "Communauté Discord: ${YELLOW}https://discord.gg/Mmj6xyUr${NC}"
echo
echo -e "Pour lancer l'application en développement:"
echo -e "1. Backend: ${YELLOW}cd backend && source venv/bin/activate && uvicorn app.main:app --reload${NC}"
echo -e "2. Frontend: ${YELLOW}cd frontend && npm run dev${NC}"
echo
echo -e "Pour déployer sur O2Switch, utilisez: ${YELLOW}./scripts/deploy_o2switch.sh${NC}"
echo

exit 0 
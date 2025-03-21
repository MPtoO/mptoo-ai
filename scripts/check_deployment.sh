#!/bin/bash

# Script de vérification des prérequis pour le déploiement MPTOO
# Copyright 2025 Mohammed Amine Taybi
# Licensed under the Apache License, Version 2.0

echo "=== Vérification des prérequis pour le déploiement MPTOO ==="
echo

# Variables
ENV_FILE=".env"
REQUIRED_COMMANDS=("python3" "pip" "psql" "docker" "docker-compose" "node" "npm")
REQUIRED_PYTHON_VERSION="3.10"
REQUIRED_NODE_VERSION="16"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Fonction pour vérifier la présence d'une commande
check_command() {
    if command -v $1 >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $1 est installé"
        return 0
    else
        echo -e "${RED}✗${NC} $1 n'est pas installé"
        return 1
    fi
}

# Fonction pour vérifier la version de Python
check_python_version() {
    local version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$version >= $REQUIRED_PYTHON_VERSION" | bc -l) -eq 1 ]]; then
        echo -e "${GREEN}✓${NC} Python version $version (>= $REQUIRED_PYTHON_VERSION requis)"
        return 0
    else
        echo -e "${RED}✗${NC} Python version $version (>= $REQUIRED_PYTHON_VERSION requis)"
        return 1
    fi
}

# Fonction pour vérifier la version de Node
check_node_version() {
    local version=$(node -v | cut -d 'v' -f 2)
    if [[ $(echo "$version >= $REQUIRED_NODE_VERSION" | bc -l) -eq 1 ]]; then
        echo -e "${GREEN}✓${NC} Node.js version $version (>= $REQUIRED_NODE_VERSION requis)"
        return 0
    else
        echo -e "${RED}✗${NC} Node.js version $version (>= $REQUIRED_NODE_VERSION requis)"
        return 1
    fi
}

# Fonction pour vérifier le fichier .env
check_env_file() {
    if [ -f "$ENV_FILE" ]; then
        echo -e "${GREEN}✓${NC} Fichier .env trouvé"
        
        # Vérifier les variables critiques
        local missing_vars=()
        local required_vars=("SECRET_KEY" "POSTGRES_USER" "POSTGRES_PASSWORD" "POSTGRES_DB")
        
        for var in "${required_vars[@]}"; do
            if ! grep -q "^$var=" "$ENV_FILE"; then
                missing_vars+=("$var")
            fi
        done
        
        if [ ${#missing_vars[@]} -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Toutes les variables d'environnement requises sont définies"
            return 0
        else
            echo -e "${RED}✗${NC} Variables d'environnement manquantes: ${missing_vars[*]}"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} Fichier .env non trouvé. Veuillez le créer à partir de .env.example"
        return 1
    fi
}

# Fonction pour vérifier si Postgres est accessible
check_postgres() {
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE"
        if PGPASSWORD=$POSTGRES_PASSWORD psql -h ${POSTGRES_HOST:-localhost} -U $POSTGRES_USER -d $POSTGRES_DB -c '\q' 2>/dev/null; then
            echo -e "${GREEN}✓${NC} Connexion à PostgreSQL réussie"
            return 0
        else
            echo -e "${RED}✗${NC} Impossible de se connecter à PostgreSQL"
            return 1
        fi
    else
        echo -e "${YELLOW}!${NC} Test PostgreSQL ignoré (fichier .env manquant)"
        return 1
    fi
}

# Fonction pour vérifier l'état des dépendances Python
check_python_deps() {
    if [ -f "backend/requirements.txt" ]; then
        echo -e "${GREEN}✓${NC} Fichier requirements.txt trouvé"
        
        local missing_deps=()
        local requirements=$(cat backend/requirements.txt | grep -v "^#" | cut -d '=' -f 1)
        
        echo "Vérification des dépendances Python..."
        for req in $requirements; do
            if ! pip list | grep -q "^$req "; then
                missing_deps+=("$req")
            fi
        done
        
        if [ ${#missing_deps[@]} -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Toutes les dépendances Python sont installées"
            return 0
        else
            echo -e "${YELLOW}!${NC} Dépendances Python manquantes: ${missing_deps[*]}"
            echo "Exécutez 'pip install -r backend/requirements.txt' pour les installer"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} Fichier requirements.txt non trouvé"
        return 1
    fi
}

# Fonction pour vérifier les dépendances Node.js
check_node_deps() {
    if [ -f "frontend/package.json" ]; then
        echo -e "${GREEN}✓${NC} Fichier package.json trouvé"
        
        if [ -d "frontend/node_modules" ]; then
            echo -e "${GREEN}✓${NC} Dépendances Node.js installées"
            return 0
        else
            echo -e "${YELLOW}!${NC} Dépendances Node.js non installées"
            echo "Exécutez 'cd frontend && npm install' pour les installer"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} Fichier package.json non trouvé"
        return 1
    fi
}

# Vérification principale
echo "1. Vérification des commandes requises..."
missing_commands=0
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    check_command "$cmd" || ((missing_commands++))
done
echo

echo "2. Vérification des versions..."
check_python_version
check_node_version
echo

echo "3. Vérification de la configuration..."
check_env_file
echo

echo "4. Vérification de la base de données..."
check_postgres
echo

echo "5. Vérification des dépendances..."
check_python_deps
check_node_deps
echo

# Résumé
if [ $missing_commands -gt 0 ]; then
    echo -e "${RED}✗ $missing_commands commande(s) requise(s) manquante(s)${NC}"
    echo "Veuillez installer les outils manquants avant de continuer."
    exit 1
fi

echo "=== Vérification terminée ==="
echo -e "${GREEN}Votre système est prêt pour le déploiement MPTOO.${NC}"
echo
echo "Pour déployer sur O2Switch:"
echo "1. Assurez-vous que votre domaine pointe vers votre hébergement O2Switch"
echo "2. Configurez correctement le fichier .env pour l'environnement de production"
echo "3. Exécutez le script de déploiement: ./scripts/deploy_o2switch.sh"
echo

exit 0 
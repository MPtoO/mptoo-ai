#!/bin/bash

# Script de sauvegarde MPTOO
# Copyright 2025 Mohammed Amine Taybi
# Licensed under the Apache License, Version 2.0

# Configuration
POSTGRES_USER=${POSTGRES_USER:-"mptoo_user"}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-"votre_mot_de_passe"}
POSTGRES_DB=${POSTGRES_DB:-"mptoo_prod"}
POSTGRES_HOST=${POSTGRES_HOST:-"localhost"}

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Date du jour pour le nom du fichier
DATE=$(date +%Y%m%d)
BACKUP_DIR=~/backups/$DATE

# Création du répertoire de sauvegarde
echo -e "${YELLOW}Création du répertoire de sauvegarde...${NC}"
mkdir -p $BACKUP_DIR

# Sauvegarde de la base de données
echo -e "${YELLOW}Sauvegarde de la base de données...${NC}"
if PGPASSWORD="$POSTGRES_PASSWORD" pg_dump -h $POSTGRES_HOST -U $POSTGRES_USER -F c $POSTGRES_DB > $BACKUP_DIR/mptoo_db.dump; then
    echo -e "${GREEN}Base de données sauvegardée avec succès.${NC}"
else
    echo -e "${RED}Échec de la sauvegarde de la base de données.${NC}"
    echo -e "${YELLOW}Vérifiez les informations de connexion PostgreSQL.${NC}"
    exit 1
fi

# Sauvegarde des fichiers uploadés
echo -e "${YELLOW}Sauvegarde des fichiers uploadés...${NC}"
if [ -d ~/mptoo/backend/data/uploads ]; then
    cp -r ~/mptoo/backend/data/uploads $BACKUP_DIR/uploads
    echo -e "${GREEN}Fichiers uploadés sauvegardés avec succès.${NC}"
else
    echo -e "${YELLOW}Répertoire d'uploads non trouvé. Ignoré.${NC}"
fi

# Sauvegarde des modèles ML
echo -e "${YELLOW}Sauvegarde des modèles d'IA/ML...${NC}"
if [ -d ~/mptoo/backend/data/models ]; then
    cp -r ~/mptoo/backend/data/models $BACKUP_DIR/models
    echo -e "${GREEN}Modèles sauvegardés avec succès.${NC}"
else
    echo -e "${YELLOW}Répertoire de modèles non trouvé. Ignoré.${NC}"
fi

# Sauvegarde de la configuration
echo -e "${YELLOW}Sauvegarde de la configuration...${NC}"
if [ -f ~/mptoo/backend/.env ]; then
    cp ~/mptoo/backend/.env $BACKUP_DIR/
    echo -e "${GREEN}Fichier de configuration sauvegardé avec succès.${NC}"
else
    echo -e "${YELLOW}Fichier .env non trouvé. Ignoré.${NC}"
fi

# Archivage
echo -e "${YELLOW}Archivage des sauvegardes...${NC}"
cd ~/backups
tar czf mptoo_backup_$DATE.tar.gz $DATE
if [ $? -eq 0 ]; then
    # Nettoyage
    rm -rf $DATE
    echo -e "${GREEN}Sauvegarde archivée avec succès: ~/backups/mptoo_backup_$DATE.tar.gz${NC}"
else
    echo -e "${RED}Échec de l'archivage des sauvegardes.${NC}"
    exit 1
fi

# Nettoyage des anciennes sauvegardes (conserver les 30 derniers jours)
echo -e "${YELLOW}Nettoyage des anciennes sauvegardes...${NC}"
find ~/backups -name "mptoo_backup_*.tar.gz" -mtime +30 -delete
echo -e "${GREEN}Anciennes sauvegardes nettoyées.${NC}"

# Résumé
echo
echo -e "${GREEN}Sauvegarde terminée avec succès!${NC}"
echo -e "Nom du fichier: ~/backups/mptoo_backup_$DATE.tar.gz"
echo -e "Taille: $(du -h ~/backups/mptoo_backup_$DATE.tar.gz | cut -f1)"
echo

exit 0 
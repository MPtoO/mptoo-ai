#!/bin/bash

# Script de déploiement MPTOO sur O2Switch
# Copyright 2025 Mohammed Amine Taybi
# Licensed under the Apache License, Version 2.0

# Configuration
SSH_USER="votreidentifiant"
SSH_HOST="votredomaine.tld"
REMOTE_DIR="~/mptoo"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Vérification des prérequis
echo -e "${YELLOW}Vérification des prérequis...${NC}"
./scripts/check_deployment.sh
if [ $? -ne 0 ]; then
  echo -e "${RED}Les prérequis ne sont pas satisfaits. Veuillez corriger les erreurs ci-dessus avant de continuer.${NC}"
  exit 1
fi

# Préparation
echo -e "${YELLOW}Préparation du frontend...${NC}"
cd frontend
npm run build
if [ $? -ne 0 ]; then
  echo -e "${RED}Échec de la compilation du frontend.${NC}"
  exit 1
fi
cd ..

# Vérification de la connexion SSH
echo -e "${YELLOW}Vérification de la connexion SSH à $SSH_HOST...${NC}"
ssh -q "$SSH_USER@$SSH_HOST" exit
if [ $? -ne 0 ]; then
  echo -e "${RED}Impossible de se connecter à $SSH_USER@$SSH_HOST. Veuillez vérifier vos informations de connexion.${NC}"
  exit 1
fi

# Création des répertoires distants
echo -e "${YELLOW}Création des répertoires distants...${NC}"
ssh "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_DIR/backend/data/models $REMOTE_DIR/backend/data/uploads $REMOTE_DIR/logs $REMOTE_DIR/frontend $REMOTE_DIR/scripts $REMOTE_DIR/backups"

# Transfert des fichiers backend
echo -e "${YELLOW}Transfert des fichiers backend...${NC}"
rsync -avz --exclude 'venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' backend/ "$SSH_USER@$SSH_HOST:$REMOTE_DIR/backend/"

# Transfert des fichiers frontend compilés
echo -e "${YELLOW}Transfert des fichiers frontend...${NC}"
rsync -avz frontend/dist/ "$SSH_USER@$SSH_HOST:$REMOTE_DIR/frontend/"

# Transfert des scripts
echo -e "${YELLOW}Transfert des scripts...${NC}"
rsync -avz scripts/ "$SSH_USER@$SSH_HOST:$REMOTE_DIR/scripts/"

# Rendre les scripts exécutables
echo -e "${YELLOW}Configuration des permissions...${NC}"
ssh "$SSH_USER@$SSH_HOST" "chmod +x $REMOTE_DIR/scripts/*.sh $REMOTE_DIR/backend/start.sh $REMOTE_DIR/backend/stop.sh"

# Configuration des fichiers .htaccess
echo -e "${YELLOW}Création des fichiers .htaccess...${NC}"

# Backend .htaccess
ssh "$SSH_USER@$SSH_HOST" "cat > $REMOTE_DIR/backend/.htaccess" << 'EOF'
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /

  # Ne pas rediriger les requêtes vers les fichiers statiques
  RewriteCond %{REQUEST_URI} ^/static/ [OR]
  RewriteCond %{REQUEST_URI} ^/.well-known/ [OR]
  RewriteCond %{REQUEST_FILENAME} -f
  RewriteRule .* - [L]

  # Rediriger toutes les autres requêtes vers le processus FastAPI
  RewriteRule ^(.*)$ http://localhost:8000/$1 [P,L]
</IfModule>

# Protection des fichiers sensibles
<FilesMatch "^\.env|.*\.py$">
  Order allow,deny
  Deny from all
</FilesMatch>

# Permettre l'accès au dossier static
<Directory "~/mptoo/backend/static">
  Allow from all
</Directory>
EOF

# Frontend .htaccess
ssh "$SSH_USER@$SSH_HOST" "cat > $REMOTE_DIR/frontend/.htaccess" << 'EOF'
<IfModule mod_rewrite.c>
  RewriteEngine On
  RewriteBase /
  
  # Si le fichier demandé existe, le servir directement
  RewriteCond %{REQUEST_FILENAME} -f
  RewriteRule ^ - [L]
  
  # Rediriger toutes les autres requêtes vers index.html
  RewriteRule ^ index.html [L]
</IfModule>

# GZIP Compression
<IfModule mod_deflate.c>
  AddOutputFilterByType DEFLATE text/html text/plain text/xml text/css text/javascript application/javascript application/x-javascript application/json
</IfModule>

# Cache headers
<IfModule mod_expires.c>
  ExpiresActive On
  ExpiresByType image/jpg "access plus 1 year"
  ExpiresByType image/jpeg "access plus 1 year"
  ExpiresByType image/gif "access plus 1 year"
  ExpiresByType image/png "access plus 1 year"
  ExpiresByType image/svg+xml "access plus 1 year"
  ExpiresByType text/css "access plus 1 month"
  ExpiresByType application/pdf "access plus 1 month"
  ExpiresByType text/javascript "access plus 1 month"
  ExpiresByType application/javascript "access plus 1 month"
  ExpiresByType application/x-javascript "access plus 1 month"
  ExpiresByType application/x-shockwave-flash "access plus 1 month"
  ExpiresByType image/x-icon "access plus 1 year"
  ExpiresDefault "access plus 2 days"
</IfModule>
EOF

# Création des scripts de démarrage/arrêt sur le serveur
echo -e "${YELLOW}Création des scripts de démarrage et d'arrêt...${NC}"

# Script de démarrage
ssh "$SSH_USER@$SSH_HOST" "cat > $REMOTE_DIR/backend/start.sh" << 'EOF'
#!/bin/bash
cd ~/mptoo/backend
source venv/bin/activate
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2 > ../logs/api.log 2>&1 &
echo $! > mptoo.pid
echo "MPTOO API démarré avec PID $(cat mptoo.pid)"
EOF

# Script d'arrêt
ssh "$SSH_USER@$SSH_HOST" "cat > $REMOTE_DIR/backend/stop.sh" << 'EOF'
#!/bin/bash
if [ -f ~/mptoo/backend/mptoo.pid ]; then
  kill $(cat ~/mptoo/backend/mptoo.pid)
  rm ~/mptoo/backend/mptoo.pid
  echo "MPTOO API arrêté"
else
  echo "Aucun PID trouvé, vérifiez si l'API est en cours d'exécution"
fi
EOF

# Installation de l'environnement Python
echo -e "${YELLOW}Configuration de l'environnement Python...${NC}"
ssh "$SSH_USER@$SSH_HOST" "cd $REMOTE_DIR/backend && python3 -m venv venv && source venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

# Configuration des tâches cron
echo -e "${YELLOW}Configuration des tâches cron...${NC}"
CRON_TMP=$(mktemp)
ssh "$SSH_USER@$SSH_HOST" "crontab -l" > "$CRON_TMP" 2>/dev/null || true
cat >> "$CRON_TMP" << EOF
# MPTOO - Vérifier toutes les 5 minutes si l'API est en cours d'exécution
*/5 * * * * if [ ! -f ~/mptoo/backend/mptoo.pid ] || ! ps -p \$(cat ~/mptoo/backend/mptoo.pid) > /dev/null; then cd ~/mptoo/backend && ./start.sh; fi

# MPTOO - Redémarrer l'API chaque jour à 3h du matin
0 3 * * * cd ~/mptoo/backend && ./stop.sh && sleep 10 && ./start.sh

# MPTOO - Sauvegarde quotidienne à 2h du matin
0 2 * * * cd ~/mptoo && ./scripts/backup.sh
EOF
ssh "$SSH_USER@$SSH_HOST" "crontab -" < "$CRON_TMP"
rm "$CRON_TMP"

# Initialisation de la base de données (optionnel, décommentez si nécessaire)
# echo -e "${YELLOW}Initialisation de la base de données...${NC}"
# ssh "$SSH_USER@$SSH_HOST" "cd $REMOTE_DIR/backend && source venv/bin/activate && python -c 'from app.db.session import init_db; init_db()'"

# Démarrage de l'API
echo -e "${YELLOW}Démarrage de l'API...${NC}"
ssh "$SSH_USER@$SSH_HOST" "cd $REMOTE_DIR/backend && ./start.sh"

# Vérification du déploiement
echo -e "${YELLOW}Vérification du déploiement...${NC}"
sleep 5
API_STATUS=$(ssh "$SSH_USER@$SSH_HOST" "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/api/health")
if [ "$API_STATUS" = "200" ]; then
  echo -e "${GREEN}Le déploiement a réussi!${NC}"
  echo -e "${GREEN}API accessible sur https://api.votredomaine.com${NC}"
  echo -e "${GREEN}Frontend accessible sur https://votredomaine.com${NC}"
else
  echo -e "${RED}Le déploiement a échoué. L'API ne répond pas correctement.${NC}"
  echo -e "${YELLOW}Vérifiez les logs: ssh $SSH_USER@$SSH_HOST 'tail -f $REMOTE_DIR/logs/api.log'${NC}"
fi

echo
echo -e "${GREEN}Déploiement terminé!${NC}"
echo -e "N'oubliez pas de:"
echo -e "1. Vérifier que votre domaine et sous-domaine sont correctement configurés"
echo -e "2. Activer HTTPS via l'interface O2Switch"
echo -e "3. Créer un utilisateur administrateur si ce n'est pas déjà fait"
echo

exit 0 
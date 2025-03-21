# Guide de déploiement sur O2Switch

Copyright 2025 Mohammed Amine Taybi  
Licensed under the Apache License, Version 2.0

Ce document explique en détail comment déployer l'application MPTOO sur un hébergement O2Switch.

## 1. Prérequis

- Un compte d'hébergement O2Switch actif
- Accès SSH à votre compte O2Switch
- Un nom de domaine configuré et pointant vers O2Switch
- Les informations d'accès à la base de données PostgreSQL d'O2Switch

## 2. Configuration du domaine et des sous-domaines

1. Connectez-vous à votre espace client O2Switch
2. Allez dans "Noms de domaine" > "Ajouter un sous-domaine"
3. Créez un sous-domaine `api.votredomaine.com` qui pointera vers le dossier `/mptoo/backend/`
4. Assurez-vous que votre domaine principal pointe vers `/mptoo/frontend/`

## 3. Accès SSH et configuration initiale

Connectez-vous à votre hébergement via SSH:

```bash
ssh votreidentifiant@votredomaine.tld
```

Créez la structure de répertoires nécessaire:

```bash
mkdir -p ~/mptoo/backend/data/models
mkdir -p ~/mptoo/backend/data/uploads
mkdir -p ~/mptoo/logs
mkdir -p ~/mptoo/frontend
mkdir -p ~/backups
```

## 4. Configuration de PostgreSQL

O2Switch propose PostgreSQL préinstallé. Utilisez phpPgAdmin pour:

1. Créer une base de données `mptoo_prod`
2. Créer un utilisateur `mptoo_user` avec un mot de passe sécurisé
3. Accorder tous les privilèges à l'utilisateur sur cette base de données

Notez les informations de connexion:
- Hôte: généralement `localhost` ou l'adresse fournie par O2Switch
- Port: généralement `5432`
- Nom de la base de données: `mptoo_prod`
- Nom d'utilisateur: `mptoo_user`
- Mot de passe: celui que vous avez défini

## 5. Configuration du Backend

### 5.1. Préparation du fichier .env

Créez un fichier `.env` dans le répertoire `backend`:

```bash
cd ~/mptoo/backend
nano .env
```

Ajoutez le contenu suivant en remplaçant les valeurs par les vôtres:

```
SECRET_KEY=votre_cle_secrete_tres_longue_et_complexe
DEBUG=false
ALLOWED_HOSTS=votredomaine.com,api.votredomaine.com,localhost,127.0.0.1

POSTGRES_USER=mptoo_user
POSTGRES_PASSWORD=votre_mot_de_passe
POSTGRES_DB=mptoo_prod
POSTGRES_HOST=localhost
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:5432/${POSTGRES_DB}

CORS_ORIGINS=https://votredomaine.com,http://localhost:8080
JWT_SECRET_KEY=autre_cle_secrete_tres_longue_et_complexe
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

MAX_UPLOAD_SIZE=10485760
UPLOAD_DIR=/home/votreidentifiant/mptoo/backend/data/uploads

REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

### 5.2. Configuration de Python et environnement virtuel

O2Switch propose Python, mais vous devez vérifier la version disponible:

```bash
python3 --version
```

Si la version est inférieure à 3.10, vous devrez compiler Python depuis les sources:

```bash
cd ~
wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
tar xzf Python-3.10.12.tgz
cd Python-3.10.12
./configure --prefix=$HOME/python3.10
make
make install
echo 'export PATH=$HOME/python3.10/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Créez un environnement virtuel:

```bash
cd ~/mptoo/backend
python3 -m venv venv
source venv/bin/activate
```

### 5.3. Installation des dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5.4. Configuration du fichier .htaccess

Créez un fichier `.htaccess` dans le répertoire du backend:

```bash
cd ~/mptoo/backend
nano .htaccess
```

Ajoutez le contenu suivant:

```apache
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
```

## 6. Configuration du Frontend

### 6.1. Compilation locale

Sur votre machine de développement, compilez le frontend:

```bash
cd frontend
npm run build
```

### 6.2. Transfert des fichiers

Utilisez SCP ou SFTP pour transférer les fichiers compilés:

```bash
scp -r dist/* votreidentifiant@votredomaine.tld:~/mptoo/frontend/
```

### 6.3. Configuration du .htaccess pour le frontend

Créez un fichier `.htaccess` dans le répertoire frontend:

```bash
cd ~/mptoo/frontend
nano .htaccess
```

Ajoutez le contenu suivant:

```apache
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
```

## 7. Démarrage du Backend FastAPI

### 7.1. Script de démarrage

Créez un script pour démarrer l'application:

```bash
cd ~/mptoo/backend
nano start.sh
```

Ajoutez le contenu suivant:

```bash
#!/bin/bash
cd ~/mptoo/backend
source venv/bin/activate
nohup uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 2 > ../logs/api.log 2>&1 &
echo $! > mptoo.pid
echo "MPTOO API démarré avec PID $(cat mptoo.pid)"
```

Rendez le script exécutable:

```bash
chmod +x start.sh
```

### 7.2. Script d'arrêt

Créez un script pour arrêter l'application:

```bash
cd ~/mptoo/backend
nano stop.sh
```

Ajoutez le contenu suivant:

```bash
#!/bin/bash
if [ -f ~/mptoo/backend/mptoo.pid ]; then
  kill $(cat ~/mptoo/backend/mptoo.pid)
  rm ~/mptoo/backend/mptoo.pid
  echo "MPTOO API arrêté"
else
  echo "Aucun PID trouvé, vérifiez si l'API est en cours d'exécution"
fi
```

Rendez le script exécutable:

```bash
chmod +x stop.sh
```

### 7.3. Démarrage automatique avec cron

Configurez cron pour redémarrer l'API en cas de panne:

```bash
crontab -e
```

Ajoutez les lignes suivantes:

```
# Vérifier toutes les 5 minutes si l'API est en cours d'exécution
*/5 * * * * if [ ! -f ~/mptoo/backend/mptoo.pid ] || ! ps -p $(cat ~/mptoo/backend/mptoo.pid) > /dev/null; then cd ~/mptoo/backend && ./start.sh; fi

# Redémarrer l'API chaque jour à 3h du matin
0 3 * * * cd ~/mptoo/backend && ./stop.sh && sleep 10 && ./start.sh

# Sauvegarde quotidienne à 2h du matin
0 2 * * * cd ~/mptoo && ./scripts/backup.sh
```

## 8. Initialisation de la base de données

Si nécessaire, initialisez la base de données:

```bash
cd ~/mptoo/backend
source venv/bin/activate
python -c 'from app.db.session import init_db; init_db()'
```

Créez un utilisateur administrateur:

```bash
python -c 'from app.db.session import get_db; from app.services.user_service import UserService; from app.schemas.user import UserCreate; db = next(get_db()); user = UserCreate(email="admin@votredomaine.com", username="admin", password="motdepasse_securise", is_admin=True); UserService.create_user(db, user)'
```

## 9. Test et vérification

1. Démarrez l'API:
   ```bash
   cd ~/mptoo/backend
   ./start.sh
   ```

2. Vérifiez qu'elle fonctionne:
   ```bash
   curl http://localhost:8000/api/health
   ```

3. Testez l'accès via le domaine:
   ```bash
   curl https://api.votredomaine.com/api/health
   ```

4. Testez l'accès au frontend en visitant `https://votredomaine.com` dans votre navigateur

## 10. Maintenance et mises à jour

### 10.1. Script de sauvegarde

Créez un script de sauvegarde:

```bash
mkdir -p ~/mptoo/scripts
cd ~/mptoo/scripts
nano backup.sh
```

Ajoutez le contenu suivant:

```bash
#!/bin/bash
DATE=$(date +%Y%m%d)
BACKUP_DIR=~/backups/$DATE

# Créer le répertoire de backup
mkdir -p $BACKUP_DIR

# Sauvegarder la base de données
PGPASSWORD="votre_mot_de_passe" pg_dump -h localhost -U mptoo_user -F c mptoo_prod > $BACKUP_DIR/mptoo_db.dump

# Sauvegarder les fichiers uploadés
cp -r ~/mptoo/backend/data/uploads $BACKUP_DIR/uploads

# Archiver le tout
cd ~/backups
tar czf mptoo_backup_$DATE.tar.gz $DATE

# Nettoyage
rm -rf $DATE

# Supprimer les backups de plus de 30 jours
find ~/backups -name "mptoo_backup_*.tar.gz" -mtime +30 -delete

echo "Sauvegarde terminée: ~/backups/mptoo_backup_$DATE.tar.gz"
```

Rendez le script exécutable:

```bash
chmod +x backup.sh
```

### 10.2. Script de mise à jour

Créez un script de mise à jour:

```bash
cd ~/mptoo/scripts
nano update.sh
```

Ajoutez le contenu suivant:

```bash
#!/bin/bash
echo "Mise à jour de MPTOO..."

# Sauvegarde avant mise à jour
./backup.sh

# Arrêt de l'API
cd ~/mptoo/backend
./stop.sh

# Pull des dernières modifications (si vous utilisez Git)
cd ~/mptoo
git pull

# Mise à jour des dépendances backend
cd ~/mptoo/backend
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Migration de la base de données si nécessaire
python -c 'from app.db.session import migrate_db; migrate_db()'

# Démarrage de l'API
./start.sh

echo "Mise à jour terminée!"
```

Rendez le script exécutable:

```bash
chmod +x update.sh
```

## 11. Résolution des problèmes

### 11.1. Vérification des logs

```bash
tail -f ~/mptoo/logs/api.log
```

### 11.2. Problèmes courants

#### L'API ne démarre pas

Vérifiez les logs et les dépendances:
```bash
cd ~/mptoo/backend
source venv/bin/activate
pip install -r requirements.txt
```

#### Erreurs de connexion à la base de données

Vérifiez les paramètres de connexion dans le fichier `.env` et assurez-vous que PostgreSQL fonctionne correctement.

#### Problèmes de redirection

Vérifiez la configuration des fichiers `.htaccess` et assurez-vous que mod_rewrite est activé.

#### Erreurs d'upload de fichiers

Vérifiez les permissions du répertoire d'upload:
```bash
chmod -R 755 ~/mptoo/backend/data/uploads
```

## 12. Sécurité

### 12.1. Mise en place de HTTPS

O2Switch propose Let's Encrypt préinstallé. Pour l'activer:

1. Connectez-vous à votre espace client O2Switch
2. Activez SSL/TLS pour votre domaine et vos sous-domaines
3. Forcez HTTPS en ajoutant au début de vos fichiers `.htaccess`:

```apache
# Redirection HTTP vers HTTPS
RewriteEngine On
RewriteCond %{HTTPS} off
RewriteRule ^(.*)$ https://%{HTTP_HOST}%{REQUEST_URI} [L,R=301]
```

### 12.2. Sécurisation des fichiers sensibles

Assurez-vous que les fichiers sensibles ne sont pas accessibles:

```bash
chmod 600 ~/mptoo/backend/.env
```

## Conclusion

Votre application MPTOO devrait maintenant être correctement déployée sur O2Switch. N'oubliez pas de surveiller régulièrement les logs et de réaliser des sauvegardes. Pour toute mise à jour future, utilisez le script `update.sh`. 
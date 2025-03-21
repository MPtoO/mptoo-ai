# Tests de déploiement MPTOO sur O2Switch

Copyright 2025 Mohammed Amine Taybi  
Licensed under the Apache License, Version 2.0

Ce document décrit les tests à effectuer pour vérifier que le déploiement de l'application MPTOO sur O2Switch a été réalisé avec succès.

## 1. Liste de vérification préalable au déploiement

Avant de procéder au déploiement, assurez-vous que:

- [ ] Le script `scripts/check_deployment.sh` a été exécuté et a réussi
- [ ] Le fichier `.env.example` a été copié vers `.env` et configuré correctement
- [ ] Les variables d'environnement pour PostgreSQL sont correctes
- [ ] Les clés API nécessaires sont configurées dans le fichier `.env`
- [ ] Le frontend peut être compilé avec succès localement

## 2. Tests après déploiement

### 2.1. Vérification du backend

- [ ] Connexion SSH à O2Switch réussie
- [ ] L'API est en cours d'exécution (vérifiez avec `ps aux | grep uvicorn`)
- [ ] Le point de terminaison `/api/health` renvoie un code 200
- [ ] La documentation Swagger est accessible à `/docs`
- [ ] Les migrations de base de données ont été appliquées avec succès

Commande pour vérifier l'état de l'API:
```bash
curl -v https://api.votredomaine.com/api/health
```

### 2.2. Vérification du frontend

- [ ] Le frontend est accessible via https://votredomaine.com
- [ ] La page d'accueil se charge correctement
- [ ] Les requêtes API du frontend vers le backend fonctionnent
- [ ] La navigation entre les différentes pages fonctionne correctement
- [ ] Le formulaire de connexion fonctionne

### 2.3. Vérification de la base de données

- [ ] La connexion à PostgreSQL fonctionne
- [ ] Les tables ont été correctement créées
- [ ] Un utilisateur administrateur a été créé

Commande pour vérifier la connexion à la base de données:
```bash
PGPASSWORD=votre_mot_de_passe psql -h localhost -U mptoo_user -d mptoo_prod -c "\dt"
```

### 2.4. Vérification des tâches planifiées

- [ ] Les tâches cron ont été correctement configurées

Commande pour vérifier les tâches cron:
```bash
crontab -l | grep mptoo
```

### 2.5. Vérification des fichiers .htaccess

- [ ] Le fichier .htaccess du frontend redirige correctement vers index.html
- [ ] Le fichier .htaccess du backend redirige correctement vers l'API

## 3. Tests fonctionnels

- [ ] Création d'un compte utilisateur
- [ ] Connexion avec cet utilisateur
- [ ] Accès aux fonctionnalités réservées aux utilisateurs connectés
- [ ] Upload de fichiers dans le backend
- [ ] Utilisation des fonctionnalités d'analyse de données
- [ ] Utilisation des fonctionnalités d'IA/ML si applicables

## 4. Tests de performance

- [ ] Temps de chargement de la page d'accueil < 2 secondes
- [ ] Temps de réponse de l'API pour les requêtes simples < 300ms
- [ ] Temps de réponse pour les opérations d'analyse de données < 5 secondes

Outil recommandé pour les tests de performance:
```bash
ab -n 100 -c 10 https://api.votredomaine.com/api/health
```

## 5. Tests de sécurité

- [ ] HTTPS est correctement configuré (vérifiez avec SSLLabs)
- [ ] Les en-têtes de sécurité sont correctement configurés
- [ ] Les fichiers sensibles ne sont pas accessibles publiquement

Commande pour vérifier l'accès aux fichiers sensibles:
```bash
curl -I https://api.votredomaine.com/.env
# Devrait retourner 403 Forbidden
```

## 6. Tests des sauvegardes

- [ ] Le script `scripts/backup.sh` s'exécute correctement
- [ ] Les sauvegardes sont créées dans le répertoire approprié
- [ ] Une sauvegarde peut être restaurée avec succès

Commande pour tester la sauvegarde manuellement:
```bash
cd ~/mptoo
./scripts/backup.sh
```

## 7. Problèmes courants et solutions

### L'API ne démarre pas

Vérifiez les logs:
```bash
tail -f ~/mptoo/logs/api.log
```

Solutions possibles:
1. Vérifiez que les dépendances Python sont installées
2. Vérifiez que le port 8000 n'est pas déjà utilisé
3. Assurez-vous que le fichier .env est correctement configuré

### Les requêtes API échouent

Solutions possibles:
1. Vérifiez que le module mod_proxy est activé dans Apache
2. Vérifiez la configuration du .htaccess
3. Assurez-vous que l'API est en cours d'exécution

### Le frontend ne se charge pas correctement

Solutions possibles:
1. Vérifiez que les fichiers ont été correctement transférés
2. Vérifiez la configuration du .htaccess
3. Vérifiez les erreurs dans la console du navigateur

### Erreurs de base de données

Solutions possibles:
1. Vérifiez les informations de connexion dans le fichier .env
2. Vérifiez que PostgreSQL est en cours d'exécution
3. Vérifiez les privilèges de l'utilisateur de la base de données

## 8. Opérations de maintenance

### Mise à jour de l'application

```bash
cd ~/mptoo
./scripts/update.sh
```

### Redémarrage manuel de l'API

```bash
cd ~/mptoo/backend
./stop.sh
./start.sh
```

### Restauration d'une sauvegarde

```bash
# Remplacez DATE par la date de la sauvegarde (format AAAAMMJJ)
cd ~/backups
tar xzf mptoo_backup_DATE.tar.gz
cd DATE
PGPASSWORD=votre_mot_de_passe pg_restore -h localhost -U mptoo_user -d mptoo_prod -c mptoo_db.dump
cp -r uploads ~/mptoo/backend/data/
cp -r models ~/mptoo/backend/data/
```

## Conclusion

Si tous les tests ci-dessus passent avec succès, votre application MPTOO est correctement déployée sur O2Switch et prête à être utilisée en production. N'oubliez pas de surveiller régulièrement les logs et de réaliser des sauvegardes pour assurer la continuité de service. 
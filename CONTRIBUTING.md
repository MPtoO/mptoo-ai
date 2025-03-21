# Guide de contribution à MPTOO AI

Nous sommes ravis que vous souhaitiez contribuer à MPTOO AI! Ce document présente les lignes directrices pour contribuer efficacement au projet.

## Liens communautaires

- **Discord**: [https://discord.gg/Mmj6xyUr](https://discord.gg/Mmj6xyUr)
- **GitHub**: [https://github.com/MPtoO](https://github.com/MPtoO)

## Comment contribuer

Voici plusieurs façons de contribuer au projet:

1. **Développement de modèles**: Améliorer les modèles existants ou en créer de nouveaux
2. **Création d'agents**: Développer de nouveaux agents spécialisés
3. **Optimisation de performance**: Améliorer l'efficacité des modèles et des inférences
4. **Documentation**: Améliorer ou traduire la documentation
5. **Rapporter des bugs**: Identifier et documenter les problèmes
6. **Suggérer des fonctionnalités**: Proposer de nouvelles capacités d'IA

## Processus de contribution

### 1. Configuration initiale

1. Forkez le dépôt sur GitHub
2. Clonez votre fork localement
   ```bash
   git clone https://github.com/VOTRE-USERNAME/mptoo-ai.git
   cd mptoo-ai
   ```
3. Configurez le dépôt upstream
   ```bash
   git remote add upstream https://github.com/MPtoO/mptoo-ai.git
   ```

### 2. Synchronisation avec upstream

Avant de commencer à travailler sur une nouvelle fonctionnalité, synchronisez votre fork:

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### 3. Créer une branche pour votre travail

```bash
git checkout -b feature/nom-de-votre-fonctionnalite
```

Ou pour une correction de bug:

```bash
git checkout -b fix/description-du-bug
```

### 4. Développement

#### Organisation du code

- Respectez la structure des dossiers existante
- Suivez les conventions de nommage établies
- Documentez votre code avec des docstrings
- Écrivez des tests pour les nouvelles fonctionnalités

#### Gestion des données sensibles

- **N'incluez JAMAIS** de clés API, tokens, mots de passe ou autres informations sensibles
- N'intégrez pas de données d'utilisateurs ou de données privées
- Pour les exemples et tests, utilisez uniquement des données synthétiques ou publiques

#### Taille des modèles

- Les petits modèles (<100 Mo) peuvent être inclus directement
- Pour les modèles plus grands, fournissez des scripts pour les télécharger
- Documentez toujours la source et la licence des modèles

### 5. Soumission

1. Commitez vos changements avec des messages clairs
   ```bash
   git add .
   git commit -m "feat(agents): ajouter un agent d'analyse financière"
   ```

2. Mettez à jour votre branche avec les derniers changements upstream
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

3. Poussez votre branche vers votre fork
   ```bash
   git push origin feature/nom-de-votre-fonctionnalite
   ```

4. Créez une Pull Request sur GitHub

### 6. Revue de code

- Un mainteneur examinera votre PR
- Soyez prêt à répondre aux questions et à apporter des modifications
- Les tests automatisés doivent passer avant la fusion

## Standards de code

### Python

- Suivez PEP 8
- Utilisez des types statiques (annotations de type)
- Documentez toutes les fonctions et classes
- Maintenez une couverture de test adéquate

### Notebooks

- Nettoyez les sorties avant de commit
- Assurez-vous que les notebooks peuvent être exécutés de bout en bout
- Documentez clairement chaque étape

## Licence des contributions

En soumettant une contribution, vous acceptez que votre travail soit licencié sous la licence Apache 2.0 du projet.

## Créer et publier un nouveau modèle

1. Développez votre modèle dans le dossier approprié
2. Créez une documentation complète:
   - Méthodologie d'entraînement
   - Données utilisées (sources, prétraitement)
   - Performances et évaluations
   - Limites connues
3. Fournissez un exemple d'utilisation
4. Soumettez une Pull Request

## Communication

- Pour les questions courtes: utilisez Discord
- Pour les discussions techniques: créez une Issue GitHub
- Pour les propositions majeures: créez une RFC dans le dossier `docs/rfcs/`

## Reconnaissance des contributions

Tous les contributeurs sont listés dans notre fichier CONTRIBUTORS.md et sur la page du projet.

Merci de contribuer à MPTOO AI! 
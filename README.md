  __  __ ____  _        ___
 |  \/  |  _ \| |_ ___ / _ \
 | |\/| | |_) | __/ _ \ | | |
 | |  | |  __/| || (_) | |_| |
 |_|  |_|_|    \__\___/ \___/

# MPTOO AI
 
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/1234567890?color=7289da&label=Discord&logo=discord&logoColor=white)](https://discord.gg/Mmj6xyUr)

Dépôt pour le développement des fonctionnalités d'IA de MPTOO: fine-tuning, agents IA, et plus encore.

## À propos

Ce dépôt contient les composants IA de l'application MPTOO, une plateforme de gestion complète. Il est conçu pour permettre une collaboration ouverte autour du développement des capacités d'intelligence artificielle de notre application.

## Structure du projet

```
├── agents/              # Agents IA pour diverses tâches
├── fine-tuning/         # Scripts et configurations pour fine-tuning des modèles
├── models/              # Définitions de modèles (pas les poids)
├── inference/           # Code pour l'inférence en production
├── notebooks/           # Jupyter notebooks pour l'exploration et l'analyse
├── scripts/             # Scripts utilitaires
├── evaluation/          # Tests et évaluations de modèles
├── api/                 # API d'intégration avec l'application principale
├── examples/            # Exemples d'utilisation
└── docs/                # Documentation
```

## Installation

1. Clonez ce dépôt:
   ```bash
   git clone https://github.com/MPtoO/mptoo-ai.git
   cd mptoo-ai
   ```

2. Créez un environnement virtuel:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   ```

3. Installez les dépendances:
   ```bash
   pip install -r requirements.txt
   ```

## Modules

### Agents IA

Les agents IA sont des assistants spécialisés qui réalisent des tâches spécifiques:

- **DataAgent**: Analyse et prépare les données
- **ReportAgent**: Génère des rapports automatisés
- **AssistantAgent**: Aide les utilisateurs avec leurs tâches quotidiennes

### Fine-tuning

Scripts et processus pour personnaliser des modèles de langage pour des tâches spécifiques à MPTOO.

### Modèles

Définitions et configurations de modèles utilisés dans l'application.

## Contribution

Nous accueillons les contributions! Consultez [CONTRIBUTING.md](CONTRIBUTING.md) pour plus d'informations.

## Documentation

Pour plus d'informations, consultez la [documentation complète](./docs/README.md).

## Liens

- **Application principale**: [mptoo.com](https://mptoo.com)
- **Documentation**: [docs.mptoo.com](https://docs.mptoo.com)
- **Discord**: [Rejoindre](https://discord.gg/Mmj6xyUr)
- **GitHub**: [MPtoO](https://github.com/MPtoO)

## Licence

Copyright 2025 Mohammed Amine Taybi  
Licensed under the Apache License, Version 2.0

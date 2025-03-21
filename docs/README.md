# Documentation MPTOO AI

Documentation complète pour les composants d'IA de MPTOO.

## Table des matières

1. [Introduction](#introduction)
2. [Agents](#agents)
3. [Fine-tuning](#fine-tuning)
4. [Modèles](#modèles)
5. [Inférence](#inférence)
6. [Développement](#développement)
7. [Déploiement](#déploiement)

## Introduction

MPTOO AI est une collection de composants d'intelligence artificielle qui étendent les fonctionnalités de l'application MPTOO. Ces composants incluent des agents IA spécialisés, des modèles fine-tunés pour des tâches spécifiques et des outils d'analyse de données.

## Agents

Les agents IA sont des assistants autonomes qui peuvent effectuer des tâches spécifiques dans l'application MPTOO.

### Types d'Agents

- **DataAgent**: Analyse des données et génère des insights ([documentation](agents/data_agent.md))
- **ReportAgent**: Génère des rapports automatisés en langage naturel ([documentation](agents/report_agent.md))
- **AssistantAgent**: Aide les utilisateurs dans leurs tâches quotidiennes ([documentation](agents/assistant_agent.md))

### Utilisation des Agents

Pour utiliser un agent, importez la classe correspondante et initialisez-la:

```python
from mptoo.agents import DataAgent

# Initialiser l'agent
agent = DataAgent()

# Exécuter une requête
result = agent.run("Analyse les ventes du dernier trimestre et identifie les tendances")
```

## Fine-tuning

Le dépôt contient des scripts et des guides pour fine-tuner des modèles de langage pour les besoins spécifiques de MPTOO.

### Méthodes de Fine-tuning

- **LoRA**: Fine-tuning efficace avec Low-Rank Adaptation ([documentation](fine-tuning/lora.md))
- **SFT**: Supervised Fine-Tuning complet ([documentation](fine-tuning/sft.md))
- **RLHF**: Reinforcement Learning from Human Feedback ([documentation](fine-tuning/rlhf.md))

### Datasets

Nous utilisons plusieurs ensembles de données pour le fine-tuning:

- **MPTOO-Instructions**: Instructions spécifiques à notre domaine
- **MPTOO-Conversations**: Conversations entre utilisateurs et agents
- **MPTOO-Analytics**: Exemples d'analyses et de rapports

## Modèles

### Modèles Disponibles

- **MPTOO-Analyst**: Modèle spécialisé dans l'analyse de données financières et commerciales
- **MPTOO-Assistant**: Modèle conversationnel pour l'assistance utilisateur
- **MPTOO-Report**: Modèle pour la génération de rapports structurés

### Format d'Entrée

Tous nos modèles utilisent un format d'instruction standardisé:

```
instruction: <instruction>
context: <contexte pertinent, peut être vide>
response: <réponse générée par le modèle>
```

## Inférence

### Optimisation des Modèles

- Quantization (4-bit, 8-bit)
- ONNX Runtime
- vLLM pour l'inférence haute performance

### API d'Inférence

L'API d'inférence est disponible via FastAPI:

```python
import requests

response = requests.post(
    "https://api.mptoo.com/v1/generate",
    json={
        "model": "mptoo-analyst",
        "instruction": "Analyse ces données de vente",
        "context": "Les ventes ont augmenté de 15% au Q3...",
        "max_tokens": 500
    }
)
```

## Développement

### Installation pour le Développement

```bash
# Cloner le dépôt
git clone https://github.com/MPtoO/mptoo-ai.git
cd mptoo-ai

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
pip install -e .  # Installation en mode développement
```

### Tests

```bash
# Exécuter tous les tests
pytest

# Exécuter les tests pour un module spécifique
pytest tests/agents/test_data_agent.py

# Exécuter les tests avec couverture
pytest --cov=mptoo
```

## Déploiement

### Environnements

- **Développement**: Local ou Cloud personnel
- **Staging**: Environnement de test interne
- **Production**: Infrastructure haute disponibilité

### Docker

```bash
# Construire l'image
docker build -t mptoo/ai-api:latest .

# Exécuter le conteneur
docker run -p 8000:8000 mptoo/ai-api:latest
```

## Liens Communautaires

- **Discord**: [https://discord.gg/Mmj6xyUr](https://discord.gg/Mmj6xyUr)
- **GitHub**: [https://github.com/MPtoO](https://github.com/MPtoO)

## Licence

Copyright 2025 Mohammed Amine Taybi  
Licensed under the Apache License, Version 2.0 
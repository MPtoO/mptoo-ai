# Données mockées pour le développement

Ce dossier contient des données mockées utilisées pour simuler les réponses de l'API pendant le développement.

## Fichiers disponibles

- `mock_analysis.json` - Simulations pour le service d'analyse
- `mock_predictions.json` - Simulations pour le service de prédiction

## Format des données

### mock_analysis.json

```json
{
  "sujet_exemple": {
    "client_version": {
      "insights": [
        {
          "title": "Titre de l'insight",
          "summary": "Résumé de l'insight",
          "confidence": 0.85
        }
      ]
    },
    "expert_version": {
      "insights": [
        {
          "title": "Titre de l'insight",
          "summary": "Résumé de l'insight",
          "description": "Description détaillée",
          "confidence": 0.85,
          "sources": ["Source 1", "Source 2"],
          "related_topics": ["Sujet connexe 1", "Sujet connexe 2"]
        }
      ],
      "metadata": {
        "analysis_depth": "deep",
        "sources_count": 15,
        "processing_time": "2.3s",
        "confidence_score": 0.87,
        "model_version": "MPtoO Analysis v2.3"
      }
    }
  }
}
```

### mock_predictions.json

```json
{
  "trend_prediction_finance": {
    "forecast": [
      { "date": "2023-07-01", "value": 120.5 },
      { "date": "2023-07-02", "value": 121.3 }
    ],
    "trend_direction": "up",
    "confidence": 0.82,
    "summary": "La tendance pour la finance est orientée à la hausse."
  },
  "sentiment_analysis_produit": {
    "overall_sentiment": "positif",
    "sentiment_score": 0.78,
    "confidence": 0.85,
    "summary": "Le sentiment général concernant ce produit est positif."
  }
}
```

## Utilisation

1. Créez un fichier JSON dans ce dossier avec le format approprié
2. Les services d'API chargeront automatiquement ces données
3. Pour tester une réponse spécifique, utilisez l'un des patterns suivants:
   - Pour l'analyse: le sujet exact en minuscules
   - Pour les prédictions: `type_prediction_sujet` en minuscules avec espaces remplacés par des underscores

## Bonnes pratiques

- Créez des données mockées pour les scénarios de test courants
- Ajoutez progressivement des données plus détaillées pour la version "expert"
- Utilisez des valeurs réalistes pour les métriques (scores, confiance, etc.)
- Évitez d'ajouter des fichiers trop volumineux (> 1MB)

## Exemple d'ajout de mock

```python
# Exemple de script pour générer un fichier mock
import json

mock_data = {
  "innovation": {
    "client_version": {
      "insights": [
        {
          "title": "Tendances en innovation",
          "summary": "Résumé de l'analyse des tendances d'innovation",
          "confidence": 0.88
        }
      ]
    }
  }
}

with open('mock_analysis.json', 'w') as f:
    json.dump(mock_data, f, indent=2)
``` 
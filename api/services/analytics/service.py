#!/usr/bin/env python3
import pandas as pd
import numpy as np
from api.config import REDIS_URL
import redis

class AnalyticsService:
    def __init__(self):
        self.redis_client = redis.from_url(REDIS_URL)
        
    def analyze_dataset(self, data, metrics=None):
        """Analyse un dataset et retourne des métriques statistiques"""
        df = pd.DataFrame(data)
        results = {}
        
        # Mise en cache des résultats fréquemment demandés
        cache_key = f"analytics:{hash(str(data))}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
            
        # Calcul des statistiques de base
        results["summary"] = df.describe().to_dict()
        results["correlations"] = df.corr().to_dict()
        results["missing_values"] = df.isnull().sum().to_dict()
        
        # Mise en cache pour 30 minutes
        self.redis_client.setex(cache_key, 1800, json.dumps(results))
        
        return results

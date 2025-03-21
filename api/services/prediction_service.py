import logging
from typing import Dict, Any, List, Optional
import time
import random
from datetime import datetime, timedelta
import json
import os

from api.models.prediction import (
    PredictionRequest, PredictionResult, TrendPredictionResult,
    ForecastPoint, Anomaly, SeasonalityPattern
)

logger = logging.getLogger(__name__)

# Charger des données mockées depuis un fichier JSON si disponible
MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_predictions.json")
mock_data = {}

try:
    if os.path.exists(MOCK_DATA_PATH):
        with open(MOCK_DATA_PATH, 'r', encoding='utf-8') as f:
            mock_data = json.load(f)
        logger.info(f"Données de prédiction mockées chargées depuis {MOCK_DATA_PATH}")
except Exception as e:
    logger.warning(f"Impossible de charger les données mockées: {str(e)}")

class PredictionService:
    """Service pour les prédictions et analyses prédictives."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mock_data = mock_data
        self.dev_mode = os.environ.get("ENVIRONMENT", "development") == "development"
        
        # Mapper les types de prédiction aux méthodes de génération
        self.prediction_generators = {
            "trend_prediction": self._generate_trend_prediction,
            "forecast": self._generate_forecast_prediction,
            "sentiment_analysis": self._generate_sentiment_prediction,
            "market_analysis": self._generate_market_prediction,
            "risk_assessment": self._generate_risk_prediction,
            "recommendation": self._generate_recommendation_prediction
        }
    
    async def predict(
        self, 
        request: PredictionRequest, 
        interface_mode: str = "client"
    ) -> PredictionResult:
        """
        Effectue une prédiction en fonction du type demandé.
        
        Args:
            request: PredictionRequest contenant le type de prédiction et les paramètres
            interface_mode: Mode d'interface (client ou expert)
            
        Returns:
            PredictionResult avec les résultats de la prédiction
        """
        self.logger.info(f"Prediction requested: {request.prediction_type}, topic: {request.topic}")
        
        # Vérifier si une réponse mockée existe pour ce type de prédiction
        mock_key = f"{request.prediction_type}_{request.topic.lower().replace(' ', '_')}"
        if mock_key in self.mock_data or request.topic.lower() in self.mock_data:
            self.logger.info(f"Utilisation d'une réponse mockée pour {mock_key}")
            
            # Simuler un délai pour être plus réaliste
            time.sleep(random.uniform(0.2, 0.8))
            
            # Récupérer les données mockées
            mock_result = self.mock_data.get(mock_key, self.mock_data.get(request.topic.lower(), {}))
            
            # Adapter au mode d'interface si nécessaire
            if interface_mode == "client" and "client_version" in mock_result:
                result_data = mock_result["client_version"]
            elif interface_mode == "expert" and "expert_version" in mock_result:
                result_data = mock_result["expert_version"]
            else:
                result_data = mock_result
                
            return PredictionResult(
                prediction_type=request.prediction_type,
                topic=request.topic,
                results=result_data,
                mode=interface_mode
            )
        
        # Simulation du temps de traitement (à remplacer par une vraie prédiction)
        processing_time = random.uniform(1.0, 3.0)
        time.sleep(min(0.5, processing_time/3))  # Simuler un délai (réduit pour le développement)
        
        # Sélectionner le générateur approprié ou utiliser un générateur par défaut
        generator = self.prediction_generators.get(
            request.prediction_type, 
            self._generate_generic_prediction
        )
        
        results = generator(request, interface_mode)
        
        # Ajouter des métadonnées communes
        if interface_mode == "expert":
            results["metadata"] = {
                "processing_time": f"{processing_time:.1f}s",
                "model_version": "MPtoO Predictor v2.1",
                "confidence_level": round(random.uniform(0.70, 0.95), 2),
                "data_freshness": (datetime.now() - timedelta(days=random.randint(1, 15))).strftime("%Y-%m-%d")
            }
        
        return PredictionResult(
            prediction_type=request.prediction_type,
            topic=request.topic,
            results=results,
            mode=interface_mode
        )
    
    def _generate_generic_prediction(
        self, 
        request: PredictionRequest, 
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère une prédiction générique quand le type n'est pas reconnu.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les résultats de la prédiction
        """
        return {
            "result": f"Simulation de prédiction pour {request.prediction_type}",
            "summary": f"Prédiction générique pour le sujet: {request.topic}",
            "confidence": round(random.uniform(0.6, 0.9), 2),
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_trend_prediction(
        self, 
        request: PredictionRequest, 
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère une prédiction de tendance simulée.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les résultats de la prédiction
        """
        # Base value for the trend
        base_value = random.uniform(50, 200)
        
        # Trend direction (up, down, stable)
        trend_directions = ["up", "down", "stable"]
        trend_weights = [0.45, 0.35, 0.2]  # Pondération pour favoriser certaines tendances
        trend_direction = random.choices(trend_directions, weights=trend_weights)[0]
        
        # Generate forecast points
        forecast = []
        today = datetime.now()
        
        # Facteur de tendance
        trend_factor = 1.03 if trend_direction == "up" else (0.97 if trend_direction == "down" else 1.0)
        
        # Générer les points de prévision
        horizon = request.horizon or 30
        current_value = base_value
        
        for i in range(horizon):
            # Ajouter un peu de bruit aléatoire
            noise = random.uniform(-0.05, 0.05)
            
            # Calculer la nouvelle valeur avec tendance et bruit
            current_value = current_value * (trend_factor + noise)
            
            # Ajouter un pattern hebdomadaire (plus élevé le weekend)
            date = today + timedelta(days=i)
            if date.weekday() >= 5:  # samedi ou dimanche
                current_value *= random.uniform(1.1, 1.2)
            
            # Créer le point de prévision
            forecast.append(ForecastPoint(
                date=date.strftime("%Y-%m-%d"),
                value=round(current_value, 2)
            ))
        
        # Créer le résultat
        result = {
            "forecast": [{"date": f.date, "value": f.value} for f in forecast],
            "trend_direction": trend_direction,
            "confidence": round(random.uniform(0.7, 0.9), 2),
            "summary": f"La tendance pour {request.topic} est orientée à la {trend_direction_fr(trend_direction)} pour les {horizon} prochains jours."
        }
        
        # Ajouter des détails supplémentaires pour le mode expert
        if mode == "expert":
            # Saisonnalité
            seasonality = {
                "weekly_pattern": "weekend_peaks",
                "monthly_pattern": "mid_month_surge" if random.random() > 0.5 else "end_month_drop"
            }
            
            # Anomalies (quelques points aberrants)
            anomalies = []
            if random.random() > 0.3:  # 70% de chances d'avoir des anomalies
                num_anomalies = random.randint(1, 3)
                for _ in range(num_anomalies):
                    anomaly_day = random.randint(5, horizon-1)
                    expected = forecast[anomaly_day].value
                    actual = expected * random.uniform(1.3, 1.5) if random.random() > 0.5 else expected * random.uniform(0.5, 0.7)
                    
                    anomalies.append({
                        "date": forecast[anomaly_day].date,
                        "expected": round(expected, 2),
                        "actual": round(actual, 2),
                        "severity": random.choice(["low", "medium", "high"])
                    })
            
            result["seasonality"] = seasonality
            result["anomalies"] = anomalies
            result["analysis"] = f"L'analyse détaillée montre une tendance {trend_direction} avec une confiance de {result['confidence']*100:.0f}%. Des variations hebdomadaires sont observées avec des pics durant les weekends. {num_anomalies if 'num_anomalies' in locals() else 'Aucune'} anomalie{'s' if 'num_anomalies' in locals() and num_anomalies > 1 else ''} détectée{'s' if 'num_anomalies' in locals() and num_anomalies > 1 else ''}."
        
        return result
    
    def _generate_forecast_prediction(
        self,
        request: PredictionRequest,
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère une prédiction de prévision avec différents scénarios.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les résultats de la prédiction
        """
        # Valeur de base et horizon
        base_value = random.uniform(100, 1000)
        horizon = request.horizon or 12  # Par défaut 12 mois
        
        # Générer trois scénarios: pessimiste, réaliste, optimiste
        scenarios = {
            "optimistic": {"factor": 1.05, "variability": 0.03, "label": "Scénario optimiste"},
            "realistic": {"factor": 1.02, "variability": 0.02, "label": "Scénario réaliste"},
            "pessimistic": {"factor": 0.99, "variability": 0.04, "label": "Scénario pessimiste"}
        }
        
        # Générer les points pour chaque scénario
        forecasts = {}
        today = datetime.now()
        
        for scenario, params in scenarios.items():
            forecast = []
            current_value = base_value
            
            for i in range(horizon):
                # Ajouter du bruit à la tendance
                noise = random.uniform(-params["variability"], params["variability"])
                
                # Calculer la nouvelle valeur
                current_value = current_value * (params["factor"] + noise)
                
                # Ajouter des effets saisonniers
                month = (today.month + i) % 12 + 1
                if month in [11, 12]:  # Fin d'année
                    current_value *= random.uniform(1.05, 1.15)
                elif month in [1, 2]:  # Début d'année
                    current_value *= random.uniform(0.9, 0.95)
                
                date = (today + timedelta(days=i*30)).strftime("%Y-%m")
                forecast.append({"date": date, "value": round(current_value, 2)})
            
            forecasts[scenario] = forecast
        
        # Déterminer le scénario le plus probable
        probabilities = {
            "optimistic": round(random.uniform(0.1, 0.3), 2),
            "realistic": round(random.uniform(0.4, 0.6), 2),
            "pessimistic": 0
        }
        # Ajuster le pessimiste pour que le total soit 1
        probabilities["pessimistic"] = round(1 - probabilities["optimistic"] - probabilities["realistic"], 2)
        
        most_likely = max(probabilities, key=probabilities.get)
        
        # Créer le résultat
        result = {
            "forecasts": forecasts,
            "most_likely_scenario": most_likely,
            "scenario_probabilities": probabilities,
            "confidence": round(random.uniform(0.65, 0.85), 2),
            "summary": f"Prévision pour {request.topic} sur {horizon} mois: le scénario {scenarios[most_likely]['label']} est le plus probable ({probabilities[most_likely]*100:.0f}%)."
        }
        
        # Ajouter des détails supplémentaires pour le mode expert
        if mode == "expert":
            result["factors"] = {
                "economic": {
                    "impact": random.choice(["positive", "negative", "neutral"]),
                    "weight": round(random.uniform(0.1, 0.5), 2)
                },
                "market": {
                    "impact": random.choice(["positive", "negative", "neutral"]),
                    "weight": round(random.uniform(0.1, 0.4), 2)
                },
                "regulatory": {
                    "impact": random.choice(["positive", "negative", "neutral"]),
                    "weight": round(random.uniform(0.05, 0.3), 2)
                }
            }
            
            # Calculer les valeurs finales avec marges d'erreur
            final_values = {}
            for scenario in scenarios:
                last_value = forecasts[scenario][-1]["value"]
                error_margin = last_value * random.uniform(0.05, 0.15)
                final_values[scenario] = {
                    "value": last_value,
                    "min": round(last_value - error_margin, 2),
                    "max": round(last_value + error_margin, 2)
                }
                
            result["final_values"] = final_values
            result["analysis"] = f"Analyse basée sur {random.randint(3, 10)} facteurs clés. La marge d'erreur moyenne est de ±{random.randint(8, 15)}%. Les variations saisonnières ont été prises en compte, avec des pics traditionnels observés en fin d'année."
        
        return result
    
    def _generate_sentiment_prediction(
        self, 
        request: PredictionRequest, 
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère une prédiction de sentiment simulée.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les résultats de la prédiction
        """
        # Sentiments de base
        sentiments = ["positif", "neutre", "négatif"]
        sentiment_weights = [0.4, 0.3, 0.3]  # Pondération
        overall_sentiment = random.choices(sentiments, weights=sentiment_weights)[0]
        
        # Score de sentiment (0-1)
        sentiment_score = random.uniform(0.5, 0.9) if overall_sentiment == "positif" else (
            random.uniform(0.4, 0.6) if overall_sentiment == "neutre" else random.uniform(0.1, 0.4)
        )
        
        # Créer le résultat de base
        result = {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": round(sentiment_score, 2),
            "confidence": round(random.uniform(0.7, 0.9), 2),
            "summary": f"Le sentiment général concernant {request.topic} est {overall_sentiment} avec un score de {sentiment_score:.2f}."
        }
        
        # Ajouter des détails supplémentaires pour le mode expert
        if mode == "expert":
            # Aspects spécifiques du sentiment
            aspects = ["prix", "qualité", "service", "fiabilité"]
            aspect_sentiments = {}
            
            for aspect in aspects:
                # Score de sentiment pour cet aspect
                aspect_score = random.uniform(0.1, 0.9)
                aspect_sentiment = "positif" if aspect_score > 0.6 else ("neutre" if aspect_score > 0.4 else "négatif")
                
                aspect_sentiments[aspect] = {
                    "sentiment": aspect_sentiment,
                    "score": round(aspect_score, 2),
                    "keywords": [f"mot_clé_{i}" for i in range(1, random.randint(3, 6))]
                }
            
            # Évolution dans le temps
            time_evolution = []
            for i in range(5):
                date = (datetime.now() - timedelta(days=i*7)).strftime("%Y-%m-%d")
                time_evolution.append({
                    "date": date,
                    "sentiment_score": round(random.uniform(0.2, 0.8), 2)
                })
            
            result["aspects"] = aspect_sentiments
            result["time_evolution"] = time_evolution
            result["analysis"] = f"Analyse basée sur {random.randint(500, 2000)} mentions. Les aspects clés montrent que le sentiment sur la qualité est {aspect_sentiments['qualité']['sentiment']} tandis que celui sur le prix est {aspect_sentiments['prix']['sentiment']}. L'évolution sur 5 semaines montre une tendance {'à la hausse' if time_evolution[0]['sentiment_score'] > time_evolution[-1]['sentiment_score'] else 'à la baisse'}."
        
        return result
    
    def _generate_market_prediction(
        self,
        request: PredictionRequest,
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère une prédiction d'analyse de marché.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les résultats de l'analyse de marché
        """
        # Définir quelques segments de marché
        segments = ["premium", "standard", "économique"]
        segment_shares = {}
        
        # Répartir le marché entre les segments
        remaining = 100
        for i, segment in enumerate(segments):
            if i == len(segments) - 1:
                segment_shares[segment] = remaining
            else:
                share = random.randint(10, remaining - (len(segments) - i - 1) * 10)
                segment_shares[segment] = share
                remaining -= share
        
        # Croissance prévue par segment
        growth_rates = {
            "premium": round(random.uniform(2.0, 8.0), 1),
            "standard": round(random.uniform(-1.0, 5.0), 1),
            "économique": round(random.uniform(-3.0, 2.0), 1)
        }
        
        # Déterminer le gagnant et le perdant
        winner = max(growth_rates, key=growth_rates.get)
        loser = min(growth_rates, key=growth_rates.get)
        
        # Créer le résultat de base
        result = {
            "market_size": random.randint(100000, 10000000),
            "currency": "EUR",
            "segments": {
                name: {
                    "share": share,
                    "growth": growth_rates[name]
                } for name, share in segment_shares.items()
            },
            "overall_growth": round(sum(growth_rates[s] * segment_shares[s] / 100 for s in segments), 1),
            "confidence": round(random.uniform(0.65, 0.85), 2),
            "summary": f"Le marché de {request.topic} devrait croître de {round(sum(growth_rates[s] * segment_shares[s] / 100 for s in segments), 1)}% globalement. Le segment {winner} montre la plus forte croissance à {growth_rates[winner]}%."
        }
        
        # Ajouter des détails supplémentaires pour le mode expert
        if mode == "expert":
            # Concurrents principaux
            competitors = [
                {"name": "Leader Inc.", "share": random.randint(15, 35)},
                {"name": "Challenger SA", "share": random.randint(10, 25)},
                {"name": "Innovator Ltd", "share": random.randint(5, 15)},
                {"name": "Newcomer", "share": random.randint(2, 8)},
                {"name": "Autres", "share": 0}  # Sera calculé
            ]
            
            # Ajuster le dernier pour atteindre 100%
            total = sum(c["share"] for c in competitors[:-1])
            competitors[-1]["share"] = 100 - total
            
            # Facteurs d'influence
            factors = [
                {"name": "Innovation produit", "impact": round(random.uniform(0.2, 0.8), 2)},
                {"name": "Réglementation", "impact": round(random.uniform(0.1, 0.6), 2)},
                {"name": "Changements démographiques", "impact": round(random.uniform(0.1, 0.5), 2)},
                {"name": "Tendances économiques", "impact": round(random.uniform(0.3, 0.7), 2)}
            ]
            
            # Prévisions à 5 ans
            forecast = []
            current_size = result["market_size"]
            for i in range(5):
                year = datetime.now().year + i
                growth = result["overall_growth"] + random.uniform(-1.0, 1.0)  # Légères variations
                current_size = current_size * (1 + growth/100)
                
                forecast.append({
                    "year": year,
                    "size": int(current_size),
                    "growth": round(growth, 1)
                })
            
            result["competitors"] = competitors
            result["influence_factors"] = factors
            result["forecast_5y"] = forecast
            result["barriers_to_entry"] = random.choice(["low", "medium", "high"])
            result["market_maturity"] = random.choice(["emerging", "growing", "mature", "declining"])
            result["analysis"] = f"Le marché est considéré comme {result['market_maturity']} avec des barrières à l'entrée {result['barriers_to_entry']}. Les leaders actuels détiennent {competitors[0]['share'] + competitors[1]['share']}% du marché. D'ici 5 ans, la taille du marché devrait atteindre {forecast[-1]['size']} {result['currency']}."
        
        return result
    
    def _generate_risk_prediction(
        self,
        request: PredictionRequest,
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère une évaluation des risques simulée.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les résultats de l'évaluation des risques
        """
        # Définir les catégories de risque
        risk_categories = ["financier", "opérationnel", "stratégique", "conformité", "réputation"]
        
        # Générer des scores de risque pour chaque catégorie
        risk_scores = {}
        risk_levels = ["faible", "modéré", "élevé", "critique"]
        risk_impacts = ["mineur", "significatif", "majeur", "sévère"]
        
        for category in risk_categories:
            score = random.randint(1, 100)
            level_idx = 0 if score < 30 else (1 if score < 60 else (2 if score < 85 else 3))
            
            risk_scores[category] = {
                "score": score,
                "level": risk_levels[level_idx],
                "probability": round(random.uniform(0.1, 0.9), 2),
                "impact": risk_impacts[random.randint(0, 3)]
            }
        
        # Déterminer le risque global
        overall_score = int(sum(r["score"] for r in risk_scores.values()) / len(risk_scores))
        overall_level_idx = 0 if overall_score < 30 else (1 if overall_score < 60 else (2 if overall_score < 85 else 3))
        overall_level = risk_levels[overall_level_idx]
        
        # Créer le résultat de base
        result = {
            "overall_risk": {
                "score": overall_score,
                "level": overall_level
            },
            "risk_categories": risk_scores,
            "confidence": round(random.uniform(0.7, 0.9), 2),
            "summary": f"L'évaluation des risques pour {request.topic} montre un niveau global {overall_level} (score: {overall_score}/100). Le risque {max(risk_scores, key=lambda k: risk_scores[k]['score'])} est le plus préoccupant."
        }
        
        # Ajouter des détails supplémentaires pour le mode expert
        if mode == "expert":
            # Facteurs spécifiques par catégorie
            risk_factors = {}
            for category in risk_categories:
                factors = []
                for i in range(random.randint(2, 4)):
                    impact_level = random.randint(1, 5)
                    factors.append({
                        "name": f"Facteur {i+1} - {category}",
                        "description": f"Description du facteur de risque {i+1} pour la catégorie {category}",
                        "impact_level": impact_level,
                        "trend": random.choice(["increasing", "stable", "decreasing"])
                    })
                risk_factors[category] = factors
            
            # Mesures d'atténuation
            mitigation_strategies = []
            num_strategies = random.randint(3, 6)
            for i in range(num_strategies):
                target_category = random.choice(risk_categories)
                mitigation_strategies.append({
                    "name": f"Stratégie {i+1}",
                    "target_risk": target_category,
                    "effectiveness": round(random.uniform(0.3, 0.9), 2),
                    "cost": random.choice(["low", "medium", "high"]),
                    "implementation_time": random.choice(["short", "medium", "long"])
                })
            
            # Historique des scores de risque
            risk_history = []
            for i in range(4):
                quarter = f"Q{(datetime.now().month-1)//3 - i if (datetime.now().month-1)//3 - i > 0 else (datetime.now().month-1)//3 - i + 4}/{datetime.now().year if (datetime.now().month-1)//3 - i > 0 else datetime.now().year - 1}"
                risk_history.append({
                    "period": quarter,
                    "overall_score": random.randint(max(1, overall_score-20), min(100, overall_score+20))
                })
            
            result["risk_factors"] = risk_factors
            result["mitigation_strategies"] = mitigation_strategies
            result["risk_history"] = risk_history
            result["next_review_date"] = (datetime.now() + timedelta(days=random.randint(30, 180))).strftime("%Y-%m-%d")
            
            # Analyse plus détaillée
            highest_risk = max(risk_scores, key=lambda k: risk_scores[k]['score'])
            high_risk_factors = len([f for f in risk_factors[highest_risk] if f["impact_level"] >= 4])
            result["analysis"] = f"Cette évaluation a identifié {sum(len(factors) for factors in risk_factors.values())} facteurs de risque distincts. La catégorie {highest_risk} présente le risque le plus élevé avec {high_risk_factors} facteurs à fort impact. L'évolution historique montre une tendance {'à la hausse' if risk_history[0]['overall_score'] > risk_history[-1]['overall_score'] else 'à la baisse'}."
        
        return result
    
    def _generate_recommendation_prediction(
        self,
        request: PredictionRequest,
        mode: str
    ) -> Dict[str, Any]:
        """
        Génère des recommandations simulées.
        
        Args:
            request: La demande de prédiction
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Dictionnaire contenant les recommandations
        """
        # Générér des recommandations générales
        recommendations = []
        num_recommendations = random.randint(3, 5)
        
        for i in range(num_recommendations):
            priority = random.choice(["haute", "moyenne", "basse"])
            complexity = random.choice(["simple", "modérée", "complexe"])
            timeframe = random.choice(["court terme", "moyen terme", "long terme"])
            
            recommendations.append({
                "id": f"REC-{random.randint(100, 999)}",
                "title": f"Recommandation {i+1} pour {request.topic}",
                "description": f"Description détaillée de la recommandation {i+1} concernant {request.topic}.",
                "priority": priority,
                "complexity": complexity,
                "timeframe": timeframe,
                "expected_impact": round(random.uniform(0.1, 0.9), 2)
            })
        
        # Trier par priorité et impact
        recommendations.sort(key=lambda x: (-{"haute": 3, "moyenne": 2, "basse": 1}[x["priority"]], -x["expected_impact"]))
        
        # Créer le résultat de base
        result = {
            "recommendations": recommendations,
            "top_recommendation": recommendations[0]["title"] if recommendations else None,
            "confidence": round(random.uniform(0.7, 0.9), 2),
            "summary": f"{num_recommendations} recommandations générées pour {request.topic}. La recommandation principale est: {recommendations[0]['title'] if recommendations else 'Aucune'}."
        }
        
        # Ajouter des détails supplémentaires pour le mode expert
        if mode == "expert":
            # Dépendances entre recommandations
            dependencies = []
            for i, rec in enumerate(recommendations):
                if i < len(recommendations) - 1 and random.random() > 0.5:
                    dependencies.append({
                        "source_id": rec["id"],
                        "target_id": recommendations[i+1]["id"],
                        "type": random.choice(["prerequisite", "enhances", "conflicts"])
                    })
            
            # Ressources nécessaires par recommandation
            for rec in recommendations:
                rec["resources"] = {
                    "budget": random.choice(["faible", "moyen", "élevé"]),
                    "personnel": random.randint(1, 10),
                    "time_days": random.randint(5, 120)
                }
            
            # Bénéfices attendus
            benefits = {
                "cost_reduction": round(random.uniform(0.0, 0.3), 2),
                "efficiency_gain": round(random.uniform(0.05, 0.4), 2),
                "risk_reduction": round(random.uniform(0.1, 0.5), 2),
                "revenue_increase": round(random.uniform(0.0, 0.25), 2)
            }
            
            # Ajouter à chaque recommandation
            for rec in recommendations:
                rec["specific_benefits"] = {k: round(v * rec["expected_impact"], 2) for k, v in benefits.items()}
            
            result["dependencies"] = dependencies
            result["overall_benefits"] = benefits
            result["implementation_plan"] = {
                "phases": random.randint(2, 4),
                "total_duration_days": sum(rec["resources"]["time_days"] for rec in recommendations) * 0.8,  # Avec parallélisation
                "critical_path": [rec["id"] for rec in recommendations if rec["priority"] == "haute"]
            }
            
            # Analyse plus détaillée
            high_priority = len([r for r in recommendations if r["priority"] == "haute"])
            result["analysis"] = f"L'analyse complète a généré {num_recommendations} recommandations dont {high_priority} à haute priorité. La mise en œuvre complète nécessiterait environ {result['implementation_plan']['total_duration_days']:.0f} jours, pour un gain d'efficacité estimé à {benefits['efficiency_gain']*100:.0f}% et une réduction des coûts de {benefits['cost_reduction']*100:.0f}%."
        
        return result

# Fonction utilitaire pour traduire les directions de tendance
def trend_direction_fr(direction):
    translations = {
        "up": "hausse",
        "down": "baisse",
        "stable": "stabilité"
    }
    return translations.get(direction, direction) 
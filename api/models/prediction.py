from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime


class ForecastPoint(BaseModel):
    """Point de données pour une prévision"""
    date: str  # Format ISO
    value: float


class Anomaly(BaseModel):
    """Modèle pour une anomalie détectée"""
    date: str  # Format ISO
    expected: float
    actual: float
    severity: str  # low, medium, high


class SeasonalityPattern(BaseModel):
    """Modèle pour les patterns de saisonnalité"""
    weekly_pattern: Optional[str] = None
    monthly_pattern: Optional[str] = None
    yearly_pattern: Optional[str] = None


class PredictionRequest(BaseModel):
    """Modèle pour une demande de prédiction"""
    prediction_type: str  # trend_prediction, sentiment_analysis, etc.
    topic: Optional[str] = None
    data_points: Optional[List[Dict[str, Any]]] = None
    horizon: Optional[int] = 30  # nombre de jours/périodes à prédire


class TrendPredictionResult(BaseModel):
    """Résultat d'une prédiction de tendance"""
    forecast: List[ForecastPoint]
    trend_direction: str  # up, down, stable
    confidence: float
    seasonality: Optional[SeasonalityPattern] = None
    anomalies: Optional[List[Anomaly]] = None


class PredictionResult(BaseModel):
    """Modèle pour un résultat de prédiction"""
    prediction_type: str
    topic: Optional[str] = None
    results: Dict[str, Any]  # Contenu spécifique selon le type de prédiction
    mode: str  # client ou expert 
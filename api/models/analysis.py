from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class Insight(BaseModel):
    """Modèle pour un point d'analyse"""
    title: str
    description: Optional[str] = None
    summary: Optional[str] = None
    confidence: float
    sources: Optional[List[str]] = None
    related_topics: Optional[List[str]] = None


class AnalysisMetadata(BaseModel):
    """Métadonnées d'analyse pour le mode expert"""
    analysis_depth: str
    sources_count: int
    processing_time: str


class AnalysisRequest(BaseModel):
    """Modèle pour une demande d'analyse"""
    topic: str
    depth: Optional[str] = "standard"  # standard, deep, quick
    context: Optional[str] = None


class AnalysisResultData(BaseModel):
    """Données de résultat d'analyse"""
    insights: List[Insight]
    metadata: Optional[AnalysisMetadata] = None


class AnalysisResult(BaseModel):
    """Modèle pour un résultat d'analyse"""
    topic: str
    results: AnalysisResultData
    mode: str  # client ou expert 
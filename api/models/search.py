from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class SearchResult(BaseModel):
    """Modèle pour un résultat de recherche"""
    id: str
    title: str
    excerpt: str
    score: float
    url: Optional[str] = None
    technical_details: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Modèle pour une demande de recherche"""
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = 10
    offset: Optional[int] = 0


class SearchResponse(BaseModel):
    """Modèle pour une réponse de recherche"""
    query: str
    results: List[SearchResult]
    total: int
    mode: str  # client ou expert 
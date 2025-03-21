from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime


class UserPreferences(BaseModel):
    """Préférences utilisateur"""
    theme: str = "light"  # light, dark
    language: str = "fr"  # fr, en
    notifications_enabled: bool = True
    email_notifications_enabled: bool = False
    expert_mode: bool = False
    dashboard_layout: Optional[Dict[str, Any]] = None


class ActivityType(str, Enum):
    """Types d'activité utilisateur"""
    SEARCH = "search"
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    CRAWL = "crawl"
    LOGIN = "login"
    SETTINGS_CHANGE = "settings_change"


class UserActivity(BaseModel):
    """Activité d'un utilisateur"""
    id: str
    user_id: str
    type: ActivityType
    timestamp: str  # ISO format datetime
    details: Dict[str, Any]


class UserDataResponse(BaseModel):
    """Réponse pour les données utilisateur"""
    preferences: UserPreferences
    recent_activities: List[UserActivity]
    saved_items: List[Dict[str, Any]]


class UserHistoryItem(BaseModel):
    """Élément d'historique utilisateur"""
    id: str
    user_id: str
    type: str  # Type d'élément (search, analysis, etc.)
    created_at: str  # ISO format datetime
    title: str
    summary: Optional[str] = None
    data: Dict[str, Any] 
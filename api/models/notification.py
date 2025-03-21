from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum
from datetime import datetime


class NotificationType(str, Enum):
    """Types de notifications disponibles"""
    INFO = "info"
    WARNING = "warning"
    SUCCESS = "success"
    ERROR = "error"
    ANALYSIS_COMPLETE = "analysis_complete"
    SEARCH_COMPLETE = "search_complete"
    PREDICTION_COMPLETE = "prediction_complete"
    CRAWL_COMPLETE = "crawl_complete"


class Notification(BaseModel):
    """Modèle de notification"""
    id: str
    user_id: str
    title: str
    message: str
    type: NotificationType
    created_at: str  # ISO format datetime
    read: bool = False
    data: Optional[Dict[str, Any]] = None
    

class NotificationCreate(BaseModel):
    """Modèle pour la création d'une notification"""
    user_id: str
    title: str
    message: str
    type: NotificationType = NotificationType.INFO
    data: Optional[Dict[str, Any]] = None


class NotificationsList(BaseModel):
    """Liste des notifications d'un utilisateur"""
    notifications: List[Notification]
    total: int
    unread: int 
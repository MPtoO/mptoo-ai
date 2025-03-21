import logging
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from api.models.user_data import (
    UserPreferences, UserActivity, UserDataResponse, 
    UserHistoryItem, ActivityType
)

logger = logging.getLogger(__name__)

class UserDataService:
    """Service pour gérer les données utilisateur."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Stockage en mémoire (à remplacer par une base de données)
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.user_activities: Dict[str, List[UserActivity]] = {}
        self.user_saved_items: Dict[str, List[Dict[str, Any]]] = {}
        self.user_history: Dict[str, List[UserHistoryItem]] = {}
    
    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """
        Récupère les préférences d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Les préférences de l'utilisateur
        """
        # Si l'utilisateur n'a pas encore de préférences, créer les préférences par défaut
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences()
        
        return self.user_preferences[user_id]
    
    async def update_user_preferences(self, user_id: str, preferences: UserPreferences) -> UserPreferences:
        """
        Met à jour les préférences d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            preferences: Les nouvelles préférences
            
        Returns:
            Les préférences mises à jour
        """
        self.user_preferences[user_id] = preferences
        return self.user_preferences[user_id]
    
    async def add_user_activity(self, user_id: str, activity_type: ActivityType, details: Dict[str, Any]) -> UserActivity:
        """
        Ajoute une activité à l'historique d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            activity_type: Type d'activité
            details: Détails de l'activité
            
        Returns:
            L'activité créée
        """
        activity_id = str(uuid.uuid4())
        
        activity = UserActivity(
            id=activity_id,
            user_id=user_id,
            type=activity_type,
            timestamp=datetime.now().isoformat(),
            details=details
        )
        
        if user_id not in self.user_activities:
            self.user_activities[user_id] = []
        
        self.user_activities[user_id].append(activity)
        
        # Limiter à 100 activités maximum par utilisateur
        if len(self.user_activities[user_id]) > 100:
            self.user_activities[user_id] = self.user_activities[user_id][-100:]
        
        return activity
    
    async def get_user_activities(self, user_id: str, limit: int = 20) -> List[UserActivity]:
        """
        Récupère les activités récentes d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre maximum d'activités à récupérer
            
        Returns:
            Liste des activités récentes
        """
        if user_id not in self.user_activities:
            return []
        
        # Trier par date (plus récent d'abord) et limiter
        sorted_activities = sorted(
            self.user_activities[user_id],
            key=lambda a: a.timestamp,
            reverse=True
        )
        
        return sorted_activities[:limit]
    
    async def add_saved_item(self, user_id: str, item_type: str, item_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sauvegarde un élément pour un utilisateur (recherche, analyse, etc.).
        
        Args:
            user_id: ID de l'utilisateur
            item_type: Type d'élément
            item_data: Données de l'élément
            
        Returns:
            L'élément sauvegardé avec son ID
        """
        item_id = str(uuid.uuid4())
        
        # Créer l'élément sauvegardé
        saved_item = {
            "id": item_id,
            "type": item_type,
            "created_at": datetime.now().isoformat(),
            "data": item_data
        }
        
        if user_id not in self.user_saved_items:
            self.user_saved_items[user_id] = []
        
        self.user_saved_items[user_id].append(saved_item)
        
        return saved_item
    
    async def get_saved_items(self, user_id: str, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Récupère les éléments sauvegardés d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            item_type: Type d'élément à filtrer (optionnel)
            
        Returns:
            Liste des éléments sauvegardés
        """
        if user_id not in self.user_saved_items:
            return []
        
        items = self.user_saved_items[user_id]
        
        # Filtrer par type si spécifié
        if item_type:
            items = [item for item in items if item["type"] == item_type]
        
        # Trier par date (plus récent d'abord)
        sorted_items = sorted(
            items,
            key=lambda item: item["created_at"],
            reverse=True
        )
        
        return sorted_items
    
    async def delete_saved_item(self, user_id: str, item_id: str) -> bool:
        """
        Supprime un élément sauvegardé.
        
        Args:
            user_id: ID de l'utilisateur
            item_id: ID de l'élément à supprimer
            
        Returns:
            True si l'élément a été supprimé, False sinon
        """
        if user_id not in self.user_saved_items:
            return False
        
        initial_count = len(self.user_saved_items[user_id])
        self.user_saved_items[user_id] = [
            item for item in self.user_saved_items[user_id] if item["id"] != item_id
        ]
        
        return len(self.user_saved_items[user_id]) < initial_count
    
    async def add_history_item(self, user_id: str, item: UserHistoryItem) -> UserHistoryItem:
        """
        Ajoute un élément à l'historique de l'utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            item: Élément d'historique
            
        Returns:
            L'élément d'historique créé
        """
        if user_id not in self.user_history:
            self.user_history[user_id] = []
        
        # S'assurer que l'ID de l'utilisateur correspond
        item.user_id = user_id
        
        # Générer un ID si non fourni
        if not item.id:
            item.id = str(uuid.uuid4())
        
        self.user_history[user_id].append(item)
        
        # Limiter à 100 éléments maximum par utilisateur
        if len(self.user_history[user_id]) > 100:
            self.user_history[user_id] = self.user_history[user_id][-100:]
        
        return item
    
    async def get_history(self, user_id: str, item_type: Optional[str] = None, limit: int = 20) -> List[UserHistoryItem]:
        """
        Récupère l'historique d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            item_type: Type d'élément à filtrer (optionnel)
            limit: Nombre maximum d'éléments à récupérer
            
        Returns:
            Liste des éléments d'historique
        """
        if user_id not in self.user_history:
            return []
        
        items = self.user_history[user_id]
        
        # Filtrer par type si spécifié
        if item_type:
            items = [item for item in items if item.type == item_type]
        
        # Trier par date (plus récent d'abord)
        sorted_items = sorted(
            items,
            key=lambda item: item.created_at,
            reverse=True
        )
        
        return sorted_items[:limit]
    
    async def clear_history(self, user_id: str, item_type: Optional[str] = None) -> int:
        """
        Efface l'historique d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            item_type: Type d'élément à effacer (optionnel, tous les types si None)
            
        Returns:
            Nombre d'éléments supprimés
        """
        if user_id not in self.user_history:
            return 0
        
        initial_count = len(self.user_history[user_id])
        
        if item_type:
            # Supprimer uniquement le type spécifié
            self.user_history[user_id] = [
                item for item in self.user_history[user_id] if item.type != item_type
            ]
        else:
            # Supprimer tout l'historique
            self.user_history[user_id] = []
        
        return initial_count - len(self.user_history[user_id])
    
    async def get_user_data(self, user_id: str) -> UserDataResponse:
        """
        Récupère toutes les données d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Toutes les données de l'utilisateur
        """
        preferences = await self.get_user_preferences(user_id)
        activities = await self.get_user_activities(user_id, limit=10)
        saved_items = await self.get_saved_items(user_id)
        
        return UserDataResponse(
            preferences=preferences,
            recent_activities=activities,
            saved_items=saved_items
        ) 
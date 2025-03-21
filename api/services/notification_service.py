import logging
import uuid
from typing import Dict, List, Optional, Set
from datetime import datetime
import asyncio
import json

from fastapi import WebSocket
from api.models.notification import Notification, NotificationCreate, NotificationsList, NotificationType

logger = logging.getLogger(__name__)

class NotificationService:
    """Service pour gérer les notifications des utilisateurs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Stockage en mémoire des notifications (à remplacer par une base de données)
        self.notifications: Dict[str, List[Notification]] = {}
        # Gestionnaire de connexions WebSocket actives
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
    async def create_notification(self, notification: NotificationCreate) -> Notification:
        """
        Crée une nouvelle notification et la notifie à l'utilisateur s'il est connecté.
        
        Args:
            notification: Les données de la notification à créer
            
        Returns:
            La notification créée
        """
        # Générer un ID unique pour la notification
        notification_id = str(uuid.uuid4())
        
        # Créer la notification
        new_notification = Notification(
            id=notification_id,
            user_id=notification.user_id,
            title=notification.title,
            message=notification.message,
            type=notification.type,
            created_at=datetime.now().isoformat(),
            read=False,
            data=notification.data
        )
        
        # Stocker la notification
        if notification.user_id not in self.notifications:
            self.notifications[notification.user_id] = []
        
        self.notifications[notification.user_id].append(new_notification)
        
        # Envoyer la notification en temps réel si l'utilisateur est connecté
        await self.notify_user(notification.user_id, new_notification)
        
        return new_notification
    
    async def get_notifications(self, user_id: str, limit: int = 20, skip: int = 0) -> NotificationsList:
        """
        Récupère les notifications d'un utilisateur.
        
        Args:
            user_id: ID de l'utilisateur
            limit: Nombre maximum de notifications à récupérer
            skip: Nombre de notifications à sauter (pagination)
            
        Returns:
            Une liste de notifications
        """
        user_notifications = self.notifications.get(user_id, [])
        
        # Trier par date de création (plus récent d'abord)
        sorted_notifications = sorted(
            user_notifications, 
            key=lambda n: n.created_at, 
            reverse=True
        )
        
        # Paginer les résultats
        paginated_notifications = sorted_notifications[skip:skip+limit]
        
        # Compter les notifications non lues
        unread_count = sum(1 for n in user_notifications if not n.read)
        
        return NotificationsList(
            notifications=paginated_notifications,
            total=len(user_notifications),
            unread=unread_count
        )
    
    async def mark_as_read(self, user_id: str, notification_id: str) -> Optional[Notification]:
        """
        Marque une notification comme lue.
        
        Args:
            user_id: ID de l'utilisateur
            notification_id: ID de la notification
            
        Returns:
            La notification mise à jour ou None si non trouvée
        """
        if user_id not in self.notifications:
            return None
        
        for notification in self.notifications[user_id]:
            if notification.id == notification_id:
                notification.read = True
                return notification
                
        return None
    
    async def mark_all_as_read(self, user_id: str) -> int:
        """
        Marque toutes les notifications d'un utilisateur comme lues.
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Nombre de notifications marquées comme lues
        """
        if user_id not in self.notifications:
            return 0
        
        count = 0
        for notification in self.notifications[user_id]:
            if not notification.read:
                notification.read = True
                count += 1
                
        return count
    
    async def delete_notification(self, user_id: str, notification_id: str) -> bool:
        """
        Supprime une notification.
        
        Args:
            user_id: ID de l'utilisateur
            notification_id: ID de la notification
            
        Returns:
            True si la notification a été supprimée, False sinon
        """
        if user_id not in self.notifications:
            return False
        
        initial_count = len(self.notifications[user_id])
        self.notifications[user_id] = [
            n for n in self.notifications[user_id] if n.id != notification_id
        ]
        
        return len(self.notifications[user_id]) < initial_count
    
    # Méthodes pour la gestion des WebSockets
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """
        Connecte un utilisateur au service de notification par WebSocket.
        
        Args:
            websocket: La connexion WebSocket
            user_id: ID de l'utilisateur
        """
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
            
        self.active_connections[user_id].add(websocket)
        self.logger.info(f"Client WebSocket connecté pour l'utilisateur {user_id}")
        
        # Envoyer les notifications non lues à la connexion
        unread_notifications = [n for n in self.notifications.get(user_id, []) if not n.read]
        if unread_notifications:
            for notification in unread_notifications:
                await self.send_notification(websocket, notification)
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """
        Déconnecte un utilisateur du service de notification.
        
        Args:
            websocket: La connexion WebSocket
            user_id: ID de l'utilisateur
        """
        if user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            self.logger.info(f"Client WebSocket déconnecté pour l'utilisateur {user_id}")
    
    async def notify_user(self, user_id: str, notification: Notification):
        """
        Notifie un utilisateur en envoyant une notification à toutes ses connexions actives.
        
        Args:
            user_id: ID de l'utilisateur
            notification: La notification à envoyer
        """
        if user_id not in self.active_connections:
            return
        
        active_websockets = list(self.active_connections[user_id])
        for websocket in active_websockets:
            try:
                await self.send_notification(websocket, notification)
            except Exception as e:
                self.logger.error(f"Erreur lors de l'envoi de la notification: {str(e)}")
                self.active_connections[user_id].discard(websocket)
    
    async def send_notification(self, websocket: WebSocket, notification: Notification):
        """
        Envoie une notification à une connexion WebSocket.
        
        Args:
            websocket: La connexion WebSocket
            notification: La notification à envoyer
        """
        try:
            await websocket.send_json(notification.dict())
        except Exception as e:
            self.logger.error(f"Erreur lors de l'envoi de la notification par WebSocket: {str(e)}")
            raise 
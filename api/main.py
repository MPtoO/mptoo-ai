#!/usr/bin/env python3
from flask import Flask, request, jsonify, Response, stream_with_context, g
from flask_cors import CORS
import json
import logging
import uuid
from datetime import datetime
import sys
import os
import requests
from functools import wraps

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import des modèles et services
from models.user import User, UserCreate, UserInDB, UserResponse
from models.token import Token
from models.analysis import AnalysisRequest
from models.search import SearchRequest
from models.prediction import PredictionRequest
from models.crawl import CrawlRequest, CrawlResult
from models.notification import Notification, NotificationCreate, NotificationsList
from models.user_data import (
    UserPreferences, UserActivity, UserDataResponse, 
    UserHistoryItem, ActivityType
)
from models.agent import (
    Agent, AgentCreate, AgentUpdate, AgentType, 
    AgentQuery, AgentResponse
)
from models.scraping import (
    ScrapingConfig, ScrapingTask, ScrapingTaskStatus, ScrapingRequest,
    ScrapingResponse, ScrapingResult
)
from models.fine_tuning import (
    FineTuningConfig, FineTuningTask, FineTuningStatus, FineTuningRequest,
    FineTuningResponse, FineTuningResult, ModelType, ModelPrediction
)

from services.auth_service import AuthService
from services.analysis_service import AnalysisService
from services.search_service import SearchService
from services.prediction_service import PredictionService
from services.crawl_service import CrawlService
from services.notification_service import NotificationService
from services.user_data_service import UserDataService
from services.agent_service import AgentService
from services.scraping_service import ScrapingService
from services.fine_tuning_service import FineTuningService
from services.ollama_service import OllamaService
from services.crewai_service import CrewAIService
from services.chat_service import ChatService

from core.config import settings

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation des services
auth_service = AuthService()
analysis_service = AnalysisService()
search_service = SearchService()
prediction_service = PredictionService()
crawl_service = CrawlService()
notification_service = NotificationService()
user_data_service = UserDataService()
agent_service = AgentService()
scraping_service = ScrapingService()
fine_tuning_service = FineTuningService()
ollama_service = OllamaService()
crewai_service = CrewAIService()
chat_service = ChatService()

app = Flask(__name__)

# Configuration CORS
CORS(app, origins=settings.BACKEND_CORS_ORIGINS, supports_credentials=True)

# Décorateur pour l'authentification
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        try:
            current_user = auth_service.get_current_user_sync(token)
            return f(current_user=current_user, *args, **kwargs)
        except Exception as e:
            return jsonify({"detail": str(e)}), 401
    return decorated

# Middleware pour détecter le mode d'interface
@app.before_request
def detect_interface_mode():
    g.interface_mode = request.headers.get("X-Interface-Mode", "client")

# Routes d'authentification
@app.route("/token", methods=["POST"])
def login_for_access_token():
    form_data = request.form
    user = auth_service.authenticate_user_sync(form_data.get("username"), form_data.get("password"))
    if not user:
        return jsonify({"detail": "Nom d'utilisateur ou mot de passe incorrect"}), 401
    
    access_token = auth_service.create_access_token_sync(data={"sub": user.username})
    
    # Ajouter une activité de connexion
    if user.id:
        user_data_service.add_user_activity_sync(
            user.id, 
            ActivityType.LOGIN, 
            {"username": user.username}
        )
    
    return jsonify(access_token)

@app.route("/users/", methods=["POST"])
def register_user():
    data = request.json
    user_create = UserCreate(**data)
    return jsonify(auth_service.register_user_sync(user_create))

@app.route("/admin/setup", methods=["POST"])
def setup_first_admin():
    """
    Crée le premier administrateur du système.
    Cette route ne fonctionne que si aucun utilisateur n'existe encore dans la base de données.
    """
    # Vérifier si aucun utilisateur n'existe
    if len(auth_service.users_db) > 0:
        return jsonify({"detail": "La configuration initiale a déjà été effectuée"}), 400
    
    # Forcer le statut d'administrateur pour le premier utilisateur
    user_admin = UserCreate(**request.json)
    user_admin.is_admin = True
    user_admin.can_use_ai_analysis = True
    user_admin.can_use_predictions = True
    
    return jsonify(auth_service.register_user_sync(user_admin))

@app.route("/admin/create", methods=["POST"])
@require_auth
def create_admin_user(current_user):
    """
    Crée un nouvel utilisateur administrateur.
    Seuls les utilisateurs déjà administrateurs peuvent créer d'autres administrateurs.
    """
    # Vérifier si l'utilisateur actuel est un administrateur
    if not current_user.is_admin:
        return jsonify({"detail": "Seuls les administrateurs peuvent créer d'autres administrateurs"}), 403
    
    # Forcer le statut d'administrateur
    user_admin = UserCreate(**request.json)
    user_admin.is_admin = True
    
    return jsonify(auth_service.register_user_sync(user_admin))

@app.route("/users/me", methods=["GET"])
@require_auth
def read_users_me(current_user):
    return jsonify(current_user)

# Routes principales
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": f"Bienvenue sur l'API {settings.PROJECT_NAME}"})

@app.route("/analyze", methods=["POST"])
@require_auth
def analyze_content(current_user):
    # Vérifier les permissions
    if not current_user.can_use_ai_analysis:
        return jsonify({"detail": "Vous n'avez pas l'autorisation d'utiliser l'analyse IA"}), 403
    
    data = request.json
    analysis_request = AnalysisRequest(**data)
    interface_mode = g.interface_mode
    result = analysis_service.analyze_content_sync(analysis_request, interface_mode)
    
    # Enregistrer l'activité
    if current_user.id:
        # Ajouter à l'historique des activités
        user_data_service.add_user_activity_sync(
            current_user.id,
            ActivityType.ANALYSIS,
            {"topic": analysis_request.topic, "depth": analysis_request.depth}
        )
        
        # Ajouter à l'historique de l'utilisateur
        history_item = UserHistoryItem(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            type="analysis",
            created_at=datetime.now().isoformat(),
            title=f"Analyse de '{analysis_request.topic}'",
            summary=f"Analyse {analysis_request.depth} sur le sujet '{analysis_request.topic}'",
            data={"topic": analysis_request.topic, "depth": analysis_request.depth}
        )
        user_data_service.add_history_item_sync(current_user.id, history_item)
        
        # Créer une notification
        notification = NotificationCreate(
            user_id=current_user.id,
            title="Analyse terminée",
            message=f"Votre analyse sur '{analysis_request.topic}' est terminée.",
            type="analysis_complete",
            data={"topic": analysis_request.topic}
        )
        notification_service.create_notification_sync(notification)
    
    return jsonify(result)

@app.route("/search", methods=["POST"])
@require_auth
def search_content(current_user):
    data = request.json
    search_request = SearchRequest(**data)
    interface_mode = g.interface_mode
    result = search_service.search_content_sync(search_request, interface_mode)
    
    # Enregistrer l'activité
    if current_user.id:
        # Ajouter à l'historique des activités
        user_data_service.add_user_activity_sync(
            current_user.id,
            ActivityType.SEARCH,
            {"query": search_request.query}
        )
        
        # Ajouter à l'historique de l'utilisateur
        history_item = UserHistoryItem(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            type="search",
            created_at=datetime.now().isoformat(),
            title=f"Recherche pour '{search_request.query}'",
            summary=f"Recherche de '{search_request.query}'",
            data={"query": search_request.query}
        )
        user_data_service.add_history_item_sync(current_user.id, history_item)
        
        # Créer une notification
        notification = NotificationCreate(
            user_id=current_user.id,
            title="Recherche terminée",
            message=f"Votre recherche pour '{search_request.query}' est terminée.",
            type="search_complete",
            data={"query": search_request.query}
        )
        notification_service.create_notification_sync(notification)
    
    return jsonify(result)

@app.route("/predict", methods=["POST"])
@require_auth
def predict(current_user):
    # Vérifier les permissions
    if not current_user.can_use_predictions:
        return jsonify({"detail": "Vous n'avez pas l'autorisation d'utiliser les prédictions"}), 403
    
    data = request.json
    prediction_request = PredictionRequest(**data)
    interface_mode = g.interface_mode
    result = prediction_service.predict_sync(prediction_request, interface_mode)
    
    # Enregistrer l'activité
    if current_user.id:
        # Ajouter à l'historique des activités
        user_data_service.add_user_activity_sync(
            current_user.id,
            ActivityType.PREDICTION,
            {"prediction_type": prediction_request.prediction_type, "topic": prediction_request.topic}
        )
        
        # Ajouter à l'historique de l'utilisateur
        history_item = UserHistoryItem(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            type="prediction",
            created_at=datetime.now().isoformat(),
            title=f"Prédiction '{prediction_request.prediction_type}'",
            summary=f"Prédiction {prediction_request.prediction_type} sur '{prediction_request.topic or 'NA'}'",
            data={"prediction_type": prediction_request.prediction_type, "topic": prediction_request.topic}
        )
        user_data_service.add_history_item_sync(current_user.id, history_item)
        
        # Créer une notification
        notification = NotificationCreate(
            user_id=current_user.id,
            title="Prédiction terminée",
            message=f"Votre prédiction '{prediction_request.prediction_type}' est terminée.",
            type="prediction_complete",
            data={"prediction_type": prediction_request.prediction_type}
        )
        notification_service.create_notification_sync(notification)
    
    return jsonify(result)

@app.route("/crawl/target", methods=["POST"])
@require_auth
def add_crawl_target(current_user):
    data = request.json
    result = crawl_service.add_crawl_target_sync(data)
    
    # Enregistrer l'activité
    if current_user.id:
        # Ajouter à l'historique des activités
        user_data_service.add_user_activity_sync(
            current_user.id,
            ActivityType.CRAWL,
            {"url": str(data['url']), "target_id": result.target_id}
        )
        
        # Créer une notification
        notification = NotificationCreate(
            user_id=current_user.id,
            title="Crawl démarré",
            message=f"Le crawl de {data['url']} a démarré.",
            type="info",
            data={"target_id": result.target_id, "url": str(data['url'])}
        )
        notification_service.create_notification_sync(notification)
    
    return jsonify(result)

@app.route("/crawl/status/<target_id>", methods=["GET"])
@require_auth
def get_crawl_status(current_user):
    result = crawl_service.get_crawl_status_sync(target_id)
    if not result:
        return jsonify({"detail": f"Crawl avec ID {target_id} non trouvé"}), 404
    return jsonify(result)

# Routes pour les notifications
@app.route("/notifications", methods=["GET"])
@require_auth
def get_notifications(current_user):
    """Récupère les notifications de l'utilisateur courant."""
    if not current_user.id:
        return jsonify(notifications=[], total=0, unread=0)
    
    return jsonify(notification_service.get_notifications_sync(current_user.id))

@app.route("/notifications/<notification_id>/read", methods=["PUT"])
@require_auth
def mark_notification_as_read(current_user):
    """Marque une notification comme lue."""
    if not current_user.id:
        return jsonify({"detail": "Notification non trouvée"}), 404
    
    notification = notification_service.mark_as_read_sync(current_user.id, request.json['notification_id'])
    if not notification:
        return jsonify({"detail": "Notification non trouvée"}), 404
    
    return jsonify(notification)

@app.route("/notifications/read-all", methods=["PUT"])
@require_auth
def mark_all_notifications_as_read(current_user):
    """Marque toutes les notifications comme lues."""
    if not current_user.id:
        return jsonify({"marked_count": 0})
    
    count = notification_service.mark_all_as_read_sync(current_user.id)
    return jsonify({"marked_count": count})

@app.route("/notifications/<notification_id>", methods=["DELETE"])
@require_auth
def delete_notification(current_user):
    """Supprime une notification."""
    if not current_user.id:
        return jsonify({"success": False})
    
    success = notification_service.delete_notification_sync(current_user.id, request.json['notification_id'])
    return jsonify({"success": success})

# WebSocket pour les notifications en temps réel
@app.route("/ws/notifications", methods=["GET"])
def websocket_notifications():
    """Endpoint WebSocket pour recevoir des notifications en temps réel."""
    # Accepter la connexion
    auth_data = request.json
    token = auth_data.get("token")
    
    # Vérifier le token
    try:
        user = auth_service.get_current_user_sync(token)
        if not user or not user.id:
            return jsonify({"code": 1008, "reason": "Authentification échouée"}), 401
    except Exception as e:
        return jsonify({"code": 1008, "reason": "Authentification échouée"}), 401
    
    # Connecter l'utilisateur au service de notification
    notification_service.connect_sync(request.environ['wsgi.websocket'], user.id)
    
    try:
        # Boucle de maintien de la connexion
        while True:
            # Attendre des messages (ping/pong ou commandes)
            message = request.environ['wsgi.websocket'].receive_text()
            # On pourrait implémenter d'autres commandes ici
            
    except Exception as e:
        logger.error(f"Erreur WebSocket: {str(e)}")
        try:
            request.environ['wsgi.websocket'].close(code=1011, reason="Erreur interne")
        except:
            pass

# Routes pour les données utilisateur
@app.route("/user/data", methods=["GET"])
@require_auth
def get_user_data(current_user):
    """Récupère toutes les données de l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    return jsonify(user_data_service.get_user_data_sync(current_user.id))

@app.route("/user/preferences", methods=["GET"])
@require_auth
def get_user_preferences(current_user):
    """Récupère les préférences de l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    return jsonify(user_data_service.get_user_preferences_sync(current_user.id))

@app.route("/user/preferences", methods=["PUT"])
@require_auth
def update_user_preferences(current_user):
    """Met à jour les préférences de l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    # Ajouter une activité
    user_data_service.add_user_activity_sync(
        current_user.id,
        ActivityType.SETTINGS_CHANGE,
        {"preferences": request.json}
    )
    
    return jsonify(user_data_service.update_user_preferences_sync(current_user.id, request.json))

@app.route("/user/history", methods=["GET"])
@require_auth
def get_user_history(current_user):
    """Récupère l'historique de l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    return jsonify(user_data_service.get_history_sync(current_user.id, request.args.get('type'), request.args.get('limit')))

@app.route("/user/history", methods=["DELETE"])
@require_auth
def clear_user_history(current_user):
    """Efface l'historique de l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    count = user_data_service.clear_history_sync(current_user.id, request.args.get('type'))
    return jsonify({"deleted_count": count})

@app.route("/user/saved-items", methods=["GET"])
@require_auth
def get_saved_items(current_user):
    """Récupère les éléments sauvegardés de l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    return jsonify(user_data_service.get_saved_items_sync(current_user.id, request.args.get('type')))

@app.route("/user/saved-items", methods=["POST"])
@require_auth
def save_item(current_user):
    """Sauvegarde un élément pour l'utilisateur courant."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    return jsonify(user_data_service.add_saved_item_sync(current_user.id, request.json['item_type'], request.json['item_data']))

@app.route("/user/saved-items/<item_id>", methods=["DELETE"])
@require_auth
def delete_saved_item(current_user):
    """Supprime un élément sauvegardé."""
    if not current_user.id:
        return jsonify({"detail": "Utilisateur non authentifié"}), 401
    
    success = user_data_service.delete_saved_item_sync(current_user.id, request.json['item_id'])
    return jsonify({"success": success})

@app.route("/status", methods=["GET"])
def get_status():
    """Endpoint simple pour vérifier si l'API est en ligne"""
    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.now().isoformat()
    })

# Ajouter de nouvelles routes pour les agents IA
@app.route("/agents", methods=["POST"])
@require_auth
def create_agent(current_user):
    """Crée un nouvel agent IA."""
    return jsonify(agent_service.create_agent_sync(request.json, current_user.id))

@app.route("/agents", methods=["GET"])
@require_auth
def get_agents(current_user):
    """Récupère tous les agents disponibles pour l'utilisateur."""
    return jsonify(agent_service.get_agents_sync(current_user.id, request.args.get('agent_type')))

@app.route("/agents/<agent_id>", methods=["GET"])
@require_auth
def get_agent(current_user):
    """Récupère un agent par son ID."""
    agent = agent_service.get_agent_sync(agent_id)
    if not agent:
        return jsonify({"detail": "Agent non trouvé"}), 404
    return jsonify(agent)

@app.route("/agents/<agent_id>", methods=["PUT"])
@require_auth
def update_agent(current_user):
    """Met à jour un agent existant."""
    updated_agent = agent_service.update_agent_sync(agent_id, request.json)
    if not updated_agent:
        return jsonify({"detail": "Agent non trouvé"}), 404
    return jsonify(updated_agent)

@app.route("/agents/<agent_id>", methods=["DELETE"])
@require_auth
def delete_agent(current_user):
    """Supprime un agent."""
    success = agent_service.delete_agent_sync(agent_id)
    if not success:
        return jsonify({"detail": "Agent non trouvé ou suppression impossible"}), 404
    return jsonify({"success": success})

@app.route("/agents/<agent_id>/query", methods=["POST"])
@require_auth
def query_agent(current_user):
    """Envoie une requête à un agent."""
    try:
        return jsonify(agent_service.query_agent_sync(agent_id, request.json))
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400

# Ajouter des routes pour le scraping web
@app.route("/scraping/tasks", methods=["POST"])
@require_auth
def create_scraping_task(current_user):
    """Crée une nouvelle tâche de scraping."""
    return jsonify(scraping_service.create_task_sync(request.json, current_user.id))

@app.route("/scraping/tasks", methods=["GET"])
@require_auth
def get_scraping_tasks(current_user):
    """Récupère toutes les tâches de scraping de l'utilisateur."""
    return jsonify(scraping_service.get_tasks_sync(current_user.id, request.args.get('status')))

@app.route("/scraping/tasks/<task_id>", methods=["GET"])
@require_auth
def get_scraping_task(current_user):
    """Récupère une tâche de scraping par son ID."""
    task = scraping_service.get_task_sync(task_id)
    if not task:
        return jsonify({"detail": "Tâche non trouvée"}), 404
    return jsonify(task)

@app.route("/scraping/tasks/<task_id>/result", methods=["GET"])
@require_auth
def get_scraping_result(current_user):
    """Récupère le résultat d'une tâche de scraping."""
    result = scraping_service.get_task_result_sync(task_id)
    if not result:
        return jsonify({"detail": "Résultat non trouvé ou tâche non terminée"}), 404
    return jsonify(result)

@app.route("/scraping/tasks/<task_id>/start", methods=["POST"])
@require_auth
def start_scraping_task(current_user):
    """Démarre une tâche de scraping en attente."""
    success = scraping_service.start_task_sync(task_id)
    if not success:
        return jsonify({"detail": "Impossible de démarrer la tâche"}), 400
    return jsonify({"success": success})

@app.route("/scraping/tasks/<task_id>/cancel", methods=["POST"])
@require_auth
def cancel_scraping_task(current_user):
    """Annule une tâche de scraping en cours ou en attente."""
    success = scraping_service.cancel_task_sync(task_id)
    if not success:
        return jsonify({"detail": "Impossible d'annuler la tâche"}), 400
    return jsonify({"success": success})

@app.route("/scraping/tasks/<task_id>", methods=["DELETE"])
@require_auth
def delete_scraping_task(current_user):
    """Supprime une tâche de scraping."""
    success = scraping_service.delete_task_sync(task_id)
    if not success:
        return jsonify({"detail": "Impossible de supprimer la tâche"}), 400
    return jsonify({"success": success})

# Ajouter des routes pour le fine-tuning
@app.route("/fine-tuning/tasks", methods=["POST"])
@require_auth
def create_fine_tuning_task(current_user):
    """Crée une nouvelle tâche de fine-tuning."""
    # Vérifier les permissions de l'utilisateur
    if not current_user.is_admin:
        return jsonify({"detail": "Seuls les administrateurs peuvent créer des tâches de fine-tuning"}), 403
    
    return jsonify(fine_tuning_service.create_task_sync(request.json, current_user.id))

@app.route("/fine-tuning/tasks", methods=["GET"])
@require_auth
def get_fine_tuning_tasks(current_user):
    """Récupère toutes les tâches de fine-tuning de l'utilisateur."""
    return jsonify(fine_tuning_service.get_tasks_sync(current_user.id, request.args.get('status')))

@app.route("/fine-tuning/tasks/<task_id>", methods=["GET"])
@require_auth
def get_fine_tuning_task(current_user):
    """Récupère une tâche de fine-tuning par son ID."""
    task = fine_tuning_service.get_task_sync(task_id)
    if not task:
        return jsonify({"detail": "Tâche non trouvée"}), 404
    return jsonify(task)

@app.route("/fine-tuning/tasks/<task_id>/result", methods=["GET"])
@require_auth
def get_fine_tuning_result(current_user):
    """Récupère le résultat d'une tâche de fine-tuning."""
    result = fine_tuning_service.get_task_result_sync(task_id)
    if not result:
        return jsonify({"detail": "Résultat non trouvé ou tâche non terminée"}), 404
    return jsonify(result)

@app.route("/fine-tuning/models", methods=["GET"])
@require_auth
def get_fine_tuned_models(current_user):
    """Récupère tous les modèles fine-tunés disponibles."""
    return jsonify(fine_tuning_service.get_models_sync(current_user.id, request.args.get('model_type')))

@app.route("/fine-tuning/models/<model_id>", methods=["GET"])
@require_auth
def get_fine_tuned_model(current_user):
    """Récupère un modèle fine-tuné par son ID."""
    model = fine_tuning_service.get_model_sync(model_id)
    if not model:
        return jsonify({"detail": "Modèle non trouvé"}), 404
    return jsonify(model)

@app.route("/fine-tuning/models/<model_id>/predict", methods=["POST"])
@require_auth
def predict_with_fine_tuned_model(current_user):
    """Génère une prédiction avec un modèle fine-tuné."""
    try:
        return jsonify(fine_tuning_service.predict_sync(request.json['model_id'], request.json['input_data']))
    except ValueError as e:
        return jsonify({"detail": str(e)}), 400

@app.route("/fine-tuning/tasks/<task_id>/start", methods=["POST"])
@require_auth
def start_fine_tuning_task(current_user):
    """Démarre une tâche de fine-tuning en attente."""
    # Vérifier les permissions de l'utilisateur
    if not current_user.is_admin:
        return jsonify({"detail": "Seuls les administrateurs peuvent démarrer des tâches de fine-tuning"}), 403
    
    success = fine_tuning_service.start_task_sync(task_id)
    if not success:
        return jsonify({"detail": "Impossible de démarrer la tâche"}), 400
    return jsonify({"success": success})

@app.route("/fine-tuning/tasks/<task_id>/cancel", methods=["POST"])
@require_auth
def cancel_fine_tuning_task(current_user):
    """Annule une tâche de fine-tuning en cours ou en attente."""
    # Vérifier les permissions de l'utilisateur
    if not current_user.is_admin:
        return jsonify({"detail": "Seuls les administrateurs peuvent annuler des tâches de fine-tuning"}), 403
    
    success = fine_tuning_service.cancel_task_sync(task_id)
    if not success:
        return jsonify({"detail": "Impossible d'annuler la tâche"}), 400
    return jsonify({"success": success})

@app.route("/fine-tuning/tasks/<task_id>", methods=["DELETE"])
@require_auth
def delete_fine_tuning_task(current_user):
    """Supprime une tâche de fine-tuning."""
    # Vérifier les permissions de l'utilisateur
    if not current_user.is_admin:
        return jsonify({"detail": "Seuls les administrateurs peuvent supprimer des tâches de fine-tuning"}), 403
    
    success = fine_tuning_service.delete_task_sync(task_id)
    if not success:
        return jsonify({"detail": "Impossible de supprimer la tâche"}), 400
    return jsonify({"success": success})

# Route proxy pour la liste phishing
@app.route("/api/proxy/phishing-list", methods=["GET"])
def proxy_phishing_list():
    response = requests.get("https://urlhaus.abuse.ch/downloads/text/", stream=True)
    
    def generate():
        for chunk in response.iter_content(chunk_size=8192):
            yield chunk
    
    return Response(
        stream_with_context(generate()),
        content_type=response.headers.get('content-type')
    )

# Routes pour Ollama
@app.route("/api/llm/models", methods=["GET"])
@require_auth
def list_ollama_models(current_user):
    """Liste tous les modèles disponibles dans Ollama."""
    return jsonify(ollama_service.list_models_sync())

@app.route("/api/llm/generate", methods=["POST"])
@require_auth
def generate_with_ollama(current_user):
    """Génère du texte avec un modèle Ollama."""
    data = request.json
    return jsonify(ollama_service.generate_sync(
        model=data.get("model", "llama2"),
        prompt=data.get("prompt", ""),
        options=data.get("options")
    ))

@app.route("/api/llm/chat", methods=["POST"])
@require_auth
def chat_with_ollama(current_user):
    """Utilise l'API de chat d'Ollama."""
    data = request.json
    return jsonify(ollama_service.chat_sync(
        model=data.get("model", "llama2"),
        messages=data.get("messages", []),
        options=data.get("options")
    ))

# Routes pour CrewAI
@app.route("/api/crew/agents", methods=["POST"])
@require_auth
def create_crew_agent(current_user):
    """Crée un nouvel agent IA."""
    return jsonify(crewai_service.create_agent_sync(request.json))

@app.route("/api/crew/crews", methods=["POST"])
@require_auth
def create_crew(current_user):
    """Crée une nouvelle équipe d'agents IA."""
    return jsonify(crewai_service.create_crew_sync(request.json))

@app.route("/api/crew/crews/<crew_id>/run", methods=["POST"])
@require_auth
def run_crew(current_user, crew_id):
    """Exécute une équipe d'agents IA."""
    return jsonify(crewai_service.run_crew_sync(crew_id, request.json))

# Routes pour le Chat
@app.route("/api/chat/conversations", methods=["POST"])
@require_auth
def create_chat_conversation(current_user):
    """Crée une nouvelle conversation."""
    return jsonify(chat_service.create_conversation_sync(current_user.id, request.json))

@app.route("/api/chat/conversations/<conversation_id>/messages", methods=["POST"])
@require_auth
def add_chat_message(current_user, conversation_id):
    """Ajoute un message à une conversation."""
    data = request.json
    return jsonify(chat_service.add_message_sync(
        user_id=current_user.id,
        conversation_id=conversation_id,
        message=data.get("message", ""),
        sender=data.get("sender", "user")
    ))

@app.route("/api/chat/conversations/<conversation_id>/generate", methods=["POST"])
@require_auth
def generate_chat_response(current_user, conversation_id):
    """Génère une réponse pour la conversation."""
    return jsonify(chat_service.generate_response_sync(
        user_id=current_user.id,
        conversation_id=conversation_id
    ))

# Point d'entrée compatible avec o2switch
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))

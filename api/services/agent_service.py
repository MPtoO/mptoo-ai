import logging
import time
import uuid
import json
import os
import random
from typing import Dict, List, Any, Optional
from datetime import datetime

from api.models.agent import (
    Agent, AgentCreate, AgentUpdate, AgentStatus, AgentType, 
    AgentQuery, AgentResponse, AgentMessage, AgentCapability
)

logger = logging.getLogger(__name__)

class AgentService:
    """Service pour la gestion des agents AI."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agents_db = {}  # Simulation de base de données
        self.conversations = {}  # Historique des conversations par agent
        self.dev_mode = os.environ.get("ENVIRONMENT", "development") == "development"
        
        # Charger les agents prédéfinis en mode développement
        if self.dev_mode:
            self._load_predefined_agents()
    
    def _load_predefined_agents(self):
        """Charge des agents prédéfinis pour le développement."""
        self.logger.info("Chargement des agents prédéfinis pour le développement")
        
        # Exemple d'agent de recherche
        researcher_id = str(uuid.uuid4())
        self.agents_db[researcher_id] = Agent(
            id=researcher_id,
            name="Assistant de recherche",
            description="Agent spécialisé dans la recherche d'informations",
            agent_type=AgentType.RESEARCHER,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            settings={
                "max_tokens": 2000,
                "temperature": 0.3,
                "model_name": "research-model",
                "capabilities": [
                    AgentCapability.WEB_SEARCH,
                    AgentCapability.DOCUMENT_PROCESSING
                ]
            }
        )
        
        # Exemple d'agent d'analyse
        analyzer_id = str(uuid.uuid4())
        self.agents_db[analyzer_id] = Agent(
            id=analyzer_id,
            name="Analyste de données",
            description="Agent spécialisé dans l'analyse de données structurées",
            agent_type=AgentType.ANALYZER,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            settings={
                "max_tokens": 3000,
                "temperature": 0.2,
                "model_name": "analysis-model",
                "capabilities": [
                    AgentCapability.DATA_ANALYSIS,
                    AgentCapability.DOCUMENT_PROCESSING
                ]
            }
        )
        
        # Exemple d'agent de scraping
        scraper_id = str(uuid.uuid4())
        self.agents_db[scraper_id] = Agent(
            id=scraper_id,
            name="Web Scraper",
            description="Agent spécialisé dans l'extraction de données web",
            agent_type=AgentType.SCRAPER,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            settings={
                "max_tokens": 1500,
                "temperature": 0.1,
                "model_name": "scraper-model",
                "capabilities": [
                    AgentCapability.WEB_SCRAPING
                ]
            }
        )
    
    async def create_agent(self, agent_create: AgentCreate, user_id: str) -> Agent:
        """
        Crée un nouvel agent.
        
        Args:
            agent_create: Données pour la création de l'agent
            user_id: ID de l'utilisateur créant l'agent
            
        Returns:
            Le nouvel agent créé
        """
        agent_id = str(uuid.uuid4())
        now = datetime.now()
        
        agent = Agent(
            id=agent_id,
            name=agent_create.name,
            description=agent_create.description,
            agent_type=agent_create.agent_type,
            settings=agent_create.settings,
            created_at=now,
            updated_at=now,
            created_by=user_id
        )
        
        self.agents_db[agent_id] = agent
        self.conversations[agent_id] = []
        
        self.logger.info(f"Agent créé: {agent.name} (ID: {agent_id}, Type: {agent.agent_type})")
        
        # Si des données d'entraînement sont fournies, démarrer le fine-tuning
        if agent_create.training_data:
            self.logger.info(f"Démarrage du fine-tuning pour l'agent {agent_id}")
            agent.status = AgentStatus.TRAINING
            # Dans une implémentation réelle, le fine-tuning serait asynchrone
            # En mode simulation, on peut juste attendre un peu
            if not self.dev_mode:
                await self._start_fine_tuning(agent_id, agent_create.training_data)
        
        return agent
    
    async def update_agent(self, agent_id: str, agent_update: AgentUpdate) -> Optional[Agent]:
        """
        Met à jour un agent existant.
        
        Args:
            agent_id: ID de l'agent à mettre à jour
            agent_update: Modifications à appliquer
            
        Returns:
            L'agent mis à jour ou None si non trouvé
        """
        if agent_id not in self.agents_db:
            return None
        
        agent = self.agents_db[agent_id]
        
        # Mettre à jour les champs fournis
        if agent_update.name is not None:
            agent.name = agent_update.name
        
        if agent_update.description is not None:
            agent.description = agent_update.description
        
        if agent_update.settings is not None:
            agent.settings = agent_update.settings
        
        if agent_update.status is not None:
            agent.status = agent_update.status
        
        agent.updated_at = datetime.now()
        
        return agent
    
    async def delete_agent(self, agent_id: str) -> bool:
        """
        Supprime un agent.
        
        Args:
            agent_id: ID de l'agent à supprimer
            
        Returns:
            True si supprimé avec succès, False sinon
        """
        if agent_id not in self.agents_db:
            return False
        
        # En production, on pourrait simplement marquer l'agent comme supprimé
        # sans le retirer complètement de la base de données
        agent = self.agents_db[agent_id]
        agent.status = AgentStatus.DELETED
        
        # Pour la simulation, on peut aussi le retirer complètement
        # del self.agents_db[agent_id]
        # if agent_id in self.conversations:
        #     del self.conversations[agent_id]
        
        return True
    
    async def get_agent(self, agent_id: str) -> Optional[Agent]:
        """
        Récupère un agent par son ID.
        
        Args:
            agent_id: ID de l'agent
            
        Returns:
            L'agent ou None si non trouvé
        """
        return self.agents_db.get(agent_id)
    
    async def get_agents(self, user_id: Optional[str] = None, agent_type: Optional[AgentType] = None) -> List[Agent]:
        """
        Récupère tous les agents, avec filtrage optionnel.
        
        Args:
            user_id: ID de l'utilisateur pour filtrer ses agents
            agent_type: Type d'agent pour filtrer par type
            
        Returns:
            Liste des agents correspondant aux critères
        """
        agents = list(self.agents_db.values())
        
        # Filtrer par utilisateur si spécifié
        if user_id:
            agents = [a for a in agents if a.created_by == user_id]
        
        # Filtrer par type si spécifié
        if agent_type:
            agents = [a for a in agents if a.agent_type == agent_type]
        
        # Ne pas inclure les agents supprimés
        agents = [a for a in agents if a.status != AgentStatus.DELETED]
        
        return agents
    
    async def query_agent(self, agent_id: str, query: AgentQuery) -> AgentResponse:
        """
        Envoie une requête à un agent et récupère sa réponse.
        
        Args:
            agent_id: ID de l'agent à interroger
            query: Requête à envoyer
            
        Returns:
            Réponse de l'agent
        """
        agent = await self.get_agent(agent_id)
        if not agent or agent.status != AgentStatus.ACTIVE:
            raise ValueError(f"Agent non disponible: {agent_id}")
        
        start_time = time.time()
        
        # Ajouter le message au contexte de conversation
        user_message = AgentMessage(
            content=query.message,
            role="user",
            timestamp=datetime.now()
        )
        
        if agent_id not in self.conversations:
            self.conversations[agent_id] = []
        
        self.conversations[agent_id].append(user_message)
        
        # Limiter le contexte à la taille de fenêtre configurée
        context_window = agent.settings.context_window
        context = self.conversations[agent_id][-context_window:]
        
        # Pour la simulation, nous générons des réponses adaptées au type d'agent
        response_content = self._generate_mock_response(agent, query.message)
        
        # Simulation de la latence du modèle
        processing_time = random.uniform(0.5, 2.0)
        time.sleep(min(0.5, processing_time))  # Latence réduite pour le développement
        
        # Créer le message de réponse
        assistant_message = AgentMessage(
            content=response_content,
            role="assistant",
            timestamp=datetime.now()
        )
        
        # Ajouter la réponse au contexte de conversation
        self.conversations[agent_id].append(assistant_message)
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        
        # Simuler l'utilisation de tokens
        message_length = len(query.message)
        response_length = len(response_content)
        token_usage = {
            "prompt_tokens": int(message_length / 4),
            "completion_tokens": int(response_length / 4),
            "total_tokens": int((message_length + response_length) / 4)
        }
        
        # Incrémenter le compteur d'utilisation de l'agent
        agent.usage_count += 1
        
        # Générer des sources simulées
        sources = self._generate_mock_sources(agent, query.message)
        
        # Créer et retourner la réponse complète
        return AgentResponse(
            message=assistant_message,
            sources=sources,
            thinking=self._generate_mock_thinking(agent, query.message),
            execution_time=execution_time,
            token_usage=token_usage
        )
    
    async def _start_fine_tuning(self, agent_id: str, training_data: Any):
        """
        Démarre le processus de fine-tuning pour un agent.
        
        Args:
            agent_id: ID de l'agent à fine-tuner
            training_data: Données d'entraînement
        """
        # En mode développement, simuler un fine-tuning rapide
        if self.dev_mode:
            time.sleep(2)
            agent = self.agents_db[agent_id]
            agent.status = AgentStatus.ACTIVE
            agent.updated_at = datetime.now()
            agent.performance_metrics = {
                "training_steps": random.randint(100, 500),
                "validation_loss": round(random.uniform(0.01, 0.1), 4),
                "training_completed": datetime.now().isoformat()
            }
            return
        
        # Implémentation réelle du fine-tuning ici
        # ...
    
    def _generate_mock_response(self, agent: Agent, query: str) -> str:
        """
        Génère une réponse simulée adaptée au type d'agent.
        
        Args:
            agent: Agent qui répond
            query: Requête reçue
            
        Returns:
            Réponse simulée
        """
        if agent.agent_type == AgentType.RESEARCHER:
            return self._generate_researcher_response(query)
        elif agent.agent_type == AgentType.ANALYZER:
            return self._generate_analyzer_response(query)
        elif agent.agent_type == AgentType.SCRAPER:
            return self._generate_scraper_response(query)
        elif agent.agent_type == AgentType.TUTOR:
            return self._generate_tutor_response(query)
        else:
            return f"Je suis l'agent {agent.name}. Voici ma réponse à votre question sur '{query[:30]}...': [Réponse simulée]"
    
    def _generate_researcher_response(self, query: str) -> str:
        """Génère une réponse simulée pour un agent de recherche."""
        responses = [
            f"D'après mes recherches sur '{query}', j'ai trouvé plusieurs sources pertinentes. Selon les dernières publications scientifiques, ce sujet est particulièrement important dans les domaines X, Y et Z. Les experts s'accordent à dire que...",
            f"Votre question sur '{query}' est intéressante. Après avoir consulté diverses sources, je peux vous indiquer que les dernières avancées dans ce domaine suggèrent que... Plusieurs études récentes soulignent également l'importance de...",
            f"Concernant '{query}', j'ai analysé plus de 15 sources récentes. Les points clés à retenir sont: 1) ..., 2) ..., 3) ... La communauté scientifique considère généralement que...",
        ]
        return random.choice(responses)
    
    def _generate_analyzer_response(self, query: str) -> str:
        """Génère une réponse simulée pour un agent d'analyse."""
        responses = [
            f"J'ai analysé les données concernant '{query}'. Les tendances principales montrent une augmentation de 23% sur la période étudiée. Les facteurs clés semblent être A, B et C. Si cette tendance se maintient, nous pouvons anticiper...",
            f"Mon analyse des données sur '{query}' révèle plusieurs insights intéressants: 1) Corrélation forte (0.87) entre X et Y, 2) Tendance à la baisse pour Z (-12% annuel), 3) Saisonnalité marquée avec des pics en [mois]. Je recommande de...",
            f"L'analyse des données pour '{query}' indique une distribution non-normale (skewness: 1.4). Les valeurs aberrantes représentent 3.2% des données et correspondent principalement à [événements]. En excluant ces valeurs, la moyenne s'établit à...",
        ]
        return random.choice(responses)
    
    def _generate_scraper_response(self, query: str) -> str:
        """Génère une réponse simulée pour un agent de scraping."""
        responses = [
            f"J'ai effectué l'extraction des données de '{query}'. J'ai pu collecter 237 entrées avec un taux de réussite de 98.3%. Les données sont structurées selon le schéma suivant: [...]. Voulez-vous que je procède à une analyse préliminaire?",
            f"Extraction terminée pour '{query}'. 412 éléments ont été extraits, dont 89% contiennent toutes les propriétés demandées. Les données sont disponibles au format JSON. Points notables: 1) La majorité des entrées (76%) proviennent de [source], 2) ...",
            f"J'ai scrappé les informations demandées sur '{query}'. La collecte a duré 3.2 minutes et a généré 1.2MB de données structurées. Quelques statistiques: nombre total d'éléments = 156, complétude moyenne = 92%, fourchette de prix = X-Y...",
        ]
        return random.choice(responses)
    
    def _generate_tutor_response(self, query: str) -> str:
        """Génère une réponse simulée pour un agent tuteur."""
        responses = [
            f"Excellente question sur '{query}'! Pour comprendre ce concept, commençons par les fondamentaux: [...]. Un exemple concret serait [...]. Pouvez-vous essayer de résoudre ce petit exercice pour vérifier votre compréhension?",
            f"Pour expliquer '{query}', je vais procéder par étapes: 1) Définition et contexte, 2) Principes fondamentaux, 3) Applications pratiques. D'abord, [...]. Avez-vous des questions sur cette première partie?",
            f"'{query}' est un sujet fascinant qui comprend plusieurs dimensions. Si je devais le résumer simplement: [...]. Une erreur courante est de penser que [...], mais en réalité [...]. Essayons une approche interactive pour approfondir.",
        ]
        return random.choice(responses)
    
    def _generate_mock_sources(self, agent: Agent, query: str) -> List[Dict[str, Any]]:
        """
        Génère des sources simulées pour la réponse.
        
        Args:
            agent: Agent qui répond
            query: Requête reçue
            
        Returns:
            Liste de sources simulées
        """
        if agent.agent_type == AgentType.RESEARCHER:
            return [
                {"title": f"Article scientifique sur {query[:20]}", "url": "https://example.com/article1", "date": "2023-05-15"},
                {"title": "Étude comparative récente", "url": "https://example.com/article2", "date": "2023-06-22"},
                {"title": "Publication académique", "url": "https://example.com/article3", "date": "2023-04-10"}
            ]
        elif agent.agent_type == AgentType.ANALYZER:
            return [
                {"title": "Jeu de données analysé", "source": "internal_database", "records": random.randint(1000, 10000)},
                {"title": "Rapport d'analyse précédent", "id": f"report_{random.randint(1000, 9999)}", "date": "2023-07-01"}
            ]
        elif agent.agent_type == AgentType.SCRAPER:
            return [
                {"title": "Site web scrapé", "url": f"https://{query.replace(' ', '-')}.com", "pages": random.randint(5, 50)},
                {"title": "API utilisée", "endpoint": "https://api.example.com/data", "rate_limit": "100 req/min"}
            ]
        else:
            return []
    
    def _generate_mock_thinking(self, agent: Agent, query: str) -> Optional[str]:
        """
        Génère un raisonnement simulé de l'agent.
        
        Args:
            agent: Agent qui répond
            query: Requête reçue
            
        Returns:
            Raisonnement simulé ou None
        """
        if random.random() > 0.5:  # Simuler le fait que le "thinking" n'est pas toujours disponible
            return None
        
        return f"""
        Analyse de la requête: '{query}'
        1. Identification des concepts clés: [...]
        2. Recherche de sources pertinentes: [...]
        3. Évaluation de la fiabilité des informations: [...]
        4. Synthèse des informations trouvées: [...]
        5. Formulation de la réponse finale
        """ 
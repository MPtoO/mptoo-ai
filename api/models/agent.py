from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime


class AgentType(str, Enum):
    """Types d'agents disponibles."""
    RESEARCHER = "researcher"  # Pour la recherche d'informations
    ANALYZER = "analyzer"      # Pour l'analyse de données
    ASSISTANT = "assistant"    # Pour l'assistance utilisateur
    SCRAPER = "scraper"        # Pour l'extraction de données web
    TUTOR = "tutor"            # Pour l'apprentissage personnalisé
    CUSTOM = "custom"          # Pour les agents personnalisés


class AgentCapability(str, Enum):
    """Capacités des agents."""
    WEB_SEARCH = "web_search"
    DATA_ANALYSIS = "data_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    CONVERSATION = "conversation"
    CODE_GENERATION = "code_generation"
    WEB_SCRAPING = "web_scraping"
    FINE_TUNING = "fine_tuning"


class AgentStatus(str, Enum):
    """Statut d'un agent."""
    ACTIVE = "active"
    PAUSED = "paused"
    TRAINING = "training"
    ERROR = "error"
    DELETED = "deleted"


class AgentSettings(BaseModel):
    """Paramètres de configuration d'un agent."""
    max_tokens: int = Field(2000, description="Nombre maximum de tokens par réponse")
    temperature: float = Field(0.7, description="Température (créativité) de l'agent", ge=0.0, le=1.0)
    context_window: int = Field(10, description="Nombre de messages de contexte à conserver")
    model_name: str = Field("default", description="Nom du modèle à utiliser")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="Capacités activées")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Paramètres personnalisés")


class AgentTrainingData(BaseModel):
    """Données pour l'entraînement ou le fine-tuning d'un agent."""
    documents: List[str] = Field(default_factory=list, description="Liste des documents d'entraînement")
    examples: List[Dict[str, str]] = Field(default_factory=list, description="Exemples d'interactions")
    data_sources: List[str] = Field(default_factory=list, description="Sources de données externes")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres d'entraînement")


class AgentBase(BaseModel):
    """Modèle de base pour les agents."""
    name: str = Field(..., description="Nom de l'agent")
    description: str = Field(..., description="Description de l'agent")
    agent_type: AgentType
    settings: AgentSettings = Field(default_factory=AgentSettings, description="Paramètres de l'agent")


class AgentCreate(AgentBase):
    """Modèle pour la création d'un agent."""
    training_data: Optional[AgentTrainingData] = Field(None, description="Données d'entraînement optionnelles")


class Agent(AgentBase):
    """Modèle complet d'un agent."""
    id: str = Field(..., description="Identifiant unique de l'agent")
    created_at: datetime
    updated_at: datetime
    created_by: str
    status: AgentStatus = AgentStatus.ACTIVE
    usage_count: int = 0
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        orm_mode = True


class AgentUpdate(BaseModel):
    """Modèle pour la mise à jour d'un agent."""
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[AgentSettings] = None
    status: Optional[AgentStatus] = None


class AgentMessage(BaseModel):
    """Message envoyé à ou reçu d'un agent."""
    content: str = Field(..., description="Contenu du message")
    role: str = Field(..., description="Rôle du message (user/assistant)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Métadonnées du message")
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentQuery(BaseModel):
    """Requête envoyée à un agent."""
    message: str = Field(..., description="Message à envoyer à l'agent")
    context: Optional[List[AgentMessage]] = Field(None, description="Contexte de conversation optionnel")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Paramètres pour la requête")


class AgentResponse(BaseModel):
    """Réponse d'un agent."""
    message: AgentMessage
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Sources utilisées")
    thinking: Optional[str] = Field(None, description="Raisonnement de l'agent")
    execution_time: float = Field(..., description="Temps d'exécution en secondes")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Utilisation de tokens") 
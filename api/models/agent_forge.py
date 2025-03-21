from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime

from api.models.agent import AgentCapability, AgentType

class FineTuningStrategy(str, Enum):
    """Stratégies disponibles pour le fine-tuning adaptatif."""
    PROGRESSIVE_TRANSFER = "progressive_transfer"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"
    MIXED_PRECISION = "mixed_precision"
    PARAMETER_EFFICIENT = "parameter_efficient"
    DOMAIN_ADAPTIVE = "domain_adaptive"
    MULTI_TASK = "multi_task"

class ContextExtractionMethod(str, Enum):
    """Méthodes d'extraction de contexte disponibles."""
    SEMANTIC_CHUNKING = "semantic_chunking"
    HIERARCHICAL_EMBEDDING = "hierarchical_embedding"
    ENTITY_BASED = "entity_based"
    KEYWORD_BASED = "keyword_based"
    HYBRID = "hybrid"

class OrchestrationStrategy(str, Enum):
    """Stratégies d'orchestration de modèles multiples."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DECISION_TREE = "decision_tree"
    VOTING_ENSEMBLE = "voting_ensemble"
    WEIGHTED_ENSEMBLE = "weighted_ensemble"
    META_LEARNING = "meta_learning"

class PersonalityDimension(str, Enum):
    """Dimensions de personnalité configurables pour les agents."""
    FORMALITY = "formality"
    EMPATHY = "empathy"
    CONCISENESS = "conciseness"
    CREATIVITY = "creativity"
    ASSERTIVENESS = "assertiveness"
    HELPFULNESS = "helpfulness"
    EXPERTISE_LEVEL = "expertise_level"
    HUMOR = "humor"

class AgentForgeConfig(BaseModel):
    """Configuration pour l'algorithme AgentForge."""
    name: str = Field(..., description="Nom de la configuration AgentForge")
    description: str = Field(..., description="Description de la configuration")
    
    # Stratégies de fine-tuning
    fine_tuning_strategy: FineTuningStrategy = Field(
        FineTuningStrategy.PROGRESSIVE_TRANSFER,
        description="Stratégie de fine-tuning à utiliser"
    )
    fine_tuning_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Paramètres spécifiques à la stratégie de fine-tuning"
    )
    
    # Extraction contextuelle
    context_extraction_method: ContextExtractionMethod = Field(
        ContextExtractionMethod.HYBRID,
        description="Méthode d'extraction de contexte à utiliser"
    )
    context_extraction_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Paramètres pour l'extraction de contexte"
    )
    
    # Orchestration multi-modèles
    orchestration_strategy: OrchestrationStrategy = Field(
        OrchestrationStrategy.WEIGHTED_ENSEMBLE,
        description="Stratégie d'orchestration de modèles multiples"
    )
    models_to_orchestrate: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Liste des modèles à orchestrer avec leurs poids/rôles"
    )
    
    # Génération de personnalité
    personality_dimensions: Dict[PersonalityDimension, float] = Field(
        default_factory=dict,
        description="Valeurs pour chaque dimension de personnalité (0.0 à 1.0)"
    )
    
    # Options supplémentaires
    adaptive_hyperparameters: bool = Field(
        True, 
        description="Active l'optimisation automatique des hyperparamètres"
    )
    cross_validation_folds: int = Field(
        5,
        description="Nombre de folds pour la validation croisée automatique",
        ge=3, le=10
    )
    data_augmentation_enabled: bool = Field(
        True,
        description="Active la génération de données synthétiques complémentaires"
    )
    
    # Intégration
    input_data_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sources de données pour l'entraînement et l'inférence"
    )
    target_capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Capacités cibles pour l'agent généré"
    )
    target_agent_type: AgentType = Field(
        AgentType.CUSTOM,
        description="Type d'agent à générer"
    )

class AgentForgeTask(BaseModel):
    """Tâche de génération d'agent avec AgentForge."""
    id: str = Field(..., description="Identifiant unique de la tâche")
    config: AgentForgeConfig = Field(..., description="Configuration AgentForge")
    created_at: datetime
    updated_at: datetime
    created_by: str
    status: str = Field("pending", description="État de la tâche")
    progress: float = Field(0.0, description="Progression de la tâche (0.0 à 1.0)")
    agent_id: Optional[str] = Field(None, description="ID de l'agent généré (si terminé)")
    logs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Logs d'exécution de la tâche"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métriques de performance"
    )

class AgentForgeResult(BaseModel):
    """Résultat d'une tâche AgentForge."""
    task_id: str = Field(..., description="ID de la tâche AgentForge")
    agent_id: str = Field(..., description="ID de l'agent généré")
    completion_time: datetime
    training_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métriques d'entraînement"
    )
    evaluation_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métriques d'évaluation"
    )
    model_architecture: Dict[str, Any] = Field(
        default_factory=dict,
        description="Description de l'architecture du modèle"
    )
    recommended_use_cases: List[str] = Field(
        default_factory=list,
        description="Cas d'utilisation recommandés pour cet agent"
    )

class AgentForgeRequest(BaseModel):
    """Requête pour démarrer une tâche AgentForge."""
    config: AgentForgeConfig
    priority: int = Field(1, description="Priorité de la tâche (1-10)", ge=1, le=10)
    callback_url: Optional[str] = Field(
        None,
        description="URL de callback à appeler lorsque la tâche est terminée"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées supplémentaires"
    )

class AgentForgeResponse(BaseModel):
    """Réponse à une requête AgentForge."""
    task_id: str
    estimated_completion_time: datetime
    status: str 
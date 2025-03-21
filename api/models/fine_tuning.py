from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime


class DatasetFormat(str, Enum):
    """Formats de données d'entraînement disponibles."""
    JSONL = "jsonl"
    CSV = "csv"
    TEXT = "text"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """Types de modèles disponibles pour le fine-tuning."""
    LANGUAGE_MODEL = "language_model"
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"
    QA = "question_answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CUSTOM = "custom"


class DatasetSplitConfig(BaseModel):
    """Configuration pour la division du dataset."""
    train_ratio: float = Field(0.8, description="Proportion pour l'entraînement", ge=0.5, le=0.95)
    validation_ratio: float = Field(0.1, description="Proportion pour la validation", ge=0.0, le=0.3)
    test_ratio: float = Field(0.1, description="Proportion pour le test", ge=0.0, le=0.3)
    
    @validator('train_ratio', 'validation_ratio', 'test_ratio')
    def check_ratios_sum(cls, v, values):
        if 'train_ratio' in values and 'validation_ratio' in values:
            total = values['train_ratio'] + values['validation_ratio'] + v
            if abs(total - 1.0) > 0.001:  # Allow for small floating point errors
                raise ValueError("Ratios must sum to 1.0")
        return v


class DataPrepConfig(BaseModel):
    """Configuration pour la préparation des données."""
    clean_text: bool = Field(True, description="Nettoyer le texte (espaces, ponctuation)")
    lowercase: bool = Field(True, description="Convertir en minuscules")
    remove_stopwords: bool = Field(False, description="Supprimer les mots vides")
    stemming: bool = Field(False, description="Appliquer le stemming")
    lemmatization: bool = Field(False, description="Appliquer la lemmatisation")
    max_length: Optional[int] = Field(None, description="Longueur maximale des séquences")
    truncation: bool = Field(True, description="Tronquer les séquences trop longues")
    padding: bool = Field(True, description="Ajouter du padding aux séquences courtes")
    special_tokens: Dict[str, str] = Field(default_factory=dict, description="Tokens spéciaux à ajouter")
    custom_preprocessing: List[str] = Field(default_factory=list, description="Étapes de prétraitement personnalisées")


class HyperParams(BaseModel):
    """Hyperparamètres pour le fine-tuning."""
    learning_rate: float = Field(5e-5, description="Taux d'apprentissage")
    batch_size: int = Field(8, description="Taille des batchs")
    epochs: int = Field(3, description="Nombre d'époques")
    weight_decay: float = Field(0.01, description="Weight decay")
    warmup_steps: int = Field(500, description="Nombre d'étapes de warmup")
    gradient_accumulation_steps: int = Field(1, description="Nombre d'étapes d'accumulation du gradient")
    optimizer: str = Field("adamw", description="Optimiseur à utiliser")
    lr_scheduler: str = Field("linear", description="Scheduler de learning rate")
    max_grad_norm: float = Field(1.0, description="Norme maximale des gradients")
    fp16: bool = Field(True, description="Utiliser la précision mixte (FP16)")
    custom_params: Dict[str, Any] = Field(default_factory=dict, description="Paramètres personnalisés")


class EvaluationMetric(str, Enum):
    """Métriques d'évaluation disponibles."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    LOSS = "loss"
    CUSTOM = "custom"


class EvaluationConfig(BaseModel):
    """Configuration pour l'évaluation du modèle."""
    metrics: List[EvaluationMetric] = Field(..., description="Métriques d'évaluation")
    eval_steps: int = Field(100, description="Fréquence d'évaluation (étapes)")
    save_best_model: bool = Field(True, description="Sauvegarder le meilleur modèle")
    best_model_metric: Optional[EvaluationMetric] = Field(None, description="Métrique pour le meilleur modèle")
    early_stopping: bool = Field(False, description="Arrêt anticipé")
    patience: int = Field(3, description="Patience pour l'arrêt anticipé")
    threshold: float = Field(0.001, description="Seuil pour l'arrêt anticipé")
    custom_eval: Dict[str, Any] = Field(default_factory=dict, description="Évaluation personnalisée")


class DatasetSource(BaseModel):
    """Source de données pour le fine-tuning."""
    name: str = Field(..., description="Nom de la source")
    type: str = Field(..., description="Type de source (file, api, database, etc.)")
    location: str = Field(..., description="Emplacement/URL de la source")
    format: DatasetFormat = Field(..., description="Format des données")
    credentials: Optional[Dict[str, str]] = Field(None, description="Identifiants si nécessaire")
    options: Dict[str, Any] = Field(default_factory=dict, description="Options spécifiques")


class FineTuningStatus(str, Enum):
    """Statut d'une tâche de fine-tuning."""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingCheckpoint(BaseModel):
    """Point de contrôle de l'entraînement."""
    step: int = Field(..., description="Étape d'entraînement")
    timestamp: datetime = Field(default_factory=datetime.now)
    loss: float = Field(..., description="Valeur de la perte")
    learning_rate: float = Field(..., description="Taux d'apprentissage actuel")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Métriques d'évaluation")
    save_path: Optional[str] = Field(None, description="Chemin de sauvegarde du checkpoint")


class TrainingStats(BaseModel):
    """Statistiques d'entraînement."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_steps: int = 0
    total_epochs: int = 0
    training_time_seconds: int = 0
    final_loss: Optional[float] = None
    best_metrics: Dict[str, float] = Field(default_factory=dict)
    checkpoints: List[TrainingCheckpoint] = Field(default_factory=list)
    learning_curve: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Retourne la durée en secondes si l'entraînement est terminé."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class FineTuningConfig(BaseModel):
    """Configuration complète pour une tâche de fine-tuning."""
    name: str = Field(..., description="Nom de la configuration")
    description: Optional[str] = Field(None, description="Description de la configuration")
    model_type: ModelType = Field(..., description="Type de modèle")
    base_model: str = Field(..., description="Modèle de base à fine-tuner")
    dataset_sources: List[DatasetSource] = Field(..., description="Sources de données")
    dataset_split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig, description="Configuration de split")
    data_prep: DataPrepConfig = Field(default_factory=DataPrepConfig, description="Préparation des données")
    hyperparameters: HyperParams = Field(default_factory=HyperParams, description="Hyperparamètres")
    evaluation: EvaluationConfig = Field(..., description="Configuration d'évaluation")
    output_dir: str = Field(..., description="Répertoire de sortie")
    save_steps: int = Field(1000, description="Fréquence de sauvegarde (étapes)")
    tags: List[str] = Field(default_factory=list, description="Tags pour la tâche")


class FineTuningTask(BaseModel):
    """Tâche de fine-tuning."""
    id: str = Field(..., description="Identifiant unique de la tâche")
    config: FineTuningConfig
    status: FineTuningStatus = FineTuningStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(..., description="Utilisateur ayant créé la tâche")
    stats: TrainingStats = Field(default_factory=TrainingStats)
    model_path: Optional[str] = Field(None, description="Chemin vers le modèle fine-tuné")
    error_details: Optional[str] = None
    
    class Config:
        orm_mode = True


class FineTuningRequest(BaseModel):
    """Requête pour créer une tâche de fine-tuning."""
    config: FineTuningConfig
    priority: int = Field(1, description="Priorité de la tâche (1-10)")
    notify_on_completion: bool = Field(True, description="Notifier à la fin")


class FineTuningResponse(BaseModel):
    """Réponse à une requête de fine-tuning."""
    task_id: str
    message: str
    status: FineTuningStatus


class FineTuningResult(BaseModel):
    """Résultat d'une tâche de fine-tuning."""
    task: FineTuningTask
    metrics: Dict[str, float]
    model_info: Dict[str, Any]
    test_results: Optional[Dict[str, Any]] = None
    deploy_options: List[str] = ["api", "download", "hub"]


class ModelPrediction(BaseModel):
    """Prédiction d'un modèle fine-tuné."""
    model_id: str
    input: Any
    output: Any
    confidence: Optional[float] = None
    processing_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict) 
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime


class ScrapingStrategy(str, Enum):
    """Stratégies de scraping disponibles."""
    HTML_PARSING = "html_parsing"  # Analyse du DOM HTML
    API_EXTRACTION = "api_extraction"  # Extraction via API
    HEADLESS_BROWSER = "headless_browser"  # Navigation avec navigateur headless
    STRUCTURED_DATA = "structured_data"  # Extraction de données structurées (JSON-LD, etc.)
    PDF_EXTRACTION = "pdf_extraction"  # Extraction depuis PDF
    CUSTOM = "custom"  # Stratégie personnalisée


class ScrapingSelector(BaseModel):
    """Sélecteur pour l'extraction de données."""
    type: str = Field(..., description="Type de sélecteur (css, xpath, json, regex, etc.)")
    value: str = Field(..., description="Valeur du sélecteur")
    attribute: Optional[str] = Field(None, description="Attribut à extraire (optionnel)")
    name: Optional[str] = Field(None, description="Nom du champ extrait (optionnel)")


class DataTransformation(str, Enum):
    """Types de transformations disponibles pour les données extraites."""
    TO_TEXT = "to_text"
    TO_NUMBER = "to_number"
    TO_DATE = "to_date"
    TO_BOOL = "to_bool"
    STRIP = "strip"
    REPLACE = "replace"
    EXTRACT_PATTERN = "extract_pattern"
    JOIN = "join"
    SPLIT = "split"


class TransformationStep(BaseModel):
    """Étape de transformation des données extraites."""
    type: DataTransformation
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ScrapingField(BaseModel):
    """Définition d'un champ à extraire."""
    name: str = Field(..., description="Nom du champ")
    selector: ScrapingSelector
    required: bool = Field(True, description="Si le champ est requis")
    transformations: List[TransformationStep] = Field(default_factory=list)
    fallback_value: Optional[Any] = Field(None, description="Valeur par défaut si non trouvé")


class PaginationStrategy(str, Enum):
    """Stratégies de pagination."""
    LINK_NEXT = "link_next"  # Suivre les liens "suivant"
    PAGE_PARAMETER = "page_parameter"  # Paramètre de page dans l'URL
    INFINITE_SCROLL = "infinite_scroll"  # Scroll infini (chargement dynamique)
    LOAD_MORE = "load_more"  # Bouton "charger plus"
    NONE = "none"  # Pas de pagination


class PaginationConfig(BaseModel):
    """Configuration de la pagination."""
    strategy: PaginationStrategy
    selector: Optional[ScrapingSelector] = None
    max_pages: int = Field(1, description="Nombre maximum de pages à scraper")
    delay: float = Field(1.0, description="Délai entre les pages (secondes)")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ProxyConfig(BaseModel):
    """Configuration d'un proxy pour le scraping."""
    url: str
    auth_required: bool = False
    username: Optional[str] = None
    password: Optional[str] = None


class ScrapingConfig(BaseModel):
    """Configuration complète pour une tâche de scraping."""
    name: str = Field(..., description="Nom de la configuration")
    description: Optional[str] = Field(None, description="Description de la configuration")
    strategy: ScrapingStrategy = ScrapingStrategy.HTML_PARSING
    base_url: HttpUrl
    start_urls: List[str] = Field(..., description="URLs de départ")
    fields: List[ScrapingField] = Field(..., description="Champs à extraire")
    item_selector: Optional[ScrapingSelector] = Field(None, description="Sélecteur pour les items répétitifs")
    pagination: Optional[PaginationConfig] = Field(None, description="Configuration de pagination")
    headers: Dict[str, str] = Field(default_factory=dict, description="En-têtes HTTP")
    cookies: Dict[str, str] = Field(default_factory=dict, description="Cookies")
    proxy: Optional[ProxyConfig] = Field(None, description="Configuration du proxy")
    javascript_required: bool = Field(False, description="Si JavaScript est nécessaire")
    request_delay: float = Field(0.0, description="Délai entre les requêtes (secondes)")
    timeout: float = Field(30.0, description="Timeout des requêtes (secondes)")
    retry_count: int = Field(3, description="Nombre de tentatives en cas d'échec")
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class ScrapingTaskStatus(str, Enum):
    """Statut d'une tâche de scraping."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class ScrapedItem(BaseModel):
    """Item extrait par le scraper."""
    data: Dict[str, Any] = Field(..., description="Données extraites")
    url: str = Field(..., description="URL source")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ScrapingStats(BaseModel):
    """Statistiques d'une tâche de scraping."""
    pages_crawled: int = 0
    items_scraped: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    errors_count: int = 0
    retry_count: int = 0
    success_rate: float = 1.0
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Retourne la durée en secondes si la tâche est terminée."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class ScrapingTask(BaseModel):
    """Tâche de scraping."""
    id: str = Field(..., description="Identifiant unique de la tâche")
    config: ScrapingConfig
    status: ScrapingTaskStatus = ScrapingTaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: str = Field(..., description="Utilisateur ayant créé la tâche")
    stats: ScrapingStats = Field(default_factory=ScrapingStats)
    results: List[ScrapedItem] = Field(default_factory=list)
    error_details: Optional[str] = None
    
    class Config:
        orm_mode = True


class ScrapingRequest(BaseModel):
    """Requête pour créer une tâche de scraping."""
    config: ScrapingConfig
    schedule: Optional[str] = Field(None, description="Expression cron pour la planification (optionnel)")
    priority: int = Field(1, description="Priorité de la tâche (1-10)")
    tags: List[str] = Field(default_factory=list, description="Tags pour la tâche")


class ScrapingResponse(BaseModel):
    """Réponse à une requête de scraping."""
    task_id: str
    message: str
    status: ScrapingTaskStatus


class ScrapingResult(BaseModel):
    """Résultat complet d'une tâche de scraping."""
    task: ScrapingTask
    items: List[ScrapedItem]
    total_count: int
    export_formats: List[str] = ["json", "csv", "excel"]


class ReceivedData(BaseModel):
    """Option de réception des données scrappées."""
    webhook_url: Optional[HttpUrl] = None
    email_notification: Optional[str] = None
    store_in_database: bool = True
    export_format: str = "json" 
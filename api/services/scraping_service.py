import logging
import time
import uuid
import json
import os
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from api.models.scraping import (
    ScrapingConfig, ScrapingTask, ScrapingTaskStatus, ScrapingRequest,
    ScrapingResponse, ScrapedItem, ScrapingResult, ScrapingStats,
    ScrapingStrategy, ScrapingSelector, PaginationStrategy
)

logger = logging.getLogger(__name__)

class ScrapingService:
    """Service pour la gestion des tâches de scraping."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tasks_db = {}  # Simulation de base de données
        self.active_tasks = {}  # Tâches en cours d'exécution
        self.dev_mode = os.environ.get("ENVIRONMENT", "development") == "development"
        
        # En mode développement, charger des tâches prédéfinies
        if self.dev_mode:
            self._load_predefined_tasks()
    
    def _load_predefined_tasks(self):
        """Charge des tâches prédéfinies pour le développement."""
        self.logger.info("Chargement des tâches de scraping prédéfinies pour le développement")
        
        # Exemple de tâche complétée
        completed_task_id = str(uuid.uuid4())
        
        # Configuration du scraping de produits
        product_config = ScrapingConfig(
            name="E-commerce Products Scraper",
            description="Extraction de produits depuis un site e-commerce",
            strategy=ScrapingStrategy.HTML_PARSING,
            base_url="https://example-store.com",
            start_urls=["https://example-store.com/products"],
            fields=[
                {
                    "name": "title",
                    "selector": {
                        "type": "css",
                        "value": "h2.product-title"
                    },
                    "required": True
                },
                {
                    "name": "price",
                    "selector": {
                        "type": "css",
                        "value": "span.product-price",
                        "attribute": "text"
                    },
                    "transformations": [
                        {
                            "type": "strip",
                            "parameters": {}
                        },
                        {
                            "type": "to_number",
                            "parameters": {
                                "decimal_separator": ".",
                                "thousands_separator": ","
                            }
                        }
                    ]
                },
                {
                    "name": "image_url",
                    "selector": {
                        "type": "css",
                        "value": "img.product-image",
                        "attribute": "src"
                    },
                    "required": False
                }
            ],
            item_selector={
                "type": "css",
                "value": "div.product-item"
            },
            pagination={
                "strategy": PaginationStrategy.LINK_NEXT,
                "selector": {
                    "type": "css",
                    "value": "a.next-page"
                },
                "max_pages": 5
            }
        )
        
        # Création d'une tâche complétée avec des résultats simulés
        self.tasks_db[completed_task_id] = ScrapingTask(
            id=completed_task_id,
            config=product_config,
            status=ScrapingTaskStatus.COMPLETED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            stats=ScrapingStats(
                pages_crawled=5,
                items_scraped=87,
                start_time=datetime.now(),
                end_time=datetime.now(),
                errors_count=2,
                retry_count=3,
                success_rate=0.95
            ),
            results=[
                ScrapedItem(
                    data={
                        "title": "Product Example 1",
                        "price": 29.99,
                        "image_url": "https://example-store.com/images/product1.jpg"
                    },
                    url="https://example-store.com/products/1",
                    metadata={"index": 1, "category": "electronics"}
                ),
                ScrapedItem(
                    data={
                        "title": "Product Example 2",
                        "price": 49.99,
                        "image_url": "https://example-store.com/images/product2.jpg"
                    },
                    url="https://example-store.com/products/2",
                    metadata={"index": 2, "category": "electronics"}
                )
            ]
        )
        
        # Exemple de tâche en attente
        pending_task_id = str(uuid.uuid4())
        
        # Configuration du scraping d'articles de blog
        blog_config = ScrapingConfig(
            name="Blog Articles Scraper",
            description="Extraction d'articles de blog avec leur contenu",
            strategy=ScrapingStrategy.HTML_PARSING,
            base_url="https://example-blog.com",
            start_urls=["https://example-blog.com/blog"],
            fields=[
                {
                    "name": "title",
                    "selector": {
                        "type": "css",
                        "value": "h1.article-title"
                    },
                    "transformations": [
                        {"type": "strip"}
                    ]
                },
                {
                    "name": "author",
                    "selector": {
                        "type": "css",
                        "value": "span.author-name"
                    },
                    "required": False
                },
                {
                    "name": "date",
                    "selector": {
                        "type": "css",
                        "value": "time.published-date",
                        "attribute": "datetime"
                    },
                    "transformations": [
                        {"type": "to_date"}
                    ],
                    "required": False
                },
                {
                    "name": "content",
                    "selector": {
                        "type": "css",
                        "value": "div.article-content"
                    }
                }
            ],
            item_selector={
                "type": "css",
                "value": "article.blog-post"
            },
            pagination={
                "strategy": PaginationStrategy.PAGE_PARAMETER,
                "parameters": {
                    "param_name": "page",
                    "start_value": 1,
                    "increment": 1
                },
                "max_pages": 10
            }
        )
        
        self.tasks_db[pending_task_id] = ScrapingTask(
            id=pending_task_id,
            config=blog_config,
            status=ScrapingTaskStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            stats=ScrapingStats()
        )
    
    async def create_task(self, request: ScrapingRequest, user_id: str) -> ScrapingResponse:
        """
        Crée une nouvelle tâche de scraping.
        
        Args:
            request: Requête de configuration du scraping
            user_id: ID de l'utilisateur créant la tâche
            
        Returns:
            Réponse avec l'ID de la tâche créée
        """
        task_id = str(uuid.uuid4())
        
        # Créer la tâche
        task = ScrapingTask(
            id=task_id,
            config=request.config,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by=user_id
        )
        
        self.tasks_db[task_id] = task
        
        self.logger.info(f"Tâche de scraping créée: {task.config.name} (ID: {task_id})")
        
        # En mode développement, on peut exécuter immédiatement
        if self.dev_mode and random.random() > 0.5:
            asyncio.create_task(self._execute_task(task_id))
            message = "Tâche créée et démarrée (mode développement)"
        else:
            message = "Tâche créée avec succès et mise en file d'attente"
        
        return ScrapingResponse(
            task_id=task_id,
            message=message,
            status=task.status
        )
    
    async def get_task(self, task_id: str) -> Optional[ScrapingTask]:
        """
        Récupère une tâche par son ID.
        
        Args:
            task_id: ID de la tâche
            
        Returns:
            La tâche ou None si non trouvée
        """
        return self.tasks_db.get(task_id)
    
    async def get_task_result(self, task_id: str) -> Optional[ScrapingResult]:
        """
        Récupère le résultat complet d'une tâche.
        
        Args:
            task_id: ID de la tâche
            
        Returns:
            Le résultat ou None si non trouvé ou tâche non terminée
        """
        task = await self.get_task(task_id)
        if not task:
            return None
        
        if task.status != ScrapingTaskStatus.COMPLETED:
            return None
        
        return ScrapingResult(
            task=task,
            items=task.results,
            total_count=len(task.results)
        )
    
    async def get_tasks(self, user_id: Optional[str] = None, status: Optional[ScrapingTaskStatus] = None) -> List[ScrapingTask]:
        """
        Récupère toutes les tâches, avec filtrage optionnel.
        
        Args:
            user_id: ID de l'utilisateur pour filtrer ses tâches
            status: Statut des tâches à récupérer
            
        Returns:
            Liste des tâches correspondant aux critères
        """
        tasks = list(self.tasks_db.values())
        
        # Filtrer par utilisateur si spécifié
        if user_id:
            tasks = [t for t in tasks if t.created_by == user_id]
        
        # Filtrer par statut si spécifié
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        return tasks
    
    async def start_task(self, task_id: str) -> bool:
        """
        Démarre l'exécution d'une tâche en attente.
        
        Args:
            task_id: ID de la tâche à démarrer
            
        Returns:
            True si la tâche a été démarrée, False sinon
        """
        task = await self.get_task(task_id)
        if not task or task.status != ScrapingTaskStatus.PENDING:
            return False
        
        # Lancer l'exécution asynchrone
        asyncio.create_task(self._execute_task(task_id))
        
        return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Annule une tâche en cours ou en attente.
        
        Args:
            task_id: ID de la tâche à annuler
            
        Returns:
            True si la tâche a été annulée, False sinon
        """
        task = await self.get_task(task_id)
        if not task or task.status not in [ScrapingTaskStatus.PENDING, ScrapingTaskStatus.RUNNING, ScrapingTaskStatus.PAUSED]:
            return False
        
        # Si la tâche est en cours d'exécution, il faudrait l'arrêter
        if task.status == ScrapingTaskStatus.RUNNING and task_id in self.active_tasks:
            # TODO: implémenter l'arrêt réel de la tâche
            pass
        
        task.status = ScrapingTaskStatus.CANCELLED
        task.updated_at = datetime.now()
        
        return True
    
    async def delete_task(self, task_id: str) -> bool:
        """
        Supprime une tâche.
        
        Args:
            task_id: ID de la tâche à supprimer
            
        Returns:
            True si la tâche a été supprimée, False sinon
        """
        if task_id not in self.tasks_db:
            return False
        
        # Ne pas supprimer une tâche en cours
        if self.tasks_db[task_id].status == ScrapingTaskStatus.RUNNING:
            return False
        
        del self.tasks_db[task_id]
        
        return True
    
    async def _execute_task(self, task_id: str):
        """
        Exécute une tâche de scraping.
        
        Args:
            task_id: ID de la tâche à exécuter
        """
        task = self.tasks_db[task_id]
        
        # Marquer la tâche comme en cours d'exécution
        task.status = ScrapingTaskStatus.RUNNING
        task.updated_at = datetime.now()
        task.stats.start_time = datetime.now()
        
        self.active_tasks[task_id] = True
        
        try:
            # Récupérer la configuration
            config = task.config
            
            self.logger.info(f"Démarrage de la tâche de scraping {task_id}: {config.name}")
            
            # Simuler le délai d'exécution pour le développement
            if self.dev_mode:
                # Simuler un temps d'exécution plus court en développement
                execution_time = random.uniform(1.0, 5.0)
                time.sleep(execution_time)
                
                # Générer des résultats simulés
                results = await self._generate_mock_results(config)
                task.results = results
                
                # Mettre à jour les statistiques
                task.stats.pages_crawled = random.randint(1, config.pagination.max_pages if config.pagination else 1)
                task.stats.items_scraped = len(results)
                task.stats.errors_count = random.randint(0, 3)
                task.stats.retry_count = random.randint(0, 5)
                task.stats.success_rate = max(0.0, min(1.0, 1.0 - (task.stats.errors_count / max(1, task.stats.items_scraped + task.stats.errors_count))))
            else:
                # Implémentation réelle du scraping
                results, stats = await self._perform_scraping(config)
                task.results = results
                task.stats = stats
            
            # Marquer la tâche comme terminée
            task.status = ScrapingTaskStatus.COMPLETED
            
        except Exception as e:
            # En cas d'erreur, marquer la tâche comme échouée
            task.status = ScrapingTaskStatus.FAILED
            task.error_details = str(e)
            self.logger.error(f"Erreur lors de l'exécution de la tâche {task_id}: {str(e)}")
        
        finally:
            # Finaliser la tâche
            task.updated_at = datetime.now()
            task.stats.end_time = datetime.now()
            
            # Supprimer de la liste des tâches actives
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
    
    async def _perform_scraping(self, config: ScrapingConfig) -> Tuple[List[ScrapedItem], ScrapingStats]:
        """
        Exécute le scraping selon la configuration fournie.
        
        Args:
            config: Configuration du scraping
            
        Returns:
            Tuple contenant les résultats et les statistiques
        """
        # Ici, nous implémenterions le scraping réel selon la stratégie configurée
        # Pour cette version, nous retournons des données simulées
        return await self._generate_mock_results(config), ScrapingStats()
    
    async def _generate_mock_results(self, config: ScrapingConfig) -> List[ScrapedItem]:
        """
        Génère des résultats simulés pour le développement.
        
        Args:
            config: Configuration du scraping
            
        Returns:
            Liste d'items scraped simulés
        """
        results = []
        
        # Nombre d'items à générer
        num_items = random.randint(5, 20)
        
        for i in range(num_items):
            # Générer des données simulées pour chaque champ
            data = {}
            for field in config.fields:
                # Simuler une valeur selon le nom du champ
                if "title" in field.name or "name" in field.name:
                    data[field.name] = f"{config.name} Item {i+1}"
                elif "price" in field.name:
                    data[field.name] = round(random.uniform(10.0, 200.0), 2)
                elif "date" in field.name:
                    data[field.name] = (datetime.now().replace(
                        day=random.randint(1, 28),
                        month=random.randint(1, 12)
                    )).isoformat()
                elif "image" in field.name or "url" in field.name:
                    data[field.name] = f"https://example.com/images/item{i+1}.jpg"
                elif "description" in field.name or "content" in field.name:
                    data[field.name] = f"This is a sample description for item {i+1}. It contains some text that would be extracted from the target website."
                elif "author" in field.name:
                    data[field.name] = f"Author {random.randint(1, 10)}"
                else:
                    data[field.name] = f"Sample value for {field.name} {i+1}"
            
            # URL source simulée
            page_num = i // 5 + 1  # 5 items par page
            url = f"{config.base_url}/page/{page_num}/item/{i+1}"
            
            # Créer l'item
            item = ScrapedItem(
                data=data,
                url=url,
                timestamp=datetime.now(),
                metadata={"page": page_num, "index": i, "simulated": True}
            )
            
            results.append(item)
        
        return results
    
    async def _enrich_scraped_data(self, items: List[ScrapedItem], config: ScrapingConfig) -> List[ScrapedItem]:
        """
        Enrichit les données scrapées avec des informations supplémentaires via l'IA.
        Utilise les modèles d'IA pour extraire des entités, analyser le sentiment, classer le contenu, etc.
        """
        if not items:
            return items
            
        try:
            from api.services.ollama_service import OllamaService
            from api.services.agent_service import AgentService
            
            # Initialiser les services
            ollama_service = OllamaService()
            agent_service = AgentService()
            
            # Configuration des enrichissements
            enrichments = config.post_processing.get("enrichments", [])
            
            if not enrichments:
                return items
                
            self.logger.info(f"Enrichissement des données scrapées avec {len(enrichments)} transformations")
            
            for item in items:
                # Pour chaque élément, appliquer les enrichissements configurés
                for enrichment in enrichments:
                    enrichment_type = enrichment.get("type")
                    
                    if enrichment_type == "entity_extraction":
                        # Extraire les entités du texte
                        item.data = await self._extract_entities(item.data, enrichment, ollama_service)
                        
                    elif enrichment_type == "sentiment_analysis":
                        # Analyser le sentiment du texte
                        item.data = await self._analyze_sentiment(item.data, enrichment, ollama_service)
                        
                    elif enrichment_type == "text_classification":
                        # Classifier le texte
                        item.data = await self._classify_text(item.data, enrichment, ollama_service)
                        
                    elif enrichment_type == "summarization":
                        # Résumer le texte
                        item.data = await self._summarize_text(item.data, enrichment, ollama_service)
                        
                    elif enrichment_type == "keyword_extraction":
                        # Extraire les mots-clés
                        item.data = await self._extract_keywords(item.data, enrichment, ollama_service)
                        
                    elif enrichment_type == "custom_prompt":
                        # Appliquer une instruction personnalisée
                        item.data = await self._apply_custom_prompt(item.data, enrichment, ollama_service)
            
            return items
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enrichissement des données: {str(e)}")
            # En cas d'erreur, retourner les données non enrichies
            return items
    
    async def _extract_entities(self, data: Dict[str, Any], config: Dict[str, Any], ollama_service: Any) -> Dict[str, Any]:
        """Extrait les entités (personnes, lieux, organisations, etc.) du texte."""
        try:
            # Identifier les champs à traiter
            fields = config.get("fields", [])
            
            if not fields:
                return data
                
            # Obtenir le modèle à utiliser
            model = config.get("model", "llama2")
            
            for field in fields:
                if field in data and isinstance(data[field], str):
                    text = data[field]
                    
                    # Créer le prompt pour l'extraction d'entités
                    prompt = f"""
                    Extrait les entités nommées du texte suivant et retourne-les au format JSON:
                    
                    Texte:
                    {text}
                    
                    Format de sortie attendu:
                    {{
                        "personnes": ["nom1", "nom2", ...],
                        "organisations": ["org1", "org2", ...],
                        "lieux": ["lieu1", "lieu2", ...],
                        "dates": ["date1", "date2", ...],
                        "autres": ["autre1", "autre2", ...]
                    }}
                    
                    Réponds uniquement avec le JSON, sans commentaire.
                    """
                    
                    # Appeler le modèle
                    result = ollama_service.generate_sync(model, prompt, {"temperature": 0.1})
                    
                    if result and "response" in result:
                        # Extraire la partie JSON de la réponse
                        response = result["response"]
                        try:
                            import json
                            import re
                            
                            # Chercher le JSON dans la réponse
                            json_match = re.search(r'({[\s\S]*})', response)
                            if json_match:
                                extracted_json = json_match.group(1)
                                entities = json.loads(extracted_json)
                                
                                # Ajouter les entités extraites aux données
                                field_entities = f"{field}_entities"
                                data[field_entities] = entities
                        except Exception as json_error:
                            self.logger.error(f"Erreur lors du parsing JSON des entités: {str(json_error)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction d'entités: {str(e)}")
            return data
    
    async def _analyze_sentiment(self, data: Dict[str, Any], config: Dict[str, Any], ollama_service: Any) -> Dict[str, Any]:
        """Analyse le sentiment du texte (positif, négatif, neutre)."""
        try:
            # Identifier les champs à traiter
            fields = config.get("fields", [])
            
            if not fields:
                return data
                
            # Obtenir le modèle à utiliser
            model = config.get("model", "llama2")
            
            for field in fields:
                if field in data and isinstance(data[field], str):
                    text = data[field]
                    
                    # Créer le prompt pour l'analyse de sentiment
                    prompt = f"""
                    Analyse le sentiment du texte suivant et réponds uniquement avec un JSON:
                    
                    Texte:
                    {text}
                    
                    Format de sortie attendu:
                    {{
                        "sentiment": "positif|négatif|neutre",
                        "score": 0.X (nombre entre 0 et 1 indiquant la confiance),
                        "raison": "explication brève du classement"
                    }}
                    
                    Réponds uniquement avec le JSON, sans commentaire.
                    """
                    
                    # Appeler le modèle
                    result = ollama_service.generate_sync(model, prompt, {"temperature": 0.1})
                    
                    if result and "response" in result:
                        # Extraire la partie JSON de la réponse
                        response = result["response"]
                        try:
                            import json
                            import re
                            
                            # Chercher le JSON dans la réponse
                            json_match = re.search(r'({[\s\S]*})', response)
                            if json_match:
                                extracted_json = json_match.group(1)
                                sentiment = json.loads(extracted_json)
                                
                                # Ajouter le sentiment aux données
                                field_sentiment = f"{field}_sentiment"
                                data[field_sentiment] = sentiment
                        except Exception as json_error:
                            self.logger.error(f"Erreur lors du parsing JSON du sentiment: {str(json_error)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de sentiment: {str(e)}")
            return data
    
    async def _classify_text(self, data: Dict[str, Any], config: Dict[str, Any], ollama_service: Any) -> Dict[str, Any]:
        """Classifie le texte selon des catégories prédéfinies."""
        try:
            # Identifier les champs à traiter
            fields = config.get("fields", [])
            categories = config.get("categories", ["business", "technology", "politics", "science", "health"])
            
            if not fields:
                return data
                
            # Obtenir le modèle à utiliser
            model = config.get("model", "llama2")
            
            for field in fields:
                if field in data and isinstance(data[field], str):
                    text = data[field]
                    
                    # Construire les catégories
                    categories_str = ", ".join(categories)
                    
                    # Créer le prompt pour la classification
                    prompt = f"""
                    Classifie le texte suivant dans une des catégories suivantes: {categories_str}
                    
                    Texte:
                    {text}
                    
                    Format de sortie attendu:
                    {{
                        "catégorie": "nom_de_la_catégorie",
                        "confiance": 0.X,
                        "explication": "brève explication du choix"
                    }}
                    
                    Réponds uniquement avec le JSON, sans commentaire.
                    """
                    
                    # Appeler le modèle
                    result = ollama_service.generate_sync(model, prompt, {"temperature": 0.1})
                    
                    if result and "response" in result:
                        # Extraire la partie JSON de la réponse
                        response = result["response"]
                        try:
                            import json
                            import re
                            
                            # Chercher le JSON dans la réponse
                            json_match = re.search(r'({[\s\S]*})', response)
                            if json_match:
                                extracted_json = json_match.group(1)
                                classification = json.loads(extracted_json)
                                
                                # Ajouter la classification aux données
                                field_category = f"{field}_category"
                                data[field_category] = classification
                        except Exception as json_error:
                            self.logger.error(f"Erreur lors du parsing JSON de la classification: {str(json_error)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors de la classification: {str(e)}")
            return data
            
    async def _summarize_text(self, data: Dict[str, Any], config: Dict[str, Any], ollama_service: Any) -> Dict[str, Any]:
        """Résume le texte en conservant les informations essentielles."""
        try:
            # Identifier les champs à traiter
            fields = config.get("fields", [])
            max_words = config.get("max_words", 100)
            
            if not fields:
                return data
                
            # Obtenir le modèle à utiliser
            model = config.get("model", "llama2")
            
            for field in fields:
                if field in data and isinstance(data[field], str):
                    text = data[field]
                    
                    # Calculer le nombre de mots dans le texte d'origine
                    words_count = len(text.split())
                    
                    # Ne résumer que si le texte est vraiment plus long que le résumé souhaité
                    if words_count > max_words + 20:
                        # Créer le prompt pour la génération de résumé
                        prompt = f"""
                        Résume le texte suivant en {max_words} mots maximum:
                        
                        {text}
                        
                        Format de sortie attendu:
                        {{
                            "résumé": "texte résumé ici",
                            "mots_clés": ["mot1", "mot2", "mot3", ...]
                        }}
                        
                        Réponds uniquement avec le JSON, sans commentaire.
                        """
                        
                        # Appeler le modèle
                        result = ollama_service.generate_sync(model, prompt, {"temperature": 0.3})
                        
                        if result and "response" in result:
                            # Extraire la partie JSON de la réponse
                            response = result["response"]
                            try:
                                import json
                                import re
                                
                                # Chercher le JSON dans la réponse
                                json_match = re.search(r'({[\s\S]*})', response)
                                if json_match:
                                    extracted_json = json_match.group(1)
                                    summary = json.loads(extracted_json)
                                    
                                    # Ajouter le résumé aux données
                                    field_summary = f"{field}_summary"
                                    data[field_summary] = summary
                            except Exception as json_error:
                                self.logger.error(f"Erreur lors du parsing JSON du résumé: {str(json_error)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du résumé: {str(e)}")
            return data
    
    async def _extract_keywords(self, data: Dict[str, Any], config: Dict[str, Any], ollama_service: Any) -> Dict[str, Any]:
        """Extrait les mots-clés ou termes importants du texte."""
        try:
            # Identifier les champs à traiter
            fields = config.get("fields", [])
            max_keywords = config.get("max_keywords", 10)
            
            if not fields:
                return data
                
            # Obtenir le modèle à utiliser
            model = config.get("model", "llama2")
            
            for field in fields:
                if field in data and isinstance(data[field], str):
                    text = data[field]
                    
                    # Créer le prompt pour l'extraction de mots-clés
                    prompt = f"""
                    Extrait les {max_keywords} mots-clés les plus importants du texte suivant:
                    
                    Texte:
                    {text}
                    
                    Format de sortie attendu:
                    {{
                        "mots_clés": [
                            {{"mot": "mot1", "importance": 0.X}},
                            {{"mot": "mot2", "importance": 0.X}},
                            ...
                        ]
                    }}
                    
                    Réponds uniquement avec le JSON, sans commentaire.
                    """
                    
                    # Appeler le modèle
                    result = ollama_service.generate_sync(model, prompt, {"temperature": 0.2})
                    
                    if result and "response" in result:
                        # Extraire la partie JSON de la réponse
                        response = result["response"]
                        try:
                            import json
                            import re
                            
                            # Chercher le JSON dans la réponse
                            json_match = re.search(r'({[\s\S]*})', response)
                            if json_match:
                                extracted_json = json_match.group(1)
                                keywords = json.loads(extracted_json)
                                
                                # Ajouter les mots-clés aux données
                                field_keywords = f"{field}_keywords"
                                data[field_keywords] = keywords
                        except Exception as json_error:
                            self.logger.error(f"Erreur lors du parsing JSON des mots-clés: {str(json_error)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors de l'extraction des mots-clés: {str(e)}")
            return data
    
    async def _apply_custom_prompt(self, data: Dict[str, Any], config: Dict[str, Any], ollama_service: Any) -> Dict[str, Any]:
        """Applique une instruction personnalisée via un prompt défini par l'utilisateur."""
        try:
            # Obtenir les paramètres
            fields = config.get("fields", [])
            prompt_template = config.get("prompt", "")
            output_field = config.get("output_field", "custom_output")
            
            if not fields or not prompt_template:
                return data
                
            # Obtenir le modèle à utiliser
            model = config.get("model", "llama2")
            
            # Variables disponibles dans le template
            variables = {}
            
            # Ajouter toutes les données comme variables
            for key, value in data.items():
                if isinstance(value, (str, int, float, bool)):
                    variables[key] = value
            
            # Formatter le prompt avec les variables disponibles
            try:
                # Utiliser un format plus avancé pour le template
                from string import Template
                template = Template(prompt_template)
                prompt = template.safe_substitute(variables)
                
                # Appeler le modèle
                result = ollama_service.generate_sync(model, prompt, {"temperature": config.get("temperature", 0.5)})
                
                if result and "response" in result:
                    # Ajouter le résultat au champ de sortie
                    data[output_field] = result["response"]
                    
                    # Si format JSON demandé, tenter de parser
                    if config.get("json_output", False):
                        try:
                            import json
                            import re
                            
                            # Chercher le JSON dans la réponse
                            json_match = re.search(r'({[\s\S]*})', result["response"])
                            if json_match:
                                extracted_json = json_match.group(1)
                                parsed_json = json.loads(extracted_json)
                                data[output_field] = parsed_json
                        except:
                            # Si échec du parsing, garder la réponse texte
                            pass
                    
            except Exception as template_error:
                self.logger.error(f"Erreur lors de l'application du template personnalisé: {str(template_error)}")
            
            return data
        except Exception as e:
            self.logger.error(f"Erreur lors de l'application du prompt personnalisé: {str(e)}")
            return data 
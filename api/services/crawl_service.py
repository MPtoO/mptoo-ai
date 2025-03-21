import logging
import asyncio
import uuid
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

from api.models.crawl import CrawlRequest, CrawlResult, CrawlPageResult, CrawlStatus

logger = logging.getLogger(__name__)

class CrawlService:
    """Service pour le crawling de sites web."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_crawls = {}  # Stockage des crawls actifs
        self.crawl_results = {}  # Stockage des résultats de crawl
    
    async def add_crawl_target(self, request: CrawlRequest) -> CrawlResult:
        """
        Ajoute une cible de crawl à la file d'attente.
        
        Args:
            request: CrawlRequest contenant l'URL à crawler
            
        Returns:
            CrawlResult avec le statut initial
        """
        target_id = str(uuid.uuid4())
        self.logger.info(f"Adding crawl target {request.url} with ID {target_id}")
        
        # Créer un résultat initial
        result = CrawlResult(
            target_id=target_id,
            url=str(request.url),
            status=CrawlStatus.PENDING,
            message=f"Crawl pour {request.url} ajouté à la file d'attente"
        )
        
        # Stocker le résultat
        self.crawl_results[target_id] = result
        
        # Lancer le crawl en arrière-plan (asynchrone)
        asyncio.create_task(self._simulate_crawl(target_id, request))
        
        return result
    
    async def get_crawl_status(self, target_id: str) -> Optional[CrawlResult]:
        """
        Récupère le statut d'un crawl.
        
        Args:
            target_id: L'ID du crawl à vérifier
            
        Returns:
            CrawlResult avec le statut actuel ou None si non trouvé
        """
        return self.crawl_results.get(target_id)
    
    async def _simulate_crawl(self, target_id: str, request: CrawlRequest):
        """
        Simule un crawl (à remplacer par un vrai crawler).
        
        Args:
            target_id: L'ID du crawl
            request: La demande de crawl d'origine
        """
        # Marquer comme en cours
        self.crawl_results[target_id].status = CrawlStatus.RUNNING
        self.crawl_results[target_id].start_time = datetime.now().isoformat()
        
        # Simuler un délai de crawl
        max_pages = min(request.max_pages or 100, 100)  # Limiter à 100 pages max
        
        # Nombre de pages à crawler pour la simulation
        pages_to_crawl = min(max_pages, 20)  # Pour la démo, on limite à 20 pages max
        
        # Initialiser les compteurs
        self.crawl_results[target_id].pages_crawled = 0
        self.crawl_results[target_id].pages_success = 0
        self.crawl_results[target_id].pages_failed = 0
        
        # Simuler le crawl page par page
        page_results = []
        base_url = str(request.url)
        
        for i in range(pages_to_crawl):
            # Simuler le temps de traitement d'une page
            await asyncio.sleep(0.2)  # Réduit pour la démo
            
            # Simuler un succès ou un échec
            success = random.random() > 0.1  # 10% de chances d'échec
            
            # Mettre à jour les compteurs
            self.crawl_results[target_id].pages_crawled += 1
            
            if success:
                self.crawl_results[target_id].pages_success += 1
                
                # Créer un résultat de page
                page_url = f"{base_url}/{i}" if i > 0 else base_url
                page_result = CrawlPageResult(
                    url=page_url,
                    title=f"Page {i} - {base_url}",
                    content_snippet=f"Extrait du contenu de la page {i}...",
                    status_code=200,
                    crawl_time=datetime.now().isoformat(),
                    links_found=random.randint(5, 20),
                    size=random.randint(10000, 100000)
                )
                page_results.append(page_result)
            else:
                self.crawl_results[target_id].pages_failed += 1
        
        # Mettre à jour le résultat final
        self.crawl_results[target_id].status = CrawlStatus.COMPLETED
        self.crawl_results[target_id].end_time = datetime.now().isoformat()
        self.crawl_results[target_id].message = f"Crawl terminé : {self.crawl_results[target_id].pages_success} pages traitées avec succès, {self.crawl_results[target_id].pages_failed} échecs"
        self.crawl_results[target_id].results = page_results
        
        self.logger.info(f"Crawl {target_id} completed with {len(page_results)} pages")

# Import standard à l'extérieur pour éviter les problèmes circulaires
import random 
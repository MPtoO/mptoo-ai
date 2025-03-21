import os
import logging
import asyncio
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import re

import aiohttp
from bs4 import BeautifulSoup
from selectolax.parser import HTMLParser
from urllib.parse import urljoin, urlparse
import pandas as pd

from api.models.scraping import (
    ScrapingConfig, ScrapingTask, ScrapingTaskStatus, 
    ScrapingRequest, ScrapingSelector, PaginationStrategy,
    ScrapedItem
)

logger = logging.getLogger(__name__)

class AdvancedScraper:
    """
    Scraper avancé pour l'extraction de données web structurées.
    Supporte le scraping statique, dynamique (JavaScript) et la gestion de pagination.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.proxy_settings = self._load_proxy_settings()
        self.user_agents = self._load_user_agents()
        
    def _load_proxy_settings(self) -> Dict[str, str]:
        """Charge les paramètres de proxy depuis la configuration."""
        # Implémenter le chargement depuis un fichier de configuration
        return {
            "http": os.environ.get("HTTP_PROXY", None),
            "https": os.environ.get("HTTPS_PROXY", None)
        }
    
    def _load_user_agents(self) -> List[str]:
        """Charge une liste d'user-agents pour la rotation."""
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36"
        ]
    
    def _get_random_user_agent(self) -> str:
        """Retourne un user-agent aléatoire de la liste."""
        import random
        return random.choice(self.user_agents)
    
    async def scrape_url(self, 
                         url: str, 
                         config: ScrapingConfig, 
                         use_js: bool = False) -> Tuple[List[ScrapedItem], Dict[str, Any]]:
        """
        Scrape une URL spécifique selon la configuration fournie.
        
        Args:
            url: L'URL à scraper
            config: Configuration de scraping
            use_js: Si True, utilise le rendering JavaScript via Playwright
            
        Returns:
            Tuple contenant les éléments scrapés et les statistiques
        """
        self.logger.info(f"Démarrage du scraping de l'URL: {url}")
        start_time = time.time()
        
        try:
            if use_js:
                html_content = await self._fetch_with_js(url)
            else:
                html_content = await self._fetch_html(url)
                
            if not html_content:
                return [], {"error": "Impossible de récupérer le contenu HTML"}
            
            items = await self._extract_items(html_content, config)
            
            # Calculer les statistiques
            stats = {
                "url": url,
                "items_count": len(items),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Scraping terminé pour {url}: {len(items)} éléments extraits")
            return items, stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors du scraping de {url}: {str(e)}")
            return [], {"error": str(e), "url": url}
    
    async def execute_scraping_task(self, task: ScrapingTask) -> List[ScrapedItem]:
        """
        Exécute une tâche de scraping complète, y compris la pagination si configurée.
        
        Args:
            task: Tâche de scraping à exécuter
            
        Returns:
            Liste des éléments scrapés
        """
        config = task.config
        results = []
        
        # URLs de départ
        start_urls = config.start_urls if config.start_urls else [config.base_url]
        visited_urls = set()
        all_stats = []
        
        for start_url in start_urls:
            # Vérifier si l'URL est valide
            if not self._is_valid_url(start_url):
                self.logger.warning(f"URL invalide ignorée: {start_url}")
                continue
                
            current_url = start_url
            page_num = 1
            use_js = config.strategy == "JAVASCRIPT_RENDERING"
            
            while current_url and page_num <= (config.pagination.max_pages if config.pagination else 1):
                if current_url in visited_urls:
                    self.logger.warning(f"URL déjà visitée, évitement de boucle: {current_url}")
                    break
                    
                visited_urls.add(current_url)
                self.logger.info(f"Scraping de la page {page_num}: {current_url}")
                
                # Ajouter un délai pour éviter la surcharge du serveur
                await asyncio.sleep(config.request_delay if hasattr(config, 'request_delay') else 1)
                
                # Scraper l'URL courante
                items, stats = await self.scrape_url(current_url, config, use_js)
                results.extend(items)
                all_stats.append(stats)
                
                # Si la pagination est configurée, trouver l'URL suivante
                if config.pagination and page_num < config.pagination.max_pages:
                    current_url = await self._get_next_page_url(current_url, config.pagination, stats.get("html_content", ""))
                    if not current_url:
                        self.logger.info("Fin de la pagination: pas d'URL suivante trouvée")
                        break
                else:
                    # Pas de pagination ou limite atteinte
                    break
                    
                page_num += 1
        
        # Enrichir les données si demandé
        if config.post_processing and config.post_processing.get("enrich", False):
            results = await self._enrich_scraped_data(results, config)
            
        # Transformer en format de sortie si spécifié
        if config.output_format:
            results = self._format_output(results, config.output_format)
            
        self.logger.info(f"Tâche de scraping terminée: {len(results)} éléments extraits au total")
        return results
            
    async def _fetch_html(self, url: str) -> Optional[str]:
        """Récupère le contenu HTML d'une URL via HTTP."""
        headers = {
            "User-Agent": self._get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, proxy=self.proxy_settings.get("http")) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        self.logger.error(f"Échec de la requête HTTP: {response.status} pour {url}")
                        return None
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de {url}: {str(e)}")
            return None
            
    async def _fetch_with_js(self, url: str) -> Optional[str]:
        """Récupère le contenu HTML d'une URL avec rendu JavaScript via Playwright."""
        try:
            # Importer Playwright uniquement si nécessaire
            import playwright.async_api as pw
            
            self.logger.info(f"Utilisation de Playwright pour {url}")
            async with pw.async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(
                    user_agent=self._get_random_user_agent()
                )
                
                await page.goto(url, wait_until="networkidle", timeout=30000)
                
                # Attendre que le contenu soit chargé
                await asyncio.sleep(2)
                
                # Récupérer le HTML après rendu JS
                content = await page.content()
                
                await browser.close()
                return content
                
        except ImportError:
            self.logger.error("Playwright n'est pas installé. Installation recommandée: pip install playwright")
            self.logger.info("Retour à la méthode standard sans JavaScript...")
            return await self._fetch_html(url)
        except Exception as e:
            self.logger.error(f"Erreur lors du rendu JavaScript pour {url}: {str(e)}")
            return None
    
    async def _extract_items(self, html_content: str, config: ScrapingConfig) -> List[ScrapedItem]:
        """Extrait les éléments selon la configuration de scraping."""
        results = []
        
        # Utiliser le parseur HTML approprié
        if config.parser == "selectolax":
            parser = HTMLParser(html_content)
            items = parser.css(config.item_selector.value)
            
            for item in items:
                extracted_item = {}
                
                # Extraire chaque champ configuré
                for field in config.fields:
                    selector = field["selector"]
                    value = None
                    
                    if selector.type == "css":
                        element = item.css_first(selector.value)
                        if element:
                            value = element.text() if selector.attribute == "text" else element.attributes.get(selector.attribute, "")
                    elif selector.type == "xpath":
                        # Pour XPath, on revient à BeautifulSoup
                        bs_item = BeautifulSoup(item.html, "html.parser")
                        import lxml.etree as et
                        dom = et.HTML(str(bs_item))
                        elements = dom.xpath(selector.value)
                        if elements:
                            value = elements[0].text if selector.attribute == "text" else elements[0].get(selector.attribute, "")
                    
                    # Appliquer les transformations
                    if value and field.get("transformations"):
                        value = self._apply_transformations(value, field["transformations"])
                        
                    extracted_item[field["name"]] = value
                
                # Vérifier les champs requis
                if all(extracted_item.get(field["name"]) for field in config.fields if field.get("required", False)):
                    results.append(ScrapedItem(
                        source_url=config.base_url,
                        extracted_at=datetime.now().isoformat(),
                        data=extracted_item
                    ))
        
        else:  # Par défaut: BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            items = soup.select(config.item_selector.value)
            
            for item in items:
                extracted_item = {}
                
                # Extraire chaque champ configuré
                for field in config.fields:
                    selector = field["selector"]
                    value = None
                    
                    if selector.type == "css":
                        element = item.select_one(selector.value)
                        if element:
                            value = element.text.strip() if selector.attribute == "text" else element.get(selector.attribute, "")
                    
                    # Appliquer les transformations
                    if value and field.get("transformations"):
                        value = self._apply_transformations(value, field["transformations"])
                        
                    extracted_item[field["name"]] = value
                
                # Vérifier les champs requis
                if all(extracted_item.get(field["name"]) for field in config.fields if field.get("required", False)):
                    results.append(ScrapedItem(
                        source_url=config.base_url,
                        extracted_at=datetime.now().isoformat(),
                        data=extracted_item
                    ))
        
        return results
    
    def _apply_transformations(self, value: str, transformations: List[Dict[str, Any]]) -> Any:
        """Applique les transformations configurées à une valeur."""
        result = value
        
        for transform in transformations:
            transform_type = transform.get("type")
            parameters = transform.get("parameters", {})
            
            if transform_type == "strip":
                result = result.strip()
            elif transform_type == "to_number":
                result = self._string_to_number(result, parameters)
            elif transform_type == "replace":
                result = result.replace(parameters.get("old", ""), parameters.get("new", ""))
            elif transform_type == "regex_extract":
                pattern = parameters.get("pattern", "")
                if pattern:
                    matches = re.search(pattern, result)
                    if matches:
                        result = matches.group(parameters.get("group", 0))
            elif transform_type == "to_date":
                try:
                    from dateutil import parser
                    result = parser.parse(result).isoformat()
                except:
                    pass
                    
        return result
    
    def _string_to_number(self, value: str, params: Dict[str, Any]) -> Union[int, float, str]:
        """Convertit une chaîne en nombre."""
        try:
            # Nettoyer la chaîne
            clean_value = value.strip()
            
            # Remplacer les séparateurs selon la configuration
            if "thousands_separator" in params:
                clean_value = clean_value.replace(params["thousands_separator"], "")
            if "decimal_separator" in params:
                clean_value = clean_value.replace(params["decimal_separator"], ".")
                
            # Extraction des chiffres si spécifié
            if params.get("extract_digits", False):
                clean_value = "".join(c for c in clean_value if c.isdigit() or c == ".")
                
            # Conversion selon le type
            if "." in clean_value:
                return float(clean_value)
            else:
                return int(clean_value)
                
        except (ValueError, TypeError):
            return value
    
    async def _get_next_page_url(self, current_url: str, pagination_config: Dict[str, Any], html_content: str) -> Optional[str]:
        """Détermine l'URL de la page suivante selon la stratégie de pagination."""
        strategy = pagination_config.get("strategy", PaginationStrategy.LINK_NEXT)
        
        if strategy == PaginationStrategy.LINK_NEXT:
            # Rechercher un lien "suivant"
            soup = BeautifulSoup(html_content, "html.parser")
            next_link = soup.select_one(pagination_config.get("selector", {}).get("value", "a.next"))
            
            if next_link and next_link.has_attr("href"):
                next_url = next_link["href"]
                return urljoin(current_url, next_url)
                
        elif strategy == PaginationStrategy.URL_PATTERN:
            # Construire une URL selon un motif
            pattern = pagination_config.get("url_pattern", "")
            if pattern:
                # Extraire le numéro de page actuel
                current_page = self._extract_page_number(current_url)
                if current_page is not None:
                    next_page = current_page + 1
                    next_url = pattern.replace("{page}", str(next_page))
                    return next_url
                    
        elif strategy == PaginationStrategy.INCREMENT_PARAMETER:
            # Incrémenter un paramètre dans l'URL
            param = pagination_config.get("parameter", "page")
            # Extraire le numéro de page actuel
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            
            parsed_url = urlparse(current_url)
            query_params = parse_qs(parsed_url.query)
            
            current_page = int(query_params.get(param, [1])[0])
            next_page = current_page + 1
            
            query_params[param] = [str(next_page)]
            new_query = urlencode(query_params, doseq=True)
            
            parsed_url = parsed_url._replace(query=new_query)
            return urlunparse(parsed_url)
            
        return None
    
    def _extract_page_number(self, url: str) -> Optional[int]:
        """Extrait le numéro de page d'une URL."""
        # Tentative avec les paramètres de requête courants
        common_params = ["page", "p", "pg", "pagina"]
        
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        
        for param in common_params:
            if param in query_params:
                try:
                    return int(query_params[param][0])
                except (ValueError, IndexError):
                    pass
        
        # Tentative avec motif d'URL
        patterns = [
            r'/page/(\d+)/?',
            r'/p(\d+)/?',
            r'-page-(\d+)/?',
            r'page-(\d+)\.html'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    pass
                    
        # Par défaut: page 1
        return 1
    
    def _is_valid_url(self, url: str) -> bool:
        """Vérifie si une URL est valide."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    async def _enrich_scraped_data(self, items: List[ScrapedItem], config: ScrapingConfig) -> List[ScrapedItem]:
        """Enrichit les données scrapées avec des informations supplémentaires si configuré."""
        # Implémentation à faire selon les besoins
        return items
        
    def _format_output(self, items: List[ScrapedItem], output_format: str) -> Any:
        """Formate les résultats dans le format spécifié."""
        if not items:
            return []
            
        if output_format == "json":
            return [item.dict() for item in items]
            
        elif output_format == "csv":
            # Convertir en DataFrame puis en CSV
            data = [item.data for item in items]
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
            
        elif output_format == "excel":
            # Convertir en DataFrame puis en Excel
            data = [item.data for item in items]
            df = pd.DataFrame(data)
            import io
            output = io.BytesIO()
            df.to_excel(output, index=False)
            return output.getvalue()
            
        # Par défaut: retourner les éléments tels quels
        return items 
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
import json

from api.models.agent import (
    Agent, AgentQuery, AgentResponse, AgentMessage, 
    AgentStatus, AgentType, AgentCapability
)
from api.services.ollama_service import OllamaService

logger = logging.getLogger(__name__)

class AssistantAgent:
    """
    Agent IA avancé utilisant Ollama et offrant diverses capacités comme l'analyse de documents,
    la recherche web et le scraping.
    """
    
    def __init__(self, agent_config: Agent):
        self.config = agent_config
        self.logger = logging.getLogger(__name__)
        self.ollama_service = OllamaService()
        self.conversation_history = []
        self.capabilities = agent_config.settings.get("capabilities", [])
        self.model_name = agent_config.settings.get("model_name", "llama2")
        
    async def process_query(self, query: AgentQuery) -> AgentResponse:
        """
        Traite une requête utilisateur et génère une réponse en fonction des capacités de l'agent.
        
        Args:
            query: Requête de l'utilisateur
            
        Returns:
            AgentResponse: Réponse générée par l'agent
        """
        start_time = asyncio.get_event_loop().time()
        
        # Ajouter la requête à l'historique de conversation
        self.conversation_history.append({
            "role": "user",
            "content": query.content
        })
        
        # Déterminer quelle capacité utiliser
        thinking_process = []
        sources = []
        
        # Analyser l'intention de la requête
        thinking_process.append(f"Analyse de l'intention de la requête: '{query.content}'")
        
        # Exécuter les capacités appropriées
        if AgentCapability.WEB_SEARCH in self.capabilities and self._should_use_web_search(query.content):
            thinking_process.append("Utilisation de la recherche web pour collecter des informations...")
            web_results = await self._perform_web_search(query.content)
            thinking_process.append(f"Résultats de recherche web obtenus: {len(web_results)} sources")
            sources.extend(web_results)
        
        if AgentCapability.DOCUMENT_PROCESSING in self.capabilities and query.documents:
            thinking_process.append(f"Traitement de {len(query.documents)} documents...")
            doc_results = await self._process_documents(query.documents)
            thinking_process.append("Documents analysés et informations extraites")
            sources.extend(doc_results)
        
        if AgentCapability.WEB_SCRAPING in self.capabilities and self._should_use_web_scraping(query.content):
            thinking_process.append("Extraction de données web (scraping) initiée...")
            scraping_results = await self._perform_scraping(query.content)
            thinking_process.append(f"Extraction web terminée: {len(scraping_results)} éléments extraits")
            sources.extend(scraping_results)
            
        # Enrichir le prompt avec les sources
        enriched_prompt = self._build_enriched_prompt(query.content, sources)
        thinking_process.append("Génération de la réponse basée sur les informations collectées")
        
        # Générer la réponse avec Ollama
        response_content = await self._generate_response(enriched_prompt)
        
        # Ajouter la réponse à l'historique
        self.conversation_history.append({
            "role": "assistant",
            "content": response_content
        })
        
        # Calcul du temps de traitement
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Construire et retourner la réponse
        return AgentResponse(
            content=response_content,
            sources=sources,
            thinking_process="\n".join(thinking_process) if query.explain_thinking else None,
            processing_time=processing_time,
            conversation_id=query.conversation_id
        )
    
    def _should_use_web_search(self, query: str) -> bool:
        """Détermine si la recherche web devrait être utilisée pour cette requête."""
        # Mots-clés indiquant un besoin de recherche
        search_keywords = ["cherche", "trouve", "recherche", "information sur", "qu'est-ce que", "qui est"]
        return any(keyword in query.lower() for keyword in search_keywords)
    
    def _should_use_web_scraping(self, query: str) -> bool:
        """Détermine si le scraping web devrait être utilisé pour cette requête."""
        # Mots-clés indiquant un besoin de scraping
        scraping_keywords = ["extraire", "collecter", "scraper", "données de", "information de"]
        return any(keyword in query.lower() for keyword in scraping_keywords)
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Effectue une recherche web et retourne les résultats."""
        # TODO: Implémenter l'intégration réelle avec un service de recherche
        # Simulation des résultats de recherche
        return [
            {
                "title": f"Résultat de recherche pour {query}",
                "url": "https://example.com/search-result-1",
                "snippet": f"Informations pertinentes concernant {query}...",
                "relevance": 0.92
            },
            {
                "title": f"Documentation sur {query}",
                "url": "https://example.com/search-result-2",
                "snippet": f"Guide détaillé sur {query} et ses applications...",
                "relevance": 0.85
            }
        ]
    
    async def _process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Traite les documents fournis et extrait les informations pertinentes."""
        # TODO: Implémenter le traitement réel des documents
        # Simulation du traitement de documents
        return [
            {
                "title": doc.get("title", "Document sans titre"),
                "content_extract": f"Extrait du document {i}: informations extraites...",
                "relevance": 0.8
            }
            for i, doc in enumerate(documents)
        ]
    
    async def _perform_scraping(self, query: str) -> List[Dict[str, Any]]:
        """Effectue un scraping web basé sur la requête."""
        # TODO: Implémenter l'intégration réelle avec le service de scraping
        # Simulation des résultats de scraping
        return [
            {
                "source": "https://example.com/data-source-1",
                "data_type": "tableau",
                "extracted_at": "2023-06-15T14:30:00Z",
                "content": f"Données extraites concernant {query}..."
            }
        ]
    
    def _build_enriched_prompt(self, query: str, sources: List[Dict[str, Any]]) -> str:
        """Construit un prompt enrichi avec les sources pour le LLM."""
        prompt = f"Réponds à la question suivante: {query}\n\n"
        
        if sources:
            prompt += "Voici des informations pertinentes pour répondre:\n"
            for i, source in enumerate(sources):
                prompt += f"Source {i+1}:\n"
                for key, value in source.items():
                    if key not in ["relevance"]:  # Exclure certains champs techniques
                        prompt += f"- {key}: {value}\n"
                prompt += "\n"
        
        prompt += "\nRéponds de manière concise et précise en te basant sur les informations fournies."
        return prompt
    
    async def _generate_response(self, prompt: str) -> str:
        """Génère une réponse en utilisant Ollama."""
        try:
            # Conversion de la méthode synchrone en asynchrone
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.ollama_service.generate_sync(
                    model=self.model_name,
                    prompt=prompt,
                    options={"temperature": self.config.settings.get("temperature", 0.7)}
                )
            )
            
            return response.get("response", "Je n'ai pas pu générer de réponse. Veuillez réessayer.")
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            return f"Une erreur s'est produite lors de la génération de la réponse: {str(e)}" 
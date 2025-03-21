import logging
import requests
from typing import Dict, List, Any, Optional

class OllamaService:
    """Service pour interagir avec Ollama pour les modèles locaux."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.base_url = "http://localhost:11434"
        
    def list_models_sync(self) -> List[Dict[str, Any]]:
        """Liste tous les modèles disponibles dans Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.json().get("models", [])
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des modèles Ollama: {str(e)}")
            return []
    
    def generate_sync(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Génère une réponse avec un modèle Ollama."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "options": options or {}
            }
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            return response.json()
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération avec Ollama: {str(e)}")
            return {"error": str(e)}
    
    def chat_sync(self, model: str, messages: List[Dict[str, str]], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Utilise l'API de chat Ollama."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "options": options or {}
            }
            response = requests.post(f"{self.base_url}/api/chat", json=payload)
            return response.json()
        except Exception as e:
            self.logger.error(f"Erreur lors de l'utilisation du chat Ollama: {str(e)}")
            return {"error": str(e)}

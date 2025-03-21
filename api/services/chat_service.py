import logging
from typing import Dict, List, Any, Optional
import json
import time

class ChatService:
    """Service pour gérer les conversations et le chat avec les modèles."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conversations = {}  # Stocker les conversations par utilisateur
        
        # Vérifier si Oumi est disponible
        try:
            import oumi
            self.oumi_available = True
        except ImportError:
            self.logger.warning("Oumi n'est pas disponible. Le service de chat utilisera des alternatives.")
            self.oumi_available = False
    
    def create_conversation_sync(self, user_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Crée une nouvelle conversation pour un utilisateur."""
        try:
            conversation_id = f"conv_{user_id}_{int(time.time())}"
            
            if user_id not in self.conversations:
                self.conversations[user_id] = {}
            
            self.conversations[user_id][conversation_id] = {
                "id": conversation_id,
                "messages": [],
                "created_at": time.time(),
                "config": config or {
                    "model": "ollama/llama2",
                    "context_length": 10,
                    "system_prompt": "Tu es un assistant AI utile qui fournit des réponses claires et précises."
                }
            }
            
            return {
                "conversation_id": conversation_id,
                "status": "créée",
                "created_at": self.conversations[user_id][conversation_id]["created_at"]
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de la conversation: {str(e)}")
            return {"error": str(e)}
    
    def add_message_sync(self, user_id: str, conversation_id: str, 
                      message: str, sender: str = "user") -> Dict[str, Any]:
        """Ajoute un message à une conversation existante."""
        try:
            if user_id not in self.conversations or conversation_id not in self.conversations[user_id]:
                return {"error": "Conversation non trouvée"}
            
            msg_id = f"msg_{len(self.conversations[user_id][conversation_id]['messages']) + 1}"
            msg_data = {
                "id": msg_id,
                "sender": sender,
                "content": message,
                "timestamp": time.time()
            }
            
            self.conversations[user_id][conversation_id]["messages"].append(msg_data)
            
            return {
                "message_id": msg_id,
                "status": "ajouté",
                "conversation_id": conversation_id
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout du message: {str(e)}")
            return {"error": str(e)}
    
    def generate_response_sync(self, user_id: str, conversation_id: str) -> Dict[str, Any]:
        """Génère une réponse pour la conversation en cours."""
        try:
            if user_id not in self.conversations or conversation_id not in self.conversations[user_id]:
                return {"error": "Conversation non trouvée"}
            
            conversation = self.conversations[user_id][conversation_id]
            messages = conversation["messages"]
            config = conversation["config"]
            
            # Préparer l'historique de messages
            history = []
            for msg in messages:
                if msg["sender"] == "user":
                    history.append({"role": "user", "content": msg["content"]})
                elif msg["sender"] == "assistant":
                    history.append({"role": "assistant", "content": msg["content"]})
            
            # Générer une réponse en utilisant Oumi ou une alternative
            if self.oumi_available and "oumi" in config.get("model", ""):
                # Utiliser Oumi pour la génération
                response_content = self._generate_with_oumi(history, config)
            else:
                # Utiliser Ollama comme fallback
                from services.ollama_service import OllamaService
                ollama_service = OllamaService()
                response = ollama_service.chat_sync(
                    model=config["model"].replace("ollama/", ""),
                    messages=history
                )
                response_content = response.get("message", {}).get("content", "Je ne peux pas générer de réponse pour le moment.")
            
            # Ajouter la réponse à la conversation
            response_id = self.add_message_sync(
                user_id=user_id,
                conversation_id=conversation_id,
                message=response_content,
                sender="assistant"
            )
            
            return {
                "response_id": response_id.get("message_id"),
                "content": response_content,
                "conversation_id": conversation_id
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la réponse: {str(e)}")
            return {"error": str(e)}
    
    def _generate_with_oumi(self, messages: List[Dict[str, str]], config: Dict[str, Any]) -> str:
        """Génère une réponse en utilisant Oumi."""
        # Version simplifiée sans oumi
        return "Je suis un assistant IA. Cette réponse est générée sans oumi pour le moment."

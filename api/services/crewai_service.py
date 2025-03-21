import logging
from typing import Dict, List, Any, Optional
import json

class CrewAIService:
    """Service pour créer et gérer des équipes d'agents IA avec CrewAI."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.crews = {}  # Stocker les équipes créées
        self.agents = {}  # Stocker les agents créés
        
        # Vérifier si CrewAI est disponible
        try:
            import crewai
            self.crewai_available = True
        except ImportError:
            self.logger.warning("CrewAI n'est pas disponible. Certaines fonctionnalités seront limitées.")
            self.crewai_available = False
    
    def create_agent_sync(self, agent_config: Dict[str, Any]) -> Dict[str, Any]:
        """Crée un nouvel agent IA basé sur la configuration."""
        if not self.crewai_available:
            return {"error": "CrewAI n'est pas disponible sur ce serveur."}
        
        try:
            from crewai import Agent
            from langchain.llms.ollama import Ollama
            
            # Configurer le modèle LLM (Ollama)
            llm_type = agent_config.get("llm_type", "ollama")
            if llm_type == "ollama":
                llm = Ollama(model=agent_config.get("model", "llama2"))
            else:
                return {"error": f"Type de LLM non supporté: {llm_type}"}
            
            # Créer l'agent
            agent = Agent(
                role=agent_config.get("role", "Assistant"),
                goal=agent_config.get("goal", "Aider l'utilisateur"),
                backstory=agent_config.get("backstory", ""),
                llm=llm,
                verbose=True
            )
            
            # Stocker l'agent
            agent_id = agent_config.get("id", str(len(self.agents) + 1))
            self.agents[agent_id] = agent
            
            return {
                "id": agent_id,
                "role": agent_config.get("role"),
                "goal": agent_config.get("goal"),
                "status": "créé avec succès"
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de l'agent: {str(e)}")
            return {"error": str(e)}
    
    def create_crew_sync(self, crew_config: Dict[str, Any]) -> Dict[str, Any]:
        """Crée une équipe d'agents IA."""
        if not self.crewai_available:
            return {"error": "CrewAI n'est pas disponible sur ce serveur."}
        
        try:
            from crewai import Crew, Task, Process
            
            # Récupérer les agents
            crew_agents = []
            for agent_id in crew_config.get("agent_ids", []):
                if agent_id in self.agents:
                    crew_agents.append(self.agents[agent_id])
                else:
                    return {"error": f"Agent non trouvé: {agent_id}"}
            
            if not crew_agents:
                return {"error": "Aucun agent valide fourni pour l'équipe"}
            
            # Créer les tâches
            tasks = []
            for task_config in crew_config.get("tasks", []):
                agent_id = task_config.get("agent_id")
                if agent_id not in self.agents:
                    return {"error": f"Agent non trouvé pour la tâche: {agent_id}"}
                
                task = Task(
                    description=task_config.get("description", ""),
                    agent=self.agents[agent_id],
                    expected_output=task_config.get("expected_output", "")
                )
                tasks.append(task)
            
            # Créer l'équipe
            crew = Crew(
                agents=crew_agents,
                tasks=tasks,
                verbose=True,
                process=Process.sequential
            )
            
            # Stocker l'équipe
            crew_id = crew_config.get("id", str(len(self.crews) + 1))
            self.crews[crew_id] = crew
            
            return {
                "id": crew_id,
                "name": crew_config.get("name", "Équipe IA"),
                "agent_count": len(crew_agents),
                "task_count": len(tasks),
                "status": "créée avec succès"
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de la création de l'équipe: {str(e)}")
            return {"error": str(e)}
    
    def run_crew_sync(self, crew_id: str, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Exécute une équipe d'agents avec les entrées fournies."""
        if not self.crewai_available:
            return {"error": "CrewAI n'est pas disponible sur ce serveur."}
        
        try:
            if crew_id not in self.crews:
                return {"error": f"Équipe non trouvée: {crew_id}"}
            
            crew = self.crews[crew_id]
            result = crew.kickoff(inputs=inputs or {})
            
            return {
                "crew_id": crew_id,
                "status": "terminé",
                "result": result
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution de l'équipe: {str(e)}")
            return {"error": str(e)}

"""
Agent d'analyse de données MPTOO
Copyright 2025 Mohammed Amine Taybi
Licensed under the Apache License, Version 2.0
"""

import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI

# Définition de l'agent d'analyse de données
class DataAgent:
    """
    Agent intelligent qui analyse les données et génère des insights
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        verbose: bool = False,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialise l'agent d'analyse de données
        
        Args:
            model_name: Nom du modèle LLM à utiliser
            temperature: Niveau de créativité du modèle (0.0-1.0)
            verbose: Affiche les étapes de raisonnement
            tools: Outils personnalisés pour l'agent
        """
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        self.tools = tools or self._get_default_tools()
        self.agent_executor = self._create_agent()
    
    def _get_default_tools(self) -> List[BaseTool]:
        """Retourne les outils par défaut de l'agent"""
        
        @tool
        def load_csv(file_path: str) -> pd.DataFrame:
            """Charge un fichier CSV et retourne un pandas DataFrame"""
            if not os.path.exists(file_path):
                return f"Erreur: Fichier '{file_path}' introuvable"
            try:
                df = pd.read_csv(file_path)
                return df.head(5).to_string() + f"\nDimensions: {df.shape}"
            except Exception as e:
                return f"Erreur de chargement: {str(e)}"
        
        @tool
        def analyze_dataframe(df_description: str, analysis_type: str) -> str:
            """
            Analyse un DataFrame et génère des insights.
            
            Args:
                df_description: Description du DataFrame (utilisez le résultat de load_csv)
                analysis_type: Type d'analyse ('summary', 'correlation', 'missing', etc.)
            """
            return f"Analyse '{analysis_type}' du DataFrame:\n{df_description}\n[Résultats d'analyse simulés]"
        
        @tool
        def generate_visualization(data_description: str, viz_type: str) -> str:
            """
            Génère une visualisation basée sur les données.
            
            Args:
                data_description: Description des données à visualiser
                viz_type: Type de visualisation ('bar', 'line', 'scatter', etc.)
            """
            return f"[Visualisation {viz_type} générée basée sur: {data_description}]"
        
        return [load_csv, analyze_dataframe, generate_visualization]
    
    def _create_agent(self) -> AgentExecutor:
        """Crée et configure l'agent"""
        llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        
        prompt_template = """
        Tu es un expert en analyse de données pour MPTOO.
        
        ## Contexte
        MPTOO est une application de gestion qui aide les entreprises à analyser leurs données.
        Tu as accès à des outils pour charger des fichiers CSV, analyser des données, et générer des visualisations.
        
        ## Instructions
        - Sois précis et factuel dans tes analyses
        - Indique les limites ou incertitudes dans tes conclusions
        - Propose des visualisations pertinentes pour les données
        
        ## Outils
        {tools}
        
        ## Tâche
        {input}
        
        ## Réflexion pas à pas
        {agent_scratchpad}
        """
        
        prompt = PromptTemplate.from_template(prompt_template)
        agent = create_react_agent(llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=10,
        )
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Exécute l'agent avec une requête
        
        Args:
            query: Question ou instruction pour l'agent
        
        Returns:
            Résultat de l'exécution de l'agent
        """
        try:
            return self.agent_executor.invoke({"input": query})
        except Exception as e:
            return {"output": f"Erreur lors de l'exécution de l'agent: {str(e)}"}


def main():
    """Démonstration de l'utilisation de l'agent"""
    # Créer et exécuter l'agent
    agent = DataAgent(verbose=True)
    result = agent.run("Charge le fichier 'data/example.csv' et analyse les tendances principales")
    print("\nRésultat final:", result["output"])


if __name__ == "__main__":
    main() 
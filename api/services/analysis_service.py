import logging
from typing import Dict, Any, List, Optional
import time
import random
from datetime import datetime
import json
import os

from api.models.analysis import AnalysisRequest, AnalysisResult, AnalysisResultData, Insight, AnalysisMetadata

logger = logging.getLogger(__name__)

# Charger des données mockées depuis un fichier JSON si disponible
MOCK_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_analysis.json")
mock_data = {}

try:
    if os.path.exists(MOCK_DATA_PATH):
        with open(MOCK_DATA_PATH, 'r', encoding='utf-8') as f:
            mock_data = json.load(f)
        logger.info(f"Données mockées chargées depuis {MOCK_DATA_PATH}")
except Exception as e:
    logger.warning(f"Impossible de charger les données mockées: {str(e)}")

class AnalysisService:
    """Service pour l'analyse de contenu."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.mock_data = mock_data
        self.dev_mode = os.environ.get("ENVIRONMENT", "development") == "development"
        
        # Prédéfinir quelques sujets pour des réponses consistantes
        self.predefined_topics = {
            "finance": self._get_finance_insights,
            "marketing": self._get_marketing_insights,
            "technologie": self._get_technology_insights,
            "santé": self._get_health_insights,
            "default": self._generate_sample_insights,
        }
    
    async def analyze_content(self, request: AnalysisRequest, interface_mode: str = "client") -> AnalysisResult:
        """
        Analyse un contenu basé sur un sujet donné.
        
        Args:
            request: AnalysisRequest contenant le sujet à analyser
            interface_mode: Mode d'interface (client ou expert)
            
        Returns:
            AnalysisResult avec les insights générés
        """
        self.logger.info(f"Analyzing content for topic: {request.topic}, mode: {interface_mode}")
        
        # Vérifier si une réponse mockée existe pour ce sujet
        if request.topic.lower() in self.mock_data:
            self.logger.info(f"Utilisation d'une réponse mockée pour {request.topic}")
            
            # Simuler un délai pour être plus réaliste
            time.sleep(random.uniform(0.2, 0.8))
            
            mock_result = self.mock_data[request.topic.lower()]
            
            # Adapter le résultat au mode d'interface
            if interface_mode == "client" and "client_version" in mock_result:
                insights_data = mock_result["client_version"]
            elif interface_mode == "expert" and "expert_version" in mock_result:
                insights_data = mock_result["expert_version"]
            else:
                insights_data = mock_result
                
            # Convertir les données mockées en objets Insight
            insights = []
            for item in insights_data.get("insights", []):
                insight_data = {k: v for k, v in item.items()}
                insights.append(Insight(**insight_data))
            
            # Créer les métadonnées (uniquement pour le mode expert)
            metadata = None
            if interface_mode == "expert" and "metadata" in mock_result:
                metadata = AnalysisMetadata(**mock_result["metadata"])
            
            # Construction du résultat
            result_data = AnalysisResultData(insights=insights, metadata=metadata)
            
            return AnalysisResult(
                topic=request.topic,
                results=result_data,
                mode=interface_mode
            )
        
        # Si aucune réponse mockée n'existe, générer une réponse dynamique
        
        # Simulation du temps de traitement (à remplacer par une vraie analyse)
        processing_time = random.uniform(1.5, 4.0)
        time.sleep(min(1.0, processing_time/3))  # Simuler un délai (réduit pour le développement)
        
        # Sélectionner la méthode de génération en fonction du sujet
        topic_lower = request.topic.lower()
        generation_method = None
        
        for key, method in self.predefined_topics.items():
            if key in topic_lower:
                generation_method = method
                break
                
        if not generation_method:
            generation_method = self.predefined_topics["default"]
        
        # Création des insights avec la méthode sélectionnée
        insights = generation_method(request.topic, interface_mode)
        
        # Création des métadonnées (uniquement pour le mode expert)
        metadata = None
        if interface_mode == "expert":
            metadata = AnalysisMetadata(
                analysis_depth=request.depth,
                sources_count=random.randint(5, 20),
                processing_time=f"{processing_time:.1f}s",
                confidence_score=round(random.uniform(0.65, 0.95), 2),
                model_version="MPtoO Analysis v2.3"
            )
        
        # Construction du résultat
        result_data = AnalysisResultData(insights=insights, metadata=metadata)
        
        return AnalysisResult(
            topic=request.topic,
            results=result_data,
            mode=interface_mode
        )
    
    def _generate_sample_insights(self, topic: str, mode: str) -> List[Insight]:
        """
        Génère des insights d'exemple génériques pour le développement.
        
        Args:
            topic: Le sujet de l'analyse
            mode: Mode d'interface (client ou expert)
            
        Returns:
            Liste d'insights générés
        """
        insights = []
        
        # Créer quelques insights de base
        base_insights = [
            {
                "title": f"Analyse de '{topic}' - Point clé 1",
                "summary": f"Résumé simplifié du premier point clé concernant {topic}",
                "confidence": round(random.uniform(0.75, 0.95), 2)
            },
            {
                "title": f"Analyse de '{topic}' - Point clé 2",
                "summary": f"Résumé simplifié du second point clé concernant {topic}",
                "confidence": round(random.uniform(0.65, 0.85), 2)
            }
        ]
        
        # Ajouter plus de détails pour le mode expert
        for base in base_insights:
            if mode == "expert":
                insight = Insight(
                    title=base["title"],
                    description=f"Description détaillée pour {base['title']}",
                    summary=base["summary"],
                    confidence=base["confidence"],
                    sources=[f"source{i}" for i in range(1, random.randint(3, 6))],
                    related_topics=[f"related_topic{i}" for i in range(1, random.randint(3, 5))]
                )
            else:
                insight = Insight(
                    title=base["title"],
                    summary=base["summary"],
                    confidence=base["confidence"]
                )
            
            insights.append(insight)
        
        return insights
    
    def _get_finance_insights(self, topic: str, mode: str) -> List[Insight]:
        """Génère des insights spécifiques au domaine de la finance."""
        insights = []
        
        finance_insights = [
            {
                "title": "Tendances du marché financier",
                "summary": f"Analyse des tendances actuelles du marché en relation avec {topic}",
                "description": "Les indicateurs économiques montrent une tendance haussière sur les 6 derniers mois, avec une augmentation de la volatilité dans certains secteurs clés.",
                "confidence": round(random.uniform(0.70, 0.90), 2)
            },
            {
                "title": "Analyse de risque financier",
                "summary": f"Évaluation des risques financiers potentiels pour {topic}",
                "description": "L'analyse de risque indique un niveau modéré avec des facteurs d'atténuation possibles, notamment la diversification et les couvertures stratégiques.",
                "confidence": round(random.uniform(0.65, 0.85), 2)
            },
            {
                "title": "Perspectives d'investissement",
                "summary": f"Recommandations d'investissement liées à {topic}",
                "description": "Les données suggèrent des opportunités d'investissement dans les secteurs de croissance, avec un horizon temporel recommandé de 3 à 5 ans.",
                "confidence": round(random.uniform(0.60, 0.80), 2)
            }
        ]
        
        for base in finance_insights:
            if mode == "expert":
                insight = Insight(
                    title=base["title"],
                    description=base["description"],
                    summary=base["summary"],
                    confidence=base["confidence"],
                    sources=["Bloomberg", "Financial Times", "Reuters", "MarketWatch"],
                    related_topics=["Investissements", "Marchés boursiers", "Analyse de risque", "Prévisions économiques"]
                )
            else:
                insight = Insight(
                    title=base["title"],
                    summary=base["summary"],
                    confidence=base["confidence"]
                )
            
            insights.append(insight)
        
        return insights
    
    def _get_marketing_insights(self, topic: str, mode: str) -> List[Insight]:
        """Génère des insights spécifiques au domaine du marketing."""
        insights = []
        
        marketing_insights = [
            {
                "title": "Analyse du comportement des consommateurs",
                "summary": f"Étude des tendances de comportement des consommateurs pour {topic}",
                "description": "Les données montrent une évolution vers plus de consommation consciente et durable, avec un accent sur l'authenticité et la transparence des marques.",
                "confidence": round(random.uniform(0.75, 0.90), 2)
            },
            {
                "title": "Stratégies de contenu digital",
                "summary": f"Analyse des stratégies de contenu efficaces pour {topic}",
                "description": "Le contenu vidéo court et les expériences interactives montrent les meilleurs taux d'engagement, particulièrement sur les plateformes mobiles.",
                "confidence": round(random.uniform(0.70, 0.85), 2)
            },
            {
                "title": "Optimisation des canaux marketing",
                "summary": f"Recommandations sur les canaux marketing les plus efficaces pour {topic}",
                "description": "Une approche omnicanale avec accent sur les médias sociaux et le marketing d'influence offre le meilleur ROI pour ce segment de marché.",
                "confidence": round(random.uniform(0.65, 0.80), 2)
            }
        ]
        
        for base in marketing_insights:
            if mode == "expert":
                insight = Insight(
                    title=base["title"],
                    description=base["description"],
                    summary=base["summary"],
                    confidence=base["confidence"],
                    sources=["Marketing Week", "AdAge", "eMarketer", "HubSpot Research"],
                    related_topics=["Comportement consommateur", "Marketing digital", "ROI marketing", "Tendances média"]
                )
            else:
                insight = Insight(
                    title=base["title"],
                    summary=base["summary"],
                    confidence=base["confidence"]
                )
            
            insights.append(insight)
        
        return insights
    
    def _get_technology_insights(self, topic: str, mode: str) -> List[Insight]:
        """Génère des insights spécifiques au domaine de la technologie."""
        insights = []
        
        tech_insights = [
            {
                "title": "Tendances technologiques émergentes",
                "summary": f"Analyse des technologies émergentes pertinentes pour {topic}",
                "description": "L'IA générative, l'informatique quantique et les technologies vertes sont identifiées comme les tendances les plus transformatives pour les 3-5 prochaines années.",
                "confidence": round(random.uniform(0.75, 0.90), 2)
            },
            {
                "title": "Analyse d'impact technologique",
                "summary": f"Évaluation de l'impact potentiel des nouvelles technologies sur {topic}",
                "description": "L'automatisation et l'IA devraient transformer significativement les processus opérationnels, avec un potentiel d'amélioration de l'efficacité de 30-40%.",
                "confidence": round(random.uniform(0.70, 0.85), 2)
            },
            {
                "title": "Défis d'implémentation technologique",
                "summary": f"Identification des défis d'adoption technologique pour {topic}",
                "description": "La résistance organisationnelle, le manque de compétences spécialisées et l'intégration avec les systèmes existants représentent les principaux obstacles à surmonter.",
                "confidence": round(random.uniform(0.65, 0.80), 2)
            }
        ]
        
        for base in tech_insights:
            if mode == "expert":
                insight = Insight(
                    title=base["title"],
                    description=base["description"],
                    summary=base["summary"],
                    confidence=base["confidence"],
                    sources=["Gartner", "MIT Technology Review", "TechCrunch", "IEEE Spectrum"],
                    related_topics=["Intelligence artificielle", "Transformation digitale", "Innovation technologique", "Cybersécurité"]
                )
            else:
                insight = Insight(
                    title=base["title"],
                    summary=base["summary"],
                    confidence=base["confidence"]
                )
            
            insights.append(insight)
        
        return insights
    
    def _get_health_insights(self, topic: str, mode: str) -> List[Insight]:
        """Génère des insights spécifiques au domaine de la santé."""
        insights = []
        
        health_insights = [
            {
                "title": "Tendances en santé publique",
                "summary": f"Analyse des tendances actuelles en santé publique liées à {topic}",
                "description": "La médecine préventive et personnalisée gagne en importance, avec un accent croissant sur le bien-être mental et la santé holistique.",
                "confidence": round(random.uniform(0.75, 0.90), 2)
            },
            {
                "title": "Innovations médicales",
                "summary": f"Aperçu des innovations médicales pertinentes pour {topic}",
                "description": "Les thérapies géniques, la télémédecine et les dispositifs de santé connectés représentent les innovations les plus prometteuses dans ce domaine.",
                "confidence": round(random.uniform(0.70, 0.85), 2)
            },
            {
                "title": "Défis des systèmes de santé",
                "summary": f"Identification des défis systémiques en santé liés à {topic}",
                "description": "L'accessibilité, l'équité dans les soins et la durabilité financière des systèmes de santé restent les défis majeurs à adresser.",
                "confidence": round(random.uniform(0.65, 0.80), 2)
            }
        ]
        
        for base in health_insights:
            if mode == "expert":
                insight = Insight(
                    title=base["title"],
                    description=base["description"],
                    summary=base["summary"],
                    confidence=base["confidence"],
                    sources=["WHO", "The Lancet", "JAMA", "New England Journal of Medicine"],
                    related_topics=["Santé publique", "Innovation médicale", "Politique de santé", "Médecine préventive"]
                )
            else:
                insight = Insight(
                    title=base["title"],
                    summary=base["summary"],
                    confidence=base["confidence"]
                )
            
            insights.append(insight)
        
        return insights 
import logging
import time
import uuid
import json
import os
import random
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from api.models.fine_tuning import (
    FineTuningConfig, FineTuningTask, FineTuningStatus, FineTuningRequest,
    FineTuningResponse, FineTuningResult, ModelType, EvaluationMetric,
    TrainingStats, TrainingCheckpoint, ModelPrediction
)

logger = logging.getLogger(__name__)

class FineTuningService:
    """Service pour la gestion des tâches de fine-tuning."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tasks_db = {}  # Simulation de base de données
        self.models_db = {}  # Modèles fine-tunés
        self.active_tasks = {}  # Tâches en cours d'exécution
        self.dev_mode = os.environ.get("ENVIRONMENT", "development") == "development"
        
        # En mode développement, charger des tâches prédéfinies
        if self.dev_mode:
            self._load_predefined_tasks()
    
    def _load_predefined_tasks(self):
        """Charge des tâches prédéfinies pour le développement."""
        self.logger.info("Chargement des tâches de fine-tuning prédéfinies pour le développement")
        
        # Exemple de tâche complétée - Classification de sentiment
        sentiment_task_id = str(uuid.uuid4())
        model_id = f"sentiment-model-{sentiment_task_id[:8]}"
        
        sentiment_task = FineTuningTask(
            id=sentiment_task_id,
            config=FineTuningConfig(
                name="Sentiment Analysis Model",
                description="Classification de sentiment pour français et anglais",
                model_type=ModelType.CLASSIFICATION,
                base_model="distilbert-base-multilingual-cased",
                dataset_sources=[
                    {
                        "name": "French Sentiment Dataset",
                        "type": "file",
                        "location": "/data/sentiment_fr.csv",
                        "format": "csv",
                        "options": {
                            "text_column": "text",
                            "label_column": "sentiment"
                        }
                    },
                    {
                        "name": "English Sentiment Dataset",
                        "type": "file",
                        "location": "/data/sentiment_en.csv",
                        "format": "csv",
                        "options": {
                            "text_column": "text",
                            "label_column": "sentiment"
                        }
                    }
                ],
                evaluation={
                    "metrics": [
                        EvaluationMetric.ACCURACY,
                        EvaluationMetric.F1,
                        EvaluationMetric.PRECISION,
                        EvaluationMetric.RECALL
                    ],
                    "eval_steps": 100,
                    "save_best_model": True,
                    "best_model_metric": EvaluationMetric.F1
                },
                output_dir=f"/models/{model_id}",
                hyperparameters={
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "epochs": 3
                },
                tags=["sentiment", "classification", "multilingual"]
            ),
            status=FineTuningStatus.COMPLETED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            model_path=f"/models/{model_id}/final",
            stats=TrainingStats(
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_steps=1500,
                total_epochs=3,
                training_time_seconds=5400,
                final_loss=0.1823,
                best_metrics={
                    "accuracy": 0.923,
                    "f1": 0.915,
                    "precision": 0.903,
                    "recall": 0.928
                },
                checkpoints=[
                    TrainingCheckpoint(
                        step=500,
                        loss=0.3245,
                        learning_rate=2e-5,
                        metrics={
                            "accuracy": 0.856
                        },
                        save_path=f"/models/{model_id}/checkpoint-500"
                    ),
                    TrainingCheckpoint(
                        step=1000,
                        loss=0.2134,
                        learning_rate=1e-5,
                        metrics={
                            "accuracy": 0.897
                        },
                        save_path=f"/models/{model_id}/checkpoint-1000"
                    ),
                    TrainingCheckpoint(
                        step=1500,
                        loss=0.1823,
                        learning_rate=5e-6,
                        metrics={
                            "accuracy": 0.923
                        },
                        save_path=f"/models/{model_id}/checkpoint-1500"
                    )
                ]
            )
        )
        
        self.tasks_db[sentiment_task_id] = sentiment_task
        self.models_db[model_id] = {
            "id": model_id,
            "name": "Sentiment Analysis Model",
            "task_id": sentiment_task_id,
            "type": ModelType.CLASSIFICATION,
            "base_model": "distilbert-base-multilingual-cased",
            "metrics": {
                "accuracy": 0.923,
                "f1": 0.915
            },
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "size_mb": 256,
            "labels": ["positive", "negative", "neutral"]
        }
        
        # Exemple de tâche en attente - Question Answering
        qa_task_id = str(uuid.uuid4())
        
        qa_task = FineTuningTask(
            id=qa_task_id,
            config=FineTuningConfig(
                name="Question Answering Model",
                description="Modèle de réponse aux questions pour documents techniques",
                model_type=ModelType.QA,
                base_model="bert-base-multilingual-cased",
                dataset_sources=[
                    {
                        "name": "Technical Documentation QA",
                        "type": "file",
                        "location": "/data/tech_qa.json",
                        "format": "jsonl",
                        "options": {
                            "context_column": "context",
                            "question_column": "question",
                            "answer_column": "answer"
                        }
                    }
                ],
                evaluation={
                    "metrics": [
                        EvaluationMetric.ACCURACY,
                        EvaluationMetric.F1
                    ],
                    "eval_steps": 100,
                    "save_best_model": True,
                    "best_model_metric": EvaluationMetric.F1
                },
                output_dir=f"/models/qa-model-{qa_task_id[:8]}",
                hyperparameters={
                    "learning_rate": 3e-5,
                    "batch_size": 8,
                    "epochs": 2
                },
                tags=["qa", "documentation", "technical"]
            ),
            status=FineTuningStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="system",
            stats=TrainingStats()
        )
        
        self.tasks_db[qa_task_id] = qa_task
    
    async def create_task(self, request: FineTuningRequest, user_id: str) -> FineTuningResponse:
        """
        Crée une nouvelle tâche de fine-tuning.
        
        Args:
            request: Requête de configuration du fine-tuning
            user_id: ID de l'utilisateur créant la tâche
            
        Returns:
            Réponse avec l'ID de la tâche créée
        """
        task_id = str(uuid.uuid4())
        
        # Créer la tâche
        task = FineTuningTask(
            id=task_id,
            config=request.config,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by=user_id
        )
        
        self.tasks_db[task_id] = task
        
        self.logger.info(f"Tâche de fine-tuning créée: {task.config.name} (ID: {task_id})")
        
        # En mode développement, on peut exécuter immédiatement
        if self.dev_mode and random.random() > 0.7:
            asyncio.create_task(self._execute_task(task_id))
            message = "Tâche créée et démarrée (mode développement)"
        else:
            message = "Tâche créée avec succès et mise en file d'attente"
        
        return FineTuningResponse(
            task_id=task_id,
            message=message,
            status=task.status
        )
    
    async def get_task(self, task_id: str) -> Optional[FineTuningTask]:
        """
        Récupère une tâche par son ID.
        
        Args:
            task_id: ID de la tâche
            
        Returns:
            La tâche ou None si non trouvée
        """
        return self.tasks_db.get(task_id)
    
    async def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'un modèle par son ID.
        
        Args:
            model_id: ID du modèle
            
        Returns:
            Les informations du modèle ou None si non trouvé
        """
        return self.models_db.get(model_id)
    
    async def get_task_result(self, task_id: str) -> Optional[FineTuningResult]:
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
        
        if task.status != FineTuningStatus.COMPLETED:
            return None
        
        # Trouver le modèle associé
        model_id = None
        model_info = {}
        
        for mid, model in self.models_db.items():
            if model.get("task_id") == task_id:
                model_id = mid
                model_info = model
                break
        
        if not model_id:
            return None
        
        return FineTuningResult(
            task=task,
            metrics=task.stats.best_metrics,
            model_info=model_info,
            test_results={
                "accuracy": task.stats.best_metrics.get("accuracy", 0.0),
                "examples": self._generate_test_examples(task.config.model_type)
            }
        )
    
    async def get_tasks(self, user_id: Optional[str] = None, status: Optional[FineTuningStatus] = None) -> List[FineTuningTask]:
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
    
    async def get_models(self, user_id: Optional[str] = None, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        Récupère tous les modèles, avec filtrage optionnel.
        
        Args:
            user_id: ID de l'utilisateur pour filtrer ses modèles
            model_type: Type de modèle pour filtrer par type
            
        Returns:
            Liste des modèles correspondant aux critères
        """
        models = list(self.models_db.values())
        
        # Filtrer par utilisateur si spécifié
        if user_id:
            # Récupérer les tâches de l'utilisateur
            user_tasks = [t.id for t in await self.get_tasks(user_id)]
            models = [m for m in models if m.get("task_id") in user_tasks]
        
        # Filtrer par type de modèle si spécifié
        if model_type:
            models = [m for m in models if m.get("type") == model_type]
        
        return models
    
    async def start_task(self, task_id: str) -> bool:
        """
        Démarre l'exécution d'une tâche en attente.
        
        Args:
            task_id: ID de la tâche à démarrer
            
        Returns:
            True si la tâche a été démarrée, False sinon
        """
        task = await self.get_task(task_id)
        if not task or task.status != FineTuningStatus.PENDING:
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
        if not task or task.status not in [FineTuningStatus.PENDING, FineTuningStatus.TRAINING, FineTuningStatus.PREPROCESSING]:
            return False
        
        # Si la tâche est en cours d'exécution, il faudrait l'arrêter
        if task_id in self.active_tasks:
            # TODO: implémenter l'arrêt réel de la tâche
            pass
        
        task.status = FineTuningStatus.CANCELLED
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
        if self.tasks_db[task_id].status in [FineTuningStatus.TRAINING, FineTuningStatus.PREPROCESSING]:
            return False
        
        # Supprimer aussi les modèles associés
        models_to_delete = []
        for model_id, model in self.models_db.items():
            if model.get("task_id") == task_id:
                models_to_delete.append(model_id)
        
        for model_id in models_to_delete:
            del self.models_db[model_id]
        
        del self.tasks_db[task_id]
        
        return True
    
    async def predict(self, model_id: str, input_data: Any) -> ModelPrediction:
        """
        Génère une prédiction à partir d'un modèle fine-tuné.
        
        Args:
            model_id: ID du modèle à utiliser
            input_data: Données d'entrée pour la prédiction
            
        Returns:
            Résultat de la prédiction
        """
        model = await self.get_model(model_id)
        if not model:
            raise ValueError(f"Modèle non trouvé: {model_id}")
        
        start_time = time.time()
        
        # Simuler une prédiction en fonction du type de modèle
        if model["type"] == ModelType.CLASSIFICATION:
            result = self._simulate_classification_prediction(input_data, model)
        elif model["type"] == ModelType.QA:
            result = self._simulate_qa_prediction(input_data, model)
        else:
            result = self._simulate_generic_prediction(input_data, model)
        
        execution_time = time.time() - start_time
        
        return ModelPrediction(
            model_id=model_id,
            input=input_data,
            output=result["output"],
            confidence=result.get("confidence"),
            processing_time=execution_time,
            metadata={
                "model_name": model["name"],
                "model_type": model["type"],
                "version": model.get("version", "1.0.0")
            }
        )
    
    def _simulate_classification_prediction(self, input_data: Any, model: Dict[str, Any]) -> Dict[str, Any]:
        """Simule une prédiction de classification."""
        text = input_data if isinstance(input_data, str) else str(input_data)
        
        # Simuler un délai de traitement
        time.sleep(random.uniform(0.05, 0.2))
        
        # Récupérer les étiquettes possibles
        labels = model.get("labels", ["positive", "negative", "neutral"])
        
        # Simuler des poids pour chaque étiquette
        weights = [random.random() for _ in labels]
        total = sum(weights)
        normalized_weights = [w / total for w in weights]
        
        # Choisir une étiquette en fonction des poids
        prediction = random.choices(labels, weights=normalized_weights, k=1)[0]
        
        # Simuler une confiance
        confidence = random.uniform(0.7, 0.95)
        
        return {
            "output": prediction,
            "confidence": confidence,
            "probabilities": {label: weight for label, weight in zip(labels, normalized_weights)}
        }
    
    def _simulate_qa_prediction(self, input_data: Dict[str, str], model: Dict[str, Any]) -> Dict[str, Any]:
        """Simule une prédiction de question-réponse."""
        # Vérifier que l'entrée contient une question et un contexte
        if not isinstance(input_data, dict) or "question" not in input_data or "context" not in input_data:
            return {
                "output": "Format d'entrée invalide. Attendu: {'question': '...', 'context': '...'}",
                "confidence": 0.0
            }
        
        question = input_data["question"]
        context = input_data["context"]
        
        # Simuler un délai de traitement
        time.sleep(random.uniform(0.1, 0.3))
        
        # Simuler une extraction de réponse à partir du contexte
        # Dans une implémentation réelle, cela utiliserait le modèle pour extraire la réponse
        words = context.split()
        if len(words) < 10:
            answer = context
        else:
            start_idx = random.randint(0, max(0, len(words) - 10))
            end_idx = min(len(words), start_idx + random.randint(3, 10))
            answer = " ".join(words[start_idx:end_idx])
        
        confidence = random.uniform(0.6, 0.9)
        
        return {
            "output": answer,
            "confidence": confidence,
            "start_position": context.find(answer) if answer in context else 0,
            "end_position": context.find(answer) + len(answer) if answer in context else len(context)
        }
    
    def _simulate_generic_prediction(self, input_data: Any, model: Dict[str, Any]) -> Dict[str, Any]:
        """Simule une prédiction générique."""
        # Simuler un délai de traitement
        time.sleep(random.uniform(0.1, 0.5))
        
        return {
            "output": f"Prédiction simulée pour le modèle {model['name']} avec entrée: {str(input_data)[:50]}...",
            "confidence": random.uniform(0.5, 0.9)
        }
    
    async def _execute_task(self, task_id: str):
        """Exécute une tâche de fine-tuning réelle."""
        self.logger.info(f"Début de l'exécution de la tâche de fine-tuning {task_id}")
        
        # Récupérer la tâche
        task = self.tasks_db.get(task_id)
        if not task:
            self.logger.error(f"Tâche {task_id} non trouvée")
            return
            
        # Mettre à jour le statut et l'heure de début
        task.status = FineTuningStatus.RUNNING
        task.stats = TrainingStats(
            start_time=datetime.now(),
            training_time_seconds=0,
            total_steps=0,
            total_epochs=0
        )
        
        try:
            # Préparer le répertoire de sortie
            output_dir = task.config.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Charger les données
            train_data, eval_data = await self._load_datasets(task.config)
            
            # Selon le type de modèle, appeler la méthode d'entraînement appropriée
            if task.config.model_type == ModelType.CLASSIFICATION:
                model_path, stats = await self._train_classification_model(
                    train_data, eval_data, task.config, task_id
                )
            elif task.config.model_type == ModelType.QUESTION_ANSWERING:
                model_path, stats = await self._train_qa_model(
                    train_data, eval_data, task.config, task_id
                )
            else:
                model_path, stats = await self._train_ollama_model(
                    train_data, eval_data, task.config, task_id
                )
                
            # Mettre à jour les statistiques finales
            task.stats.end_time = datetime.now()
            task.stats.training_time_seconds = (task.stats.end_time - task.stats.start_time).total_seconds()
            
            # Mettre à jour le chemin du modèle et le statut
            task.model_path = model_path
            task.status = FineTuningStatus.COMPLETED
            
            # Ajouter le modèle à la base de données des modèles
            model_id = f"{task.config.name.lower().replace(' ', '-')}-{task_id[:8]}"
            self.models_db[model_id] = {
                "id": model_id,
                "name": task.config.name,
                "description": task.config.description,
                "model_type": task.config.model_type,
                "base_model": task.config.base_model,
                "path": model_path,
                "metrics": task.stats.best_metrics if hasattr(task.stats, 'best_metrics') else {},
                "created_at": datetime.now().isoformat(),
                "created_by": task.created_by,
                "task_id": task_id,
                "tags": task.config.tags
            }
            
            self.logger.info(f"Tâche de fine-tuning {task_id} terminée avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur pendant l'exécution de la tâche {task_id}: {str(e)}")
            task.status = FineTuningStatus.FAILED
            task.error = str(e)
            
        finally:
            # Supprimer la tâche des tâches actives
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
    async def _train_classification_model(self, train_data, eval_data, config, task_id):
        """Entraîne un modèle de classification avec Transformers."""
        try:
            # Importer les bibliothèques nécessaires
            from transformers import (
                AutoModelForSequenceClassification, AutoTokenizer,
                Trainer, TrainingArguments, DataCollatorWithPadding
            )
            from datasets import Dataset
            import torch
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support
            
            # Préparer les données d'entraînement et d'évaluation
            train_dataset = Dataset.from_pandas(train_data)
            eval_dataset = Dataset.from_pandas(eval_data)
            
            # Charger le tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            
            # Fonction de prétraitement
            def preprocess_function(examples):
                return tokenizer(
                    examples["text"], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=512
                )
            
            # Tokeniser les données
            tokenized_train = train_dataset.map(preprocess_function, batched=True)
            tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
            
            # Créer le data collator
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
            
            # Déterminer le nombre de labels
            num_labels = len(set(train_data["label"]))
            
            # Charger le modèle
            model = AutoModelForSequenceClassification.from_pretrained(
                config.base_model, 
                num_labels=num_labels
            )
            
            # Fonction de calcul des métriques
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=-1)
                
                precision, recall, f1, _ = precision_recall_fscore_support(
                    labels, predictions, average='weighted'
                )
                acc = accuracy_score(labels, predictions)
                
                return {
                    "accuracy": acc,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall
                }
            
            # Configurer l'entraînement
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                learning_rate=config.hyperparameters.get("learning_rate", 2e-5),
                per_device_train_batch_size=config.hyperparameters.get("batch_size", 16),
                per_device_eval_batch_size=config.hyperparameters.get("batch_size", 16),
                num_train_epochs=config.hyperparameters.get("epochs", 3),
                weight_decay=config.hyperparameters.get("weight_decay", 0.01),
                evaluation_strategy="steps",
                eval_steps=config.evaluation.get("eval_steps", 100),
                save_strategy="steps",
                save_steps=config.evaluation.get("eval_steps", 100),
                load_best_model_at_end=config.evaluation.get("save_best_model", True),
                metric_for_best_model=config.evaluation.get("best_model_metric", "f1"),
                push_to_hub=False,
                report_to="none"
            )
            
            # Créer le trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            
            # Entraîner le modèle
            train_result = trainer.train()
            
            # Évaluer le modèle
            eval_result = trainer.evaluate()
            
            # Sauvegarder le modèle final
            trainer.save_model(f"{config.output_dir}/final")
            
            # Créer les statistiques d'entraînement
            stats = TrainingStats(
                start_time=datetime.now() - timedelta(seconds=train_result.metrics.get("train_runtime", 0)),
                end_time=datetime.now(),
                total_steps=train_result.metrics.get("step", 0),
                total_epochs=train_result.metrics.get("epoch", 0),
                training_time_seconds=train_result.metrics.get("train_runtime", 0),
                final_loss=train_result.metrics.get("train_loss", 0),
                best_metrics={
                    "accuracy": eval_result.get("eval_accuracy", 0),
                    "f1": eval_result.get("eval_f1", 0),
                    "precision": eval_result.get("eval_precision", 0),
                    "recall": eval_result.get("eval_recall", 0)
                }
            )
            
            # Mettre à jour la tâche avec les données d'entraînement
            task = self.tasks_db.get(task_id)
            if task:
                task.stats = stats
            
            return f"{config.output_dir}/final", stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement du modèle de classification: {str(e)}")
            raise
            
    async def _train_qa_model(self, train_data, eval_data, config, task_id):
        """Entraîne un modèle de question-réponse avec Transformers."""
        try:
            # Importer les bibliothèques nécessaires
            from transformers import (
                AutoModelForQuestionAnswering, AutoTokenizer,
                Trainer, TrainingArguments
            )
            from datasets import Dataset
            import torch
            import numpy as np
            
            # Préparer les données d'entraînement et d'évaluation
            train_dataset = Dataset.from_pandas(train_data)
            eval_dataset = Dataset.from_pandas(eval_data)
            
            # Charger le tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.base_model)
            
            # Fonction de prétraitement pour QA
            def preprocess_function(examples):
                questions = [q.strip() for q in examples["question"]]
                contexts = [c.strip() for c in examples["context"]]
                
                inputs = tokenizer(
                    questions,
                    contexts,
                    max_length=384,
                    truncation="only_second",
                    stride=128,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    padding="max_length"
                )
                
                # Mapper les réponses aux features
                sample_map = inputs.pop("overflow_to_sample_mapping")
                offset_mapping = inputs.pop("offset_mapping")
                
                inputs["start_positions"] = []
                inputs["end_positions"] = []
                
                for i, offset in enumerate(offset_mapping):
                    sample_idx = sample_map[i]
                    
                    # On utilise des valeurs par défaut pour les exemples sans réponse
                    inputs["start_positions"].append(0)
                    inputs["end_positions"].append(0)
                    
                    # Si l'exemple a une réponse, on trouve les positions de début et de fin
                    if examples["answer"][sample_idx]:
                        answer = examples["answer"][sample_idx]
                        context = contexts[sample_idx]
                        
                        start_char = context.find(answer)
                        if start_char != -1:
                            end_char = start_char + len(answer)
                            
                            # Trouver le token qui contient le caractère de début
                            token_start_index = 0
                            while token_start_index < len(offset) and offset[token_start_index][0] <= start_char:
                                token_start_index += 1
                            token_start_index -= 1
                            
                            # Trouver le token qui contient le caractère de fin
                            token_end_index = token_start_index
                            while token_end_index < len(offset) and offset[token_end_index][1] <= end_char:
                                token_end_index += 1
                            token_end_index -= 1
                            
                            inputs["start_positions"][-1] = token_start_index
                            inputs["end_positions"][-1] = token_end_index
                
                return inputs
            
            # Tokeniser les données
            tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
            tokenized_eval = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)
            
            # Charger le modèle
            model = AutoModelForQuestionAnswering.from_pretrained(config.base_model)
            
            # Configurer l'entraînement
            training_args = TrainingArguments(
                output_dir=config.output_dir,
                learning_rate=config.hyperparameters.get("learning_rate", 3e-5),
                per_device_train_batch_size=config.hyperparameters.get("batch_size", 8),
                per_device_eval_batch_size=config.hyperparameters.get("batch_size", 8),
                num_train_epochs=config.hyperparameters.get("epochs", 2),
                weight_decay=config.hyperparameters.get("weight_decay", 0.01),
                evaluation_strategy="steps",
                eval_steps=config.evaluation.get("eval_steps", 100),
                save_strategy="steps",
                save_steps=config.evaluation.get("eval_steps", 100),
                gradient_accumulation_steps=config.hyperparameters.get("gradient_accumulation_steps", 4),
                fp16=torch.cuda.is_available(),
                load_best_model_at_end=config.evaluation.get("save_best_model", True),
                metric_for_best_model=config.evaluation.get("best_model_metric", "eval_loss"),
                greater_is_better=False,
                push_to_hub=False,
                report_to="none"
            )
            
            # Créer le trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_eval,
                tokenizer=tokenizer,
            )
            
            # Entraîner le modèle
            train_result = trainer.train()
            
            # Évaluer le modèle
            eval_result = trainer.evaluate()
            
            # Sauvegarder le modèle final
            trainer.save_model(f"{config.output_dir}/final")
            
            # Créer les statistiques d'entraînement
            stats = TrainingStats(
                start_time=datetime.now() - timedelta(seconds=train_result.metrics.get("train_runtime", 0)),
                end_time=datetime.now(),
                total_steps=train_result.metrics.get("step", 0),
                total_epochs=train_result.metrics.get("epoch", 0),
                training_time_seconds=train_result.metrics.get("train_runtime", 0),
                final_loss=train_result.metrics.get("train_loss", 0),
                best_metrics={
                    "eval_loss": eval_result.get("eval_loss", 0)
                }
            )
            
            # Mettre à jour la tâche avec les données d'entraînement
            task = self.tasks_db.get(task_id)
            if task:
                task.stats = stats
            
            return f"{config.output_dir}/final", stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement du modèle de question-réponse: {str(e)}")
            raise
            
    async def _train_ollama_model(self, train_data, eval_data, config, task_id):
        """Fine-tune un modèle Ollama avec la création d'un Modelfile."""
        try:
            # Le répertoire de sortie pour ce modèle
            model_dir = config.output_dir
            os.makedirs(model_dir, exist_ok=True)
            
            # Création du Modelfile
            model_name = config.name.lower().replace(" ", "-")
            base_model = config.base_model
            
            # Créer le contenu du Modelfile
            modelfile_content = f"""
            FROM {base_model}

            # Métadonnées du modèle
            PARAMETER temperature {config.hyperparameters.get('temperature', 0.7)}
            PARAMETER top_p {config.hyperparameters.get('top_p', 0.9)}
            PARAMETER top_k {config.hyperparameters.get('top_k', 40)}
                
            # Description et license
            SYSTEM """
            
            # Ajouter la description du modèle au Modelfile
            modelfile_content += f"{config.description}\n\n"
            
            # Ajouter les exemples d'entraînement
            for i, row in train_data.iterrows():
                if 'question' in row and 'answer' in row:
                    modelfile_content += f"<prompt>\n{row['question']}\n</prompt>\n\n<response>\n{row['answer']}\n</response>\n\n"
                elif 'input' in row and 'output' in row:
                    modelfile_content += f"<prompt>\n{row['input']}\n</prompt>\n\n<response>\n{row['output']}\n</response>\n\n"
                elif 'text' in row and 'label' in row:
                    modelfile_content += f"<prompt>\nClassifie le texte suivant: {row['text']}\n</prompt>\n\n<response>\n{row['label']}\n</response>\n\n"
            
            # Écrire le Modelfile
            modelfile_path = os.path.join(model_dir, "Modelfile")
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)
            
            # Créer le modèle avec Ollama
            import subprocess
            model_full_name = f"{model_name}:{task_id[:8]}"
            
            # Exécuter la commande Ollama pour créer le modèle
            result = subprocess.run(
                ["ollama", "create", model_full_name, "-f", modelfile_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Erreur lors de la création du modèle Ollama: {result.stderr}")
            
            # Statistiques simulées pour Ollama (pas de métriques réelles disponibles)
            stats = TrainingStats(
                start_time=datetime.now() - timedelta(seconds=60),
                end_time=datetime.now(),
                total_steps=1,
                total_epochs=1,
                training_time_seconds=60,
                final_loss=0,
                best_metrics={}
            )
            
            return model_full_name, stats
            
        except Exception as e:
            self.logger.error(f"Erreur lors du fine-tuning du modèle Ollama: {str(e)}")
            raise
    
    async def _load_datasets(self, config):
        """Charge les datasets à partir des sources configurées."""
        import pandas as pd
        
        all_train_data = []
        all_eval_data = []
        
        for source in config.dataset_sources:
            try:
                # Charger les données selon le type et le format
                if source["type"] == "file":
                    location = source["location"]
                    format = source["format"]
                    options = source.get("options", {})
                    
                    if format == "csv":
                        df = pd.read_csv(location)
                    elif format == "json":
                        df = pd.read_json(location)
                    elif format == "excel":
                        df = pd.read_excel(location)
                    else:
                        raise ValueError(f"Format non supporté: {format}")
                    
                    # Appliquer les mappings de colonnes si nécessaire
                    if "column_mapping" in options:
                        df = df.rename(columns=options["column_mapping"])
                    
                    # Division train/eval si train_ratio est spécifié
                    train_ratio = options.get("train_ratio", 0.8)
                    
                    # Si un split est déjà défini, l'utiliser
                    if "split_column" in options:
                        split_col = options["split_column"]
                        train_data = df[df[split_col] == "train"]
                        eval_data = df[df[split_col] == "eval"]
                    else:
                        # Sinon faire un split aléatoire
                        from sklearn.model_selection import train_test_split
                        train_data, eval_data = train_test_split(df, train_size=train_ratio, random_state=42)
                    
                    all_train_data.append(train_data)
                    all_eval_data.append(eval_data)
                    
                elif source["type"] == "huggingface":
                    # Charger depuis Hugging Face Datasets
                    from datasets import load_dataset
                    
                    dataset_name = source["name"]
                    config_name = source.get("config", None)
                    split = source.get("split", "train")
                    
                    dataset = load_dataset(dataset_name, config_name, split=split)
                    
                    # Convertir en DataFrame
                    df = dataset.to_pandas()
                    
                    # Appliquer les mappings de colonnes
                    if "column_mapping" in source.get("options", {}):
                        df = df.rename(columns=source["options"]["column_mapping"])
                    
                    # Split train/eval
                    from sklearn.model_selection import train_test_split
                    train_ratio = source.get("options", {}).get("train_ratio", 0.8)
                    train_data, eval_data = train_test_split(df, train_size=train_ratio, random_state=42)
                    
                    all_train_data.append(train_data)
                    all_eval_data.append(eval_data)
                    
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement des données depuis {source['name']}: {str(e)}")
                raise
        
        # Fusionner tous les dataframes
        train_df = pd.concat(all_train_data, ignore_index=True) if all_train_data else pd.DataFrame()
        eval_df = pd.concat(all_eval_data, ignore_index=True) if all_eval_data else pd.DataFrame()
        
        return train_df, eval_df
    
    def _generate_labels_for_model(self, model_type: ModelType) -> List[str]:
        """Génère des étiquettes appropriées selon le type de modèle."""
        if model_type == ModelType.CLASSIFICATION:
            return ["positive", "negative", "neutral"]
        elif model_type == ModelType.SENTIMENT_ANALYSIS:
            return ["positive", "negative", "neutral", "mixed"]
        elif model_type == ModelType.QA:
            return []  # pas d'étiquettes fixes
        elif model_type == ModelType.SUMMARIZATION:
            return []  # pas d'étiquettes fixes
        else:
            return []
    
    def _generate_test_examples(self, model_type: ModelType) -> List[Dict[str, Any]]:
        """Génère des exemples de test selon le type de modèle."""
        if model_type == ModelType.CLASSIFICATION:
            return [
                {
                    "input": "Je suis très satisfait de ce produit!",
                    "prediction": "positive",
                    "actual": "positive",
                    "confidence": 0.92
                },
                {
                    "input": "La qualité est médiocre et le service client inexistant.",
                    "prediction": "negative",
                    "actual": "negative",
                    "confidence": 0.88
                },
                {
                    "input": "Le produit fonctionne comme prévu.",
                    "prediction": "neutral",
                    "actual": "neutral",
                    "confidence": 0.75
                }
            ]
        elif model_type == ModelType.QA:
            return [
                {
                    "input": {
                        "question": "Quand a été fondée la société?",
                        "context": "La société XYZ a été fondée en 2005 par John Smith et a connu une croissance rapide."
                    },
                    "prediction": "en 2005",
                    "actual": "en 2005",
                    "confidence": 0.89
                },
                {
                    "input": {
                        "question": "Qui a fondé la société?",
                        "context": "La société XYZ a été fondée en 2005 par John Smith et a connu une croissance rapide."
                    },
                    "prediction": "John Smith",
                    "actual": "John Smith",
                    "confidence": 0.94
                }
            ]
        else:
            return [
                {
                    "input": "Exemple d'entrée 1",
                    "prediction": "Exemple de sortie 1",
                    "actual": "Exemple de référence 1",
                    "confidence": 0.82
                }
            ] 
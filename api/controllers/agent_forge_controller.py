from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from typing import List, Optional
import logging

from api.models.agent_forge import (
    AgentForgeConfig, AgentForgeTask, AgentForgeResult, 
    AgentForgeRequest, AgentForgeResponse
)
from api.services.agent_forge_service import AgentForgeService
from api.services.auth.auth_service import get_current_user

router = APIRouter(prefix="/api/v1/agent-forge", tags=["AgentForge"])
logger = logging.getLogger(__name__)

# Injection de dépendance pour le service
def get_agent_forge_service():
    return AgentForgeService()

@router.post("/tasks", response_model=AgentForgeResponse, status_code=status.HTTP_201_CREATED)
async def create_forge_task(
    request: AgentForgeRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    agent_forge_service: AgentForgeService = Depends(get_agent_forge_service)
):
    """
    Crée une nouvelle tâche AgentForge pour générer un agent IA avancé.
    
    Cette API permet de démarrer le processus de création d'un agent avec
    l'algorithme AgentForge, en spécifiant tous les paramètres nécessaires.
    """
    try:
        user_id = current_user["id"]
        response = await agent_forge_service.create_forge_task(request, user_id)
        return response
    except Exception as e:
        logger.error(f"Erreur lors de la création de la tâche AgentForge: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la création de la tâche: {str(e)}"
        )

@router.get("/tasks", response_model=List[AgentForgeTask])
async def get_forge_tasks(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
    agent_forge_service: AgentForgeService = Depends(get_agent_forge_service)
):
    """
    Récupère toutes les tâches AgentForge de l'utilisateur actuel.
    
    Permet de filtrer par statut si spécifié.
    """
    try:
        user_id = current_user["id"]
        tasks = await agent_forge_service.get_tasks(user_id=user_id, status=status)
        return tasks
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des tâches AgentForge: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération des tâches: {str(e)}"
        )

@router.get("/tasks/{task_id}", response_model=AgentForgeTask)
async def get_forge_task(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    agent_forge_service: AgentForgeService = Depends(get_agent_forge_service)
):
    """
    Récupère les détails d'une tâche AgentForge spécifique.
    """
    try:
        task = await agent_forge_service.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tâche AgentForge {task_id} non trouvée"
            )
        
        # Vérifier que l'utilisateur a le droit d'accéder à cette tâche
        user_id = current_user["id"]
        if task.created_by != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Vous n'avez pas accès à cette tâche"
            )
            
        return task
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la tâche AgentForge {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération de la tâche: {str(e)}"
        )

@router.get("/tasks/{task_id}/result", response_model=AgentForgeResult)
async def get_forge_task_result(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    agent_forge_service: AgentForgeService = Depends(get_agent_forge_service)
):
    """
    Récupère le résultat d'une tâche AgentForge spécifique.
    """
    try:
        # Vérifier d'abord que la tâche existe et que l'utilisateur y a accès
        task = await agent_forge_service.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tâche AgentForge {task_id} non trouvée"
            )
        
        # Vérifier que l'utilisateur a le droit d'accéder à cette tâche
        user_id = current_user["id"]
        if task.created_by != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Vous n'avez pas accès à cette tâche"
            )
        
        # Récupérer le résultat
        result = await agent_forge_service.get_task_result(task_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Résultat pour la tâche {task_id} non disponible"
            )
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du résultat de la tâche {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la récupération du résultat: {str(e)}"
        )

@router.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_forge_task(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    agent_forge_service: AgentForgeService = Depends(get_agent_forge_service)
):
    """
    Supprime une tâche AgentForge et ses résultats associés.
    """
    try:
        # Vérifier d'abord que la tâche existe et que l'utilisateur y a accès
        task = await agent_forge_service.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tâche AgentForge {task_id} non trouvée"
            )
        
        # Vérifier que l'utilisateur a le droit de supprimer cette tâche
        user_id = current_user["id"]
        if task.created_by != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Vous n'avez pas la permission de supprimer cette tâche"
            )
        
        # Supprimer la tâche
        success = await agent_forge_service.delete_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Échec de la suppression de la tâche {task_id}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la suppression de la tâche {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la suppression de la tâche: {str(e)}"
        )

@router.post("/tasks/{task_id}/cancel", status_code=status.HTTP_200_OK)
async def cancel_forge_task(
    task_id: str,
    current_user: dict = Depends(get_current_user),
    agent_forge_service: AgentForgeService = Depends(get_agent_forge_service)
):
    """
    Annule une tâche AgentForge en cours d'exécution.
    """
    try:
        # Vérifier d'abord que la tâche existe et que l'utilisateur y a accès
        task = await agent_forge_service.get_task(task_id)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tâche AgentForge {task_id} non trouvée"
            )
        
        # Vérifier que l'utilisateur a le droit d'annuler cette tâche
        user_id = current_user["id"]
        if task.created_by != user_id and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Vous n'avez pas la permission d'annuler cette tâche"
            )
        
        # Vérifier que la tâche est annulable (en cours)
        if task.status not in ["pending", "running"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"La tâche {task_id} ne peut pas être annulée car son statut est {task.status}"
            )
        
        # Annuler la tâche
        success = await agent_forge_service.cancel_task(task_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Échec de l'annulation de la tâche {task_id}"
            )
            
        return {"message": f"Tâche {task_id} annulée avec succès", "status": "cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'annulation de la tâche {task_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'annulation de la tâche: {str(e)}"
        ) 
from pydantic import BaseModel
from typing import Optional


class Token(BaseModel):
    """Modèle pour le token d'accès"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # en secondes


class TokenData(BaseModel):
    """Modèle pour les données encodées dans le token"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: Optional[list] = None 
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from sqlalchemy import Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base

from db.database import Base

# Modèle SQLAlchemy pour la base de données
class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}
    
    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)
    can_use_ai_analysis = Column(Boolean, default=True)
    can_use_predictions = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)


# Modèles Pydantic pour l'API
class UserBase(BaseModel):
    """Modèle de base pour les utilisateurs"""
    username: str
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    disabled: bool = False
    can_use_ai_analysis: bool = True
    can_use_predictions: bool = True
    is_admin: bool = False


class UserCreate(UserBase):
    """Modèle pour la création d'un utilisateur"""
    password: str


class UserResponse(UserBase):
    """Modèle pour la réponse d'un utilisateur (sans mot de passe)"""
    id: Optional[str] = None

    class Config:
        from_attributes = True


class UserInDB(UserBase):
    """Modèle pour l'utilisateur en base de données"""
    id: str
    hashed_password: str

    class Config:
        from_attributes = True 
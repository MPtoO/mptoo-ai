import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import uuid
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from jose import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from api.core.config import settings
from api.models.user import User, UserCreate, UserInDB
from api.models.token import Token, TokenData
from api.db.database import Base

logger = logging.getLogger(__name__)

# Configuration pour le hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    """Vérifie si un mot de passe en clair correspond à un mot de passe haché."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Génère un hash à partir d'un mot de passe en clair."""
    return pwd_context.hash(password)

# Configuration OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Configuration de la base de données
DATABASE_URL = settings.DATABASE_URL or "sqlite:///./data/mptoo.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Créer les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

# Base de données simulée pour les utilisateurs
# À utiliser uniquement si la base de données est vide
fake_users_db = {
    "admin": {
        "id": "user_001",
        "username": "admin",
        "full_name": "Administrateur",
        "email": "admin@example.com",
        "hashed_password": get_password_hash("admin123"),  # Ne jamais utiliser des mots de passe en dur en production
        "disabled": False,
        "can_use_ai_analysis": True,
        "can_use_predictions": True,
        "is_admin": True
    },
    "user": {
        "id": "user_002",
        "username": "user",
        "full_name": "Utilisateur Standard",
        "email": "user@example.com",
        "hashed_password": get_password_hash("user123"),  # Ne jamais utiliser des mots de passe en dur en production
        "disabled": False,
        "can_use_ai_analysis": True,
        "can_use_predictions": False,
        "is_admin": False
    },
    "analyst": {
        "id": "user_003",
        "username": "analyst",
        "full_name": "Analyste de Données",
        "email": "analyst@example.com",
        "hashed_password": get_password_hash("analyst123"),
        "disabled": False,
        "can_use_ai_analysis": True,
        "can_use_predictions": True,
        "is_admin": False
    },
    "viewer": {
        "id": "user_004",
        "username": "viewer",
        "full_name": "Utilisateur en Lecture Seule",
        "email": "viewer@example.com",
        "hashed_password": get_password_hash("viewer123"),
        "disabled": False,
        "can_use_ai_analysis": False,
        "can_use_predictions": False,
        "is_admin": False
    },
    "tester": {
        "id": "user_005",
        "username": "tester",
        "full_name": "Utilisateur de Test",
        "email": "test@example.com",
        "hashed_password": get_password_hash("test123"),
        "disabled": False,
        "can_use_ai_analysis": True,
        "can_use_predictions": True,
        "is_admin": False
    }
}


class AuthService:
    """Service pour l'authentification et la gestion des utilisateurs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.users_db = fake_users_db  # Conserver cette référence pour la compatibilité
        # Mode développement - permet de désactiver certaines vérifications
        self.dev_mode = settings.ENVIRONMENT == "development"
    
    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Authentifie un utilisateur par nom d'utilisateur et mot de passe.
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe en clair
            
        Returns:
            User si authentification réussie, None sinon
        """
        # Essayons d'abord de trouver l'utilisateur dans la base de données
        db = SessionLocal()
        try:
            user_db = db.query(User).filter(User.username == username).first()
            
            if user_db:
                # En mode développement, on peut ignorer la vérification du mot de passe
                # avec un mot de passe spécial "dev_bypass_123"
                if self.dev_mode and password == "dev_bypass_123":
                    self.logger.warning(f"DEV MODE: Authentication bypass utilisé pour {username}")
                    return user_db
                    
                if verify_password(password, user_db.hashed_password):
                    return User(
                        id=user_db.id,
                        username=user_db.username,
                        email=user_db.email,
                        full_name=user_db.full_name,
                        disabled=user_db.disabled,
                        can_use_ai_analysis=user_db.can_use_ai_analysis,
                        can_use_predictions=user_db.can_use_predictions,
                        is_admin=user_db.is_admin
                    )
            else:
                # Si l'utilisateur n'est pas trouvé dans la DB, vérifier dans fake_users_db
                user = self.get_user(username)
                if user and (self.dev_mode and password == "dev_bypass_123" or verify_password(password, user["hashed_password"])):
                    return User(**{k: v for k, v in user.items() if k != "hashed_password"})
            
            return None
        finally:
            db.close()
    
    async def create_access_token(self, data: Dict, expires_delta: Optional[timedelta] = None) -> Token:
        """
        Crée un token d'accès JWT.
        
        Args:
            data: Données à encoder dans le token
            expires_delta: Durée de validité du token
            
        Returns:
            Token d'accès
        """
        to_encode = data.copy()
        
        # Pour le développement, utiliser une durée plus longue par défaut
        if self.dev_mode:
            default_expire_minutes = 60 * 24 * 30  # 30 jours
        else:
            default_expire_minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=default_expire_minutes)
            
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        
        # Calculer le temps d'expiration en secondes
        expires_in = (expire - datetime.utcnow()).total_seconds()
        
        return Token(
            access_token=encoded_jwt,
            token_type="bearer",
            expires_in=int(expires_in)
        )
    
    async def get_current_user(self, token: Optional[str] = Depends(oauth2_scheme)) -> User:
        """
        Récupère l'utilisateur courant à partir du token.
        
        Args:
            token: Token JWT
            
        Returns:
            User correspondant au token
            
        Raises:
            HTTPException: Si le token est invalide
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        # Pour simplifier le développement, on accepte un utilisateur anonyme
        if token is None:
            return User(
                id="anonymous",
                username="anonymous",
                disabled=False,
                can_use_ai_analysis=True,  # Accès à l'analyse de base
                can_use_predictions=False  # Pas d'accès aux prédictions
            )
        
        try:
            # En mode développement, on peut accepter un token simplifié "dev_token_admin"
            if self.dev_mode and token in ["dev_token_admin", "dev_token_user", "dev_token_analyst"]:
                self.logger.warning(f"DEV MODE: Token simplifié utilisé - {token}")
                
                if token == "dev_token_admin":
                    user_data = self.users_db["admin"]
                elif token == "dev_token_analyst":
                    user_data = self.users_db["analyst"]
                else:
                    user_data = self.users_db["user"]
                    
                return User(**{k: v for k, v in user_data.items() if k != "hashed_password"})
            
            # Vérification normale du token
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.JWT_ALGORITHM]
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username)
        except Exception as e:
            self.logger.error(f"Erreur lors du décodage du token: {str(e)}")
            raise credentials_exception
        
        # Recherche de l'utilisateur d'abord dans la base de données puis dans fake_users_db
        db = SessionLocal()
        try:
            user_db = db.query(User).filter(User.username == token_data.username).first()
            if user_db:
                if user_db.disabled:
                    raise HTTPException(status_code=400, detail="Utilisateur désactivé")
                return user_db
        finally:
            db.close()
            
        # Si non trouvé dans la base, chercher dans fake_users_db
        user = self.get_user(token_data.username)
        if user is None:
            raise credentials_exception
            
        if user["disabled"]:
            raise HTTPException(status_code=400, detail="Utilisateur désactivé")
            
        return User(**{k: v for k, v in user.items() if k != "hashed_password"})
    
    def get_user(self, username: str) -> Optional[Dict]:
        """
        Récupère un utilisateur par son nom d'utilisateur depuis fake_users_db.
        
        Args:
            username: Nom d'utilisateur
            
        Returns:
            Dictionnaire des données utilisateur ou None si non trouvé
        """
        if username in self.users_db:
            return self.users_db[username]
        return None
    
    async def register_user(self, user_create: UserCreate) -> User:
        """
        Enregistre un nouvel utilisateur.
        
        Args:
            user_create: Données de création de l'utilisateur
            
        Returns:
            User créé
            
        Raises:
            HTTPException: Si le nom d'utilisateur existe déjà
        """
        db = SessionLocal()
        try:
            # Vérifier si l'utilisateur existe déjà
            existing_user = db.query(User).filter(User.username == user_create.username).first()
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Nom d'utilisateur déjà utilisé"
                )
            
            # Générer un ID unique pour l'utilisateur
            user_id = str(uuid.uuid4())
            
            # Créer l'utilisateur dans la base de données
            new_user = User(
                id=user_id,
                username=user_create.username,
                email=user_create.email,
                full_name=user_create.full_name,
                hashed_password=get_password_hash(user_create.password),
                disabled=user_create.disabled,
                can_use_ai_analysis=user_create.can_use_ai_analysis,
                can_use_predictions=user_create.can_use_predictions,
                is_admin=user_create.is_admin
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            return User(
                id=new_user.id,
                username=new_user.username,
                email=new_user.email,
                full_name=new_user.full_name,
                disabled=new_user.disabled,
                can_use_ai_analysis=new_user.can_use_ai_analysis,
                can_use_predictions=new_user.can_use_predictions,
                is_admin=new_user.is_admin
            )
        finally:
            db.close()
        
    # Méthode utilitaire pour le développement
    def get_all_users(self) -> Dict[str, User]:
        """
        Récupère tous les utilisateurs (utile pour le développement).
        Disponible uniquement en mode développement.
        
        Returns:
            Dictionnaire des utilisateurs (sans les mots de passe)
        """
        if not self.dev_mode:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cette fonctionnalité est disponible uniquement en mode développement"
            )
            
        return {
            username: User(**{k: v for k, v in user_data.items() if k != "hashed_password"})
            for username, user_data in self.users_db.items()
        }

    def authenticate_user_sync(self, username, password):
        # Version synchrone de authenticate_user
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.authenticate_user(username, password))
        loop.close()
        return result
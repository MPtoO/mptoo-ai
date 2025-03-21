import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext

# Modifions les imports pour qu'ils soient relatifs au répertoire courant
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importez vos modèles
from models.user import User
from core.config import settings
from db.database import Base

# Création des répertoires nécessaires
os.makedirs(os.path.join(current_dir, 'data'), exist_ok=True)

# Configuration de la base de données
DATABASE_URL = os.environ.get("DATABASE_URL") or "sqlite:///./data/mptoo.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Créer les tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

# Configuration du hashage de mot de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_admin():
    db = SessionLocal()
    
    # Vérifier si un administrateur existe déjà
    admin = db.query(User).filter(User.is_admin == True).first()
    if admin:
        print(f"Un administrateur existe déjà: {admin.username}")
        return
    
    # Créer un nouvel administrateur
    admin_user = User(
        username="admin",
        email="admin@mptoo.org",
        full_name="Administrateur Système",
        hashed_password=pwd_context.hash("Admin123!"),
        is_admin=True,
        can_use_ai_analysis=True,
        can_use_predictions=True,
        disabled=False
    )
    
    db.add(admin_user)
    db.commit()
    print(f"Administrateur créé avec succès: username=admin, password=Admin123!")

if __name__ == "__main__":
    create_admin()

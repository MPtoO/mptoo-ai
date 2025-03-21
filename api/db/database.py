from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from pathlib import Path
import os

# Assurez-vous que le répertoire data existe
data_dir = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
data_dir.mkdir(exist_ok=True)

# Configuration de la base de données
DATABASE_URL = os.environ.get("DATABASE_URL") or "sqlite:///./data/mptoo.db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Méthode pour obtenir une session de DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 
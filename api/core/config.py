import os
import json
from typing import List

class Settings:
    PROJECT_NAME = "MPToO API"
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev_secret_key_change_in_production")
    JWT_ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")
    DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./data/mptoo.db")
    DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
    PORT = int(os.environ.get("PORT", "8001"))
    
    # CORS
    _cors_origins = os.environ.get("BACKEND_CORS_ORIGINS", 
        '["http://localhost:5173", "http://localhost:8000", "http://localhost:3000", "http://localhost:8080"]')
    BACKEND_CORS_ORIGINS = json.loads(_cors_origins.replace("'", '"'))
    
    # API
    API_V1_STR = "/api"
    
    # ML
    ML_MODEL_PATH = "models"
    DEFAULT_AI_MODEL = "gpt-4"
    MAX_SEARCH_RESULTS = 100
    
    # Files
    UPLOAD_FOLDER = "uploads"
    ALLOWED_EXTENSIONS = ["pdf", "doc", "docx", "txt", "csv", "xls", "xlsx"]
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    # Crawling
    MAX_CRAWL_DEPTH = 3
    MAX_CRAWL_PAGES = 100
    CRAWL_DELAY = 2
    RESPECT_ROBOTS_TXT = True

settings = Settings() 
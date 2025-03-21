from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime
import uvicorn

from api.config import CORS_ORIGINS, DEBUG, ENVIRONMENT
from api.controllers.agent_forge_controller import router as agent_forge_router

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = FastAPI(
        title="MPTOO API v2",
        description="API pour les fonctionnalités avancées de MPTOO",
        version="2.0.0"
    )
    
    # Configuration CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Enregistrement des routeurs
    app.include_router(agent_forge_router)
    
    # Middleware pour le logging
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = datetime.now()
        response = await call_next(request)
        process_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.debug(f"{request.method} {request.url.path} completed in {process_time:.2f}ms")
        return response
    
    # Route de base
    @app.get("/")
    async def root():
        return {
            "message": "API FastAPI MPTOO en cours d'exécution",
            "version": "2.0.0",
            "timestamp": datetime.now().isoformat()
        }
    
    # Route de statut
    @app.get("/status")
    async def get_status():
        return {
            "status": "active",
            "environment": ENVIRONMENT,
            "debug": DEBUG,
            "timestamp": datetime.now().isoformat()
        }
    
    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) 
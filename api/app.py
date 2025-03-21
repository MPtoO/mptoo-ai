#!/usr/bin/env python3
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
from datetime import datetime

from api.config import CORS_ORIGINS, DEBUG, ENVIRONMENT
from api.controllers.auth_controller import auth_bp
from api.controllers.llm_controller import llm_bp
from api.controllers.analytics_controller import analytics_bp
from api.controllers.ml_controller import ml_bp

# Configuration du logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    app = Flask(__name__)
    
    # Configuration CORS
    CORS(app, resources={r"/*": {"origins": CORS_ORIGINS}})
    
    # Configuration de l'application
    app.config["DEBUG"] = DEBUG
    app.config["ENV"] = ENVIRONMENT
    
    # Enregistrement des blueprints
    app.register_blueprint(auth_bp, url_prefix="/auth")
    app.register_blueprint(llm_bp, url_prefix="/api/llm")
    app.register_blueprint(analytics_bp, url_prefix="/api/analytics")
    app.register_blueprint(ml_bp, url_prefix="/api/ml")
    
    # Middleware pour détecter le mode d'interface
    @app.before_request
    def detect_interface_mode():
        if "interface" in request.args:
            app.config["INTERFACE_MODE"] = request.args["interface"]
    
    # Route de base
    @app.route("/", methods=["GET"])
    def root():
        return jsonify({
            "message": "API MPTOO en cours d'exécution",
            "version": "1.1.0",
            "timestamp": datetime.now().isoformat()
        })
    
    # Route de statut
    @app.route("/status", methods=["GET"])
    def get_status():
        return jsonify({
            "status": "active",
            "environment": app.config["ENV"],
            "debug": app.config["DEBUG"],
            "timestamp": datetime.now().isoformat()
        })
    
    # Gestionnaire d'erreurs globales
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"detail": "Ressource non trouvée"}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        logger.error(f"Erreur serveur: {str(e)}")
        return jsonify({"detail": "Erreur serveur interne"}), 500
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

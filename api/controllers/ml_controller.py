#!/usr/bin/env python3
from flask import Blueprint, request, jsonify
from api.controllers.auth_controller import require_auth
from api.services.ml.model_service import ModelService

ml_bp = Blueprint('ml', __name__)
model_service = ModelService()

@ml_bp.route("/models", methods=["GET"])
@require_auth
def list_models(current_user):
    try:
        models = model_service.get_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@ml_bp.route("/train", methods=["POST"])
@require_auth
def train_model(current_user):
    data = request.json
    model_name = data.get("model_name")
    training_data = data.get("data")
    params = data.get("params", {})
    
    if not model_name or not training_data:
        return jsonify({"detail": "Nom du modèle et données d'entraînement requis"}), 400
    
    try:
        job_id = model_service.train_model(model_name, training_data, params)
        return jsonify({"job_id": job_id, "status": "training"})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@ml_bp.route("/predict", methods=["POST"])
@require_auth
def predict(current_user):
    data = request.json
    model_name = data.get("model_name")
    input_data = data.get("data")
    
    if not model_name or not input_data:
        return jsonify({"detail": "Nom du modèle et données d'entrée requis"}), 400
    
    try:
        predictions = model_service.predict(model_name, input_data)
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@ml_bp.route("/jobs/<job_id>", methods=["GET"])
@require_auth
def get_job_status(current_user, job_id):
    try:
        status = model_service.get_job_status(job_id)
        return jsonify(status)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

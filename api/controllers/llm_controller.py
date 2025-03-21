#!/usr/bin/env python3
from flask import Blueprint, request, jsonify
from api.controllers.auth_controller import require_auth
from api.services.ollama_service import OllamaService

llm_bp = Blueprint('llm', __name__)
ollama_service = OllamaService()

@llm_bp.route("/models", methods=["GET"])
@require_auth
def list_ollama_models(current_user):
    try:
        models = ollama_service.list_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@llm_bp.route("/generate", methods=["POST"])
@require_auth
def generate_with_ollama(current_user):
    data = request.json
    prompt = data.get("prompt")
    model = data.get("model", "llama3")
    
    if not prompt:
        return jsonify({"detail": "Un prompt est requis"}), 400
    
    try:
        response = ollama_service.generate(model, prompt)
        return jsonify(response)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@llm_bp.route("/chat", methods=["POST"])
@require_auth
def chat_with_ollama(current_user):
    data = request.json
    messages = data.get("messages", [])
    model = data.get("model", "llama3")
    
    if not messages:
        return jsonify({"detail": "Des messages sont requis"}), 400
    
    try:
        response = ollama_service.chat(model, messages)
        return jsonify(response)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500 
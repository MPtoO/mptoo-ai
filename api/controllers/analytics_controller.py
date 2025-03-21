#!/usr/bin/env python3
from flask import Blueprint, request, jsonify
from api.controllers.auth_controller import require_auth
from api.services.analytics.service import AnalyticsService

analytics_bp = Blueprint('analytics', __name__)
analytics_service = AnalyticsService()

@analytics_bp.route("/analyze", methods=["POST"])
@require_auth
def analyze_data(current_user):
    data = request.json.get("data")
    metrics = request.json.get("metrics")
    
    if not data:
        return jsonify({"detail": "Données requises pour l'analyse"}), 400
    
    try:
        results = analytics_service.analyze_dataset(data, metrics)
        return jsonify(results)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

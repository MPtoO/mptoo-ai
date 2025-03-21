#!/usr/bin/env python3
from flask import Blueprint, request, jsonify, g
from api.services.auth_service import AuthService
from api.models.user import UserCreate, UserInDB
from functools import wraps

auth_bp = Blueprint('auth', __name__)
auth_service = AuthService()

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"detail": "Authentification requise"}), 401
        
        token = auth_header.split(' ')[1]
        try:
            current_user = auth_service.get_current_user(token)
            g.current_user = current_user
            return f(current_user, *args, **kwargs)
        except Exception as e:
            return jsonify({"detail": str(e)}), 401
    
    return decorated

@auth_bp.route("/token", methods=["POST"])
def login_for_access_token():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"detail": "Nom d'utilisateur et mot de passe requis"}), 400
    
    try:
        token = auth_service.authenticate_user(username, password)
        return jsonify({"access_token": token, "token_type": "bearer"})
    except Exception as e:
        return jsonify({"detail": str(e)}), 401

@auth_bp.route("/users/", methods=["POST"])
def register_user():
    data = request.json
    user = UserCreate(**data)
    try:
        created_user = auth_service.create_user(user)
        return jsonify(created_user.dict(exclude={"hashed_password"})), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 400

@auth_bp.route("/admin/setup", methods=["POST"])
def setup_first_admin():
    data = request.json
    token = data.get("setup_token")
    admin_data = data.get("admin")
    
    if not token or token != auth_service.get_setup_token():
        return jsonify({"detail": "Token de configuration invalide"}), 401
    
    try:
        admin_user = UserCreate(**admin_data)
        created_admin = auth_service.create_admin(admin_user)
        return jsonify(created_admin.dict(exclude={"hashed_password"})), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 400

@auth_bp.route("/admin/create", methods=["POST"])
@require_auth
def create_admin_user(current_user):
    if not current_user.is_admin:
        return jsonify({"detail": "Privilèges d'administration requis"}), 403
    
    data = request.json
    try:
        admin_user = UserCreate(**data)
        created_admin = auth_service.create_admin(admin_user)
        return jsonify(created_admin.dict(exclude={"hashed_password"})), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 400

@auth_bp.route("/users/me", methods=["GET"])
@require_auth
def read_users_me(current_user):
    return jsonify(current_user.dict(exclude={"hashed_password"})) 
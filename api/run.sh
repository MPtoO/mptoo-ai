#!/bin/bash

# Configuration
FLASK_PORT=8000
FASTAPI_PORT=8001
LOG_DIR="./logs"

# Créer le répertoire de logs s'il n'existe pas
mkdir -p "$LOG_DIR"

# Fonction pour arrêter proprement tous les processus en arrière-plan
cleanup() {
    echo "Arrêt des serveurs..."
    kill $FLASK_PID 2>/dev/null
    kill $FASTAPI_PID 2>/dev/null
    exit 0
}

# Capturer SIGINT et SIGTERM pour l'arrêt propre
trap cleanup SIGINT SIGTERM

# Démarrer le serveur Flask
echo "Démarrage du serveur Flask sur le port $FLASK_PORT..."
gunicorn -w 4 -b "0.0.0.0:$FLASK_PORT" "api.wsgi:app" > "$LOG_DIR/flask.log" 2>&1 &
FLASK_PID=$!

# Vérifier que le serveur Flask a démarré correctement
sleep 2
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "Échec du démarrage du serveur Flask. Vérifiez les logs pour plus de détails."
    cat "$LOG_DIR/flask.log"
    exit 1
fi
echo "Serveur Flask démarré avec PID $FLASK_PID"

# Démarrer le serveur FastAPI
echo "Démarrage du serveur FastAPI sur le port $FASTAPI_PORT..."
uvicorn api.app_fastapi:app --host 0.0.0.0 --port $FASTAPI_PORT > "$LOG_DIR/fastapi.log" 2>&1 &
FASTAPI_PID=$!

# Vérifier que le serveur FastAPI a démarré correctement
sleep 2
if ! kill -0 $FASTAPI_PID 2>/dev/null; then
    echo "Échec du démarrage du serveur FastAPI. Vérifiez les logs pour plus de détails."
    cat "$LOG_DIR/fastapi.log"
    kill $FLASK_PID
    exit 1
fi
echo "Serveur FastAPI démarré avec PID $FASTAPI_PID"

echo "Les deux serveurs sont en cours d'exécution."
echo "- Serveur Flask    : http://localhost:$FLASK_PORT"
echo "- Serveur FastAPI  : http://localhost:$FASTAPI_PORT"
echo "- Documentation API: http://localhost:$FASTAPI_PORT/docs"

# Attendre que l'un des processus se termine
wait $FLASK_PID $FASTAPI_PID

# Si l'un des processus se termine, arrêter l'autre aussi
cleanup 
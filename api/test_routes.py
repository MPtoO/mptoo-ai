from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/agent-forge/test', methods=['GET'])
def test_agent_forge():
    return jsonify({
        "status": "success",
        "message": "La route agent-forge est accessible"
    })

@app.route('/api/agent-ai/test', methods=['GET'])
def test_agent_ai():
    return jsonify({
        "status": "success",
        "message": "La route agent-ai est accessible"
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    return jsonify({
        "status": "success",
        "message": "L'API fonctionne correctement"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

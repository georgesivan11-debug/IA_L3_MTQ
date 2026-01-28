from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle et le scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return """
    <h1>API de Classification des Iris 🌸</h1>
    <p>Utilisez l'endpoint <code>/predict</code> pour faire des prédictions.</p>
    <h2>Exemple de requête POST:</h2>
    <pre>
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    </pre>
    """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        # Extraire les caractéristiques
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Normaliser les données
        features_scaled = scaler.transform(features)
        
        # Faire la prédiction
        prediction = model.predict(features_scaled)[0]
        
        # Obtenir les probabilités si disponible
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            prob_dict = {
                'setosa': float(probabilities[0]),
                'versicolor': float(probabilities[1]),
                'virginica': float(probabilities[2])
            }
        else:
            prob_dict = None
        
        # Retourner la réponse
        response = {
            'prediction': prediction,
            'probabilities': prob_dict,
            'input': data,
            'status': 'success'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

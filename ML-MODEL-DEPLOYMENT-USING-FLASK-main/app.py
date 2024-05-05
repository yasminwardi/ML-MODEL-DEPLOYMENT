from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib

# Créez l'application Flask
app = Flask(__name__)

# Chargez le modèle à partir du fichier joblib
model = joblib.load("linear_model.pkl")

# Route pour l'index (page d'accueil)
@app.route("/")
def home():
    return render_template("index.html")

# Route pour la prédiction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données du formulaire
        float_features = [float(x) for x in request.form.values()]
        # Convertir les données en array numpy
        features = np.array(float_features).reshape(1, -1)
        
        # Faire la prédiction
        prediction = model.predict(features)[0]
        
        # Déterminer le statut de conformité
        if prediction >= 80:
            conformity_status = "conforme"
        elif 75 <= prediction < 80:
            conformity_status = "acceptable"
        else:
            conformity_status = "non conforme"
        
        # Renvoie le résultat de la prédiction et le statut de conformité
        return render_template(
            "index.html",
            prediction_text=f"La prédiction est: {prediction}",
            conformity_status=conformity_status
        )
    
    except Exception as e:
        # Gérer les exceptions et renvoyer une erreur
        return jsonify({"error": str(e)}), 500

# Démarrez l'application Flask
if __name__ == "__main__":
    app.run(debug=True)

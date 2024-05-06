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
def classer_criteres(prediction):
    if prediction > 80:
        return "Conforme"
    elif 70 <= prediction <= 80:
        return "Acceptable"
    else:
        return "Non conforme"

@app.route('/predict', methods=['POST'])
def predict():
    # Extraire les données du formulaire
    data = request.form
    humidite = float(data['Humidite'])
    proteine = float(data['Proteine'])
    durete = float(data['Durete'])
    aw = float(data['Aw'])
    fine = float(data['Fine'])
    cendre = float(data['Cendre'])
    fibre = float(data['Fibre'])
    amidon = float(data['Amidon'])

    # Créer l'input pour la prédiction
    input_data = [[humidite, proteine, durete, aw, fine, cendre, fibre, amidon]]

    # Faire la prédiction
    prediction = model.predict(input_data)[0]  # obtenir la première prédiction

    # Classer la prédiction selon les critères d'acceptation
    prediction_class = classer_criteres(prediction)

    # Formatage de la prédiction et de la classe d'acceptation
    prediction_text = f"La prédiction est de {prediction:.2f}, ce qui est {prediction_class}"

    # Rendre la page index.html avec le texte de la prédiction
    return render_template('index.html', prediction_text=prediction_text)

# Lancer l'application Flask
if __name__ == "__main__":
    app.run(debug=True)
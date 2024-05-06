import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import create_engine
import statsmodels.api as sm

# Database credentials
dbname = 'lastberasmiNchalah'
user = ''  # Remplacez par le nom d'utilisateur correct
password = '********'  # Remplacez par le mot de passe correct
host = 'localhost'
port = '55122'
driver = 'ODBC Driver 17 for SQL Server'  # Assurez-vous que ce pilote est installé
# Connection string format for SQLAlchemy with pyodbc
connection_string = f'mssql+pyodbc://{user}:{password}@{host}:{port}/{dbname}?driver={driver}'

# Create an SQLAlchemy engine
engine = create_engine(connection_string)

# SQL query to fetch specific data from FactAlco and DimProduitAlco
query = """

SELECT [Humidite], [Proteine], [Durabilite], [Aw], [Durete], [Fine], [Cendre], [Fibre], [Amidon]
FROM [FactAlco]

"""

# Execute the query and load data into a DataFrame
# Load data into a DataFrame
df = pd.read_sql(query, engine)

# Display the DataFrame to confirm successful data load
print(df)
# Convertir toutes les colonnes en float, gérer les erreurs de conversion
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Supprimer les lignes avec des valeurs manquantes
df = df.dropna()

# Préparer les données pour la régression
X = df.drop('Durabilite', axis=1)  # Toutes les colonnes sauf 'Durete'
y = df['Durabilite']  # Colonne 'Durete'

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Utiliser Statsmodels pour la régression
X_train_sm = sm.add_constant(X_train)  # Ajouter une constante pour l'intercept
model_sm = sm.OLS(y_train, X_train_sm)
results_sm = model_sm.fit()
print("Résultat avec Statsmodels:")
print(results_sm.summary())

# Utiliser Scikit-learn pour la régression
model_sk = LinearRegression()
model_sk.fit(X_train, y_train)
y_pred = model_sk.predict(X_test)

# Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nRésultat avec Scikit-learn:")
print(f"Erreur quadratique moyenne (MSE): {mse}")
print(f"Coefficient de détermination (R²): {r2}")
import matplotlib.pyplot as plt

# Code existant ici...

# Visualisation des prédictions vs les vraies valeurs
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)  # ligne de parfaite correspondance
plt.xlabel('Valeurs Réelles')
plt.ylabel('Prédictions')
plt.title('Valeurs Réelles vs Prédictions')
plt.show()

# Visualisation des résidus
residus = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residus, alpha=0.5)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Prédictions')
plt.ylabel('Résidus')
plt.title('Résidus de la Prédiction')
plt.show()


import joblib
from flask import Flask, request,render_template , jsonify


# Enregistrer le modèle dans un fichier nommé 'model.pkl'
joblib.dump(model_sk, 'linear_model.pkl')

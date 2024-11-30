# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:18:34 2024

@author: HP ELITEBOOK
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Titre de l'application
st.title("Détection de Faux Billets à partir d'un Fichier")

# Charger le modèle et le scaler
@st.cache_resource
def charger_model_et_scaler():
    model = joblib.load(open("classification_model.pkl", "rb"))
    scaler = joblib.load(open("model_kmeans.joblib", "rb"))
    return model, scaler

# rappel du model
model, scaler = charger_model_et_scaler()

#charger le fichier csv
uploaded_file = st.file_uploader("charger un fichier csv contenant les données")

if uploaded_file:
    data = pd.read_csv(upload_file)
    st.write("Apercu des données chargées:")
    st.dataframe(data, use_container_width=True)
    
    #vérification que les colonnes nécessaires sont présentes
    colonnes_attendues = ["diagonal", "heignt_left", "height_right", "margin_low", "margin_up", "length"]
    if not all(col in data.columns for col in colonnes_attendues):
        st.error(f"le fichier doit contenir les colonnes suivantes : {{, '.join(colonnes_attendues)}}")
    else:
        X = data[colonnes_attendues]
        X_scaled = scaler.transform(X)
        with st.spinner("predictions..."):
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            
        #créer un dataframe pour les résultats
        résultats_df = pd.DatFrame({
            "prediction": prédictions,
            "probabilité Faux": probabilities[:, 0],
            "probabilité Vrai": probabilities[:, 1]
            })
        
        # affichez les résultats a deux chiffres apres la virgules
        résultats_df["probabilité Faux"]= resultats_df["probabilité Faux"].map(lambda x: f"#{x:.2f}")
        résultats_df["probabilité Vrai"]= resultats_df["probabilité Vrai"].map(lambda x: f"#{x:.2f}")
        
    
        #Afficher les résultats*
        st.write("résultats des prédictions :")
        st.dataframe(results_df, use_container_width= True)
        
        # histogramme
        prediction_counts = resultats_df["prediction"].replace({0: "faux billet", 1: "vrai billet"}).values_counts().re
        prediction_counts.columns = ["Type de billet", "Nombre"]
        
        fig = px.bar(prediction_counts,
                     x="type de billet", y = "nombre")
        
    
    
    
    
    
    
    
    
    
    
    
    
    


import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialisation de l'API (va chercher kaggle.json automatiquement)
api = KaggleApi()
api.authenticate()

# Nom du dataset (slug pris de l’URL Kaggle)
dataset = "blastchar/telco-customer-churn"

# Destination de téléchargement (ton projet)
destination_dir = r"C:\Users\Эли Жоэль\PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING\infrastructure\data_lake\raw_data"

# Téléchargement et décompression
api.dataset_download_files(dataset, path=destination_dir, unzip=True)

print(f"Dataset téléchargé dans : {destination_dir}")
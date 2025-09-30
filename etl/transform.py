import os
import pandas as pd

class DataTransformer:
    def __init__(self, raw_data_dir, transformed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.transformed_data_dir = transformed_data_dir
        os.makedirs(self.transformed_data_dir, exist_ok=True)

    def transform_file(self, file_name):
        raw_file_path = os.path.join(self.raw_data_dir, file_name)
        transformed_file_path = os.path.join(self.transformed_data_dir, file_name)

        # Charger les données
        df = pd.read_csv(raw_file_path)

        # Suppression doublons
        df = df.drop_duplicates()

        # Gestion des données manquantes
        df = df.dropna()

        # Correction des types
        for col in df.columns:
            # Essayer de convertir en numérique quand c’est pertinent
            df[col] = pd.to_numeric(df[col], errors="ignore")
            # Essayer de convertir en datetime si applicable
            if df[col].dtype == object:
                try:
                    df[col] = pd.to_datetime(df[col], errors="ignore")
                except Exception:
                    pass

        # Sauvegarde
        df.to_csv(transformed_file_path, index=False)
        print(f"Fichier transformé sauvegardé : {transformed_file_path}")

    def transform_all(self):
        for file in os.listdir(self.raw_data_dir):
            if file.endswith(".csv"):
                self.transform_file(file)


# Variables de chemin
raw_data_dir = r"C:\Users\Эли Жоэль\PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING\infrastructure\data_lake\raw_data"
transformed_data_dir = r"C:\Users\Эли Жоэль\PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING\infrastructure\data_lake\transformed_data"

transformer = DataTransformer(raw_data_dir, transformed_data_dir)
transformer.transform_all()

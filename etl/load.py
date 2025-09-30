import os
import shutil

class DataLoader:
    def __init__(self, transformed_data_dir, warehouse_dir):
        self.transformed_data_dir = transformed_data_dir
        self.warehouse_dir = warehouse_dir
        os.makedirs(self.warehouse_dir, exist_ok=True)

    def load_file(self, file_name):
        source_file = os.path.join(self.transformed_data_dir, file_name)
        target_file = os.path.join(self.warehouse_dir, file_name)

        # Copier le fichier vers le data warehouse
        shutil.copy2(source_file, target_file)
        print(f"Fichier chargé dans le data warehouse : {target_file}")

    def load_all(self):
        for file in os.listdir(self.transformed_data_dir):
            if file.endswith(".csv"):
                self.load_file(file)


# Variables de chemin
transformed_data_dir = r"C:\Users\Эли Жоэль\PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING\infrastructure\data_lake\transformed_data"
warehouse_dir = r"C:\Users\Эли Жоэль\PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING\infrastructure\data_warehouse\BI_data"

loader = DataLoader(transformed_data_dir, warehouse_dir)
loader.load_all()

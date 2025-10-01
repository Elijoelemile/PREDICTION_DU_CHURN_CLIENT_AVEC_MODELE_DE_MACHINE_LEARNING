import os
import runpy

class Orchestrator:
    def __init__(self, etl_dir):
        self.etl_dir = etl_dir

    def run_script(self, script_name):
        script_path = os.path.join(self.etl_dir, script_name)
        if os.path.exists(script_path):
            print(f"Lancement de {script_name}")
            runpy.run_path(script_path)
            print(f"{script_name} terminé \n")
        else:
            print(f"Script {script_name} introuvable dans {self.etl_dir}")

    def run_all(self):
        """
        Exécute tous les scripts ETL dans l’ordre Extraction -> Transformation -> Loading
        """
        # Dans cet ordre précis
        scripts_order = [
            "extract.py",
            "transform.py",
            "load.py"
        ]

        for script in scripts_order:
            self.run_script(script)


if __name__ == "__main__":
    etl_dir = r"C:\Users\Эли Жоэль\PREDICTION_DU_CHURN_CLIENT_AVEC_MODELE_DE_MACHINE_LEARNING\etl"
    orchestrator = Orchestrator(etl_dir)
    orchestrator.run_all()

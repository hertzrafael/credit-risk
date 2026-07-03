from etl.extract import Extract
from etl.transform import Transform
from train.model_train import ModelTrain
from pathlib import Path

import mlflow
import dotenv
import os


def _setup_dir_():
    TEMP_DIR = Path(os.getenv("TEMP_DIR"))
    TEMP_DIR.mkdir(exist_ok=True)
    
    ARTIFACTS = TEMP_DIR / "mlartifacts"
    ARTIFACTS.mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri(f"sqlite:///{TEMP_DIR}/mlflow.db")
    experiment = mlflow.get_experiment_by_name("Credit Risk")

    if experiment is None:
        mlflow.create_experiment(
            "Credit Risk",
            artifact_location=ARTIFACTS.resolve().as_uri()
        )

    mlflow.set_experiment("Credit Risk")


def main():
    dotenv.load_dotenv()
    _setup_dir_()
    
    extracted_content = Extract().run()
    transformed_content = Transform(raw_content=extracted_content).run()
    
    model_train = ModelTrain(data=transformed_content)
    model_train.run()


if __name__ == "__main__":
    main()

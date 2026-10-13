import mlflow

mlflow.set_experiment("init")

with mlflow.start_run() :
    mlflow.log_params()
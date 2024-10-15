import mlflow
import os

def setup_mlflow(experiment_name, tracking_uri=None):
    """
    Setup MLflow experiment and tracking URI.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)


def log_metrics(metrics):
    """
    Log key-value pairs as metrics in MLflow.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_artifact(filepath):
    """
    Log a file (artifact) to MLflow.
    """
    mlflow.log_artifact(filepath)


def log_params(params):
    """
    Log hyperparameters as parameters in MLflow.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)


def start_run(run_name=None):
    """
    Start an MLflow run.
    """
    return mlflow.start_run(run_name=run_name)


def end_run(status='success'):
    """
    End the current MLflow run with a specified status.
    """
    mlflow.end_run(status=status)


def log_model(model, model_name):
    """
    Log a trained model to MLflow.
    """
    mlflow.sklearn.log_model(model, model_name)


def log_dataframe(df, artifact_path):
    """
    Log a pandas DataFrame as a CSV file to MLflow.
    """
    csv_file = f"{artifact_path}.csv"
    df.to_csv(csv_file, index=False)
    log_artifact(csv_file)
    os.remove(csv_file)  # Remove the CSV file after logging


def get_experiment_id(experiment_name):
    """
    Retrieve the experiment ID for a given experiment name.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")
